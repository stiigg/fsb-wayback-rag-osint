#!/usr/bin/env python3
"""Build a FAISS index from FSB Wayback snapshots with rich observability."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\s\-()]{6,}\d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FAISS index from FSB Wayback JSONL snapshots.",
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL snapshots")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for artefacts")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformers model name",
    )
    parser.add_argument(
        "--index-type",
        default="flat",
        choices=["flat", "ivfflat"],
        help="FAISS index type to build",
    )
    parser.add_argument(
        "--nlist",
        type=int,
        default=4096,
        help="Number of IVF lists when using ivfflat",
    )
    parser.add_argument("--min-chars", type=int, default=50, help="Minimum chars per chunk")
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=1000,
        help="Maximum characters per chunk",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for SentenceTransformers (e.g. cuda, cpu)",
    )
    parser.add_argument(
        "--error-log",
        type=Path,
        help="Optional path for malformed/filtered records JSONL",
    )
    parser.add_argument(
        "--sanity-query",
        action="append",
        default=[],
        help="Queries to run against the index after build",
    )
    parser.add_argument(
        "--sanitize",
        action="store_true",
        help="Enable basic text sanitization",
    )
    parser.add_argument(
        "--redact-pii",
        action="store_true",
        help="Redact detected PII when sanitizing",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (e.g. INFO, DEBUG)",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def build_output_paths(input_path: Path, output_dir: Path, index_type: str) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    hash_prefix = file_hash(input_path)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base_name = f"fsb_index_{index_type}_{timestamp}_{hash_prefix}"
    return {
        "base": output_dir / base_name,
        "index": output_dir / f"{base_name}.faiss",
        "metadata": output_dir / f"{base_name}.metadata.jsonl",
        "manifest": output_dir / f"{base_name}.manifest.json",
        "hash": hash_prefix,
    }


def open_error_log(args: argparse.Namespace, output_paths: Dict[str, Path]):
    path = args.error_log or output_paths["metadata"].with_suffix(".bad.jsonl")
    logging.info("Writing malformed/filtered records to %s", path)
    return path.open("w", encoding="utf-8")


def sanitize_text(text: str, redact_pii: bool = True) -> str:
    cleaned = "".join(ch for ch in text if ch == "\n" or ch >= " ")
    if redact_pii:
        cleaned = EMAIL_RE.sub("[redacted-email]", cleaned)
        cleaned = PHONE_RE.sub("[redacted-phone]", cleaned)
    return cleaned


def sanitize_record(record: Dict[str, Any], redact_pii: bool) -> Dict[str, Any]:
    if "extracted" in record and isinstance(record["extracted"], dict):
        extracted = record["extracted"]
        if isinstance(extracted.get("title"), str):
            extracted["title"] = sanitize_text(extracted["title"], redact_pii=redact_pii)
        if isinstance(extracted.get("text"), str):
            extracted["text"] = sanitize_text(extracted["text"], redact_pii=redact_pii)
        if isinstance(extracted.get("headings"), list):
            for heading in extracted["headings"]:
                if isinstance(heading, dict) and isinstance(heading.get("text"), str):
                    heading["text"] = sanitize_text(heading["text"], redact_pii=redact_pii)
    else:
        if isinstance(record.get("title"), str):
            record["title"] = sanitize_text(record["title"], redact_pii=redact_pii)
        if isinstance(record.get("body"), str):
            record["body"] = sanitize_text(record["body"], redact_pii=redact_pii)
    return record


def load_records(
    input_path: Path,
    min_chars: int,
    sanitize: bool,
    redact_pii: bool,
    bad_out,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    total = malformed = too_short = sanitized_count = 0
    records: List[Dict[str, Any]] = []
    for raw_line in input_path.open("r", encoding="utf-8"):
        total += 1
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:  # pragma: no cover - json errors depend on data
            malformed += 1
            bad_out.write(json.dumps({"line": line, "error": str(exc)}) + "\n")
            continue

        if sanitize:
            sanitize_record(obj, redact_pii=redact_pii)
            sanitized_count += 1

        body_text = extract_body_text(obj)
        if body_text is None or len(body_text.strip()) < min_chars:
            too_short += 1
            bad_out.write(json.dumps({"record": obj, "error": "too_short_body"}, ensure_ascii=False) + "\n")
            continue

        records.append(obj)

    stats = {
        "total_lines": total,
        "malformed": malformed,
        "too_short": too_short,
        "sanitized": sanitized_count,
    }
    logging.info(
        "Loaded %d records (malformed=%d, too_short=%d, sanitized=%d)",
        len(records),
        malformed,
        too_short,
        sanitized_count,
    )
    return records, stats


def extract_body_text(record: Dict[str, Any]) -> Optional[str]:
    extracted = record.get("extracted")
    if isinstance(extracted, dict):
        text = extracted.get("text")
        if isinstance(text, str):
            return text
    body = record.get("body")
    if isinstance(body, str):
        return body
    return None


@dataclass
class Chunk:
    text: str
    title: str
    url: str
    wayback_url: str
    timestamp: str
    date_iso: str
    chunk_idx: int


def make_chunks(record: Dict[str, Any], min_chars: int, max_chars: int) -> List[Chunk]:
    extracted = record.get("extracted") if isinstance(record.get("extracted"), dict) else {}
    title = (extracted.get("title") or record.get("title") or "").strip()
    headings_raw = extracted.get("headings") if isinstance(extracted.get("headings"), list) else []
    headings = [h.get("text", "").strip() for h in headings_raw if isinstance(h, dict) and h.get("text")]
    body = (extract_body_text(record) or "").strip()
    if not body:
        return []

    paragraphs = [p.strip() for p in body.split("\n") if p.strip()]
    chunks: List[Chunk] = []
    buffer: List[str] = []
    current_len = 0

    def flush_buffer() -> None:
        nonlocal buffer, current_len
        if not buffer:
            return
        combined_body = "\n\n".join(buffer)
        if len(combined_body) >= min_chars:
            pieces = [segment for segment in [title] + headings[:2] + [combined_body] if segment]
            text = "\n\n".join(pieces)[:max_chars]
            chunks.append(
                Chunk(
                    text=text,
                    title=title,
                    url=record.get("original_url") or record.get("original") or "",
                    wayback_url=record.get("wayback_url", ""),
                    timestamp=record.get("timestamp", ""),
                    date_iso=record.get("date_iso", ""),
                    chunk_idx=len(chunks),
                )
            )
        buffer = []
        current_len = 0

    for paragraph in paragraphs:
        if current_len + len(paragraph) > max_chars and buffer:
            flush_buffer()
        buffer.append(paragraph)
        current_len += len(paragraph)

    flush_buffer()
    return chunks


def build_embeddings(
    texts: List[str],
    model_name: str,
    batch_size: int,
    device: str,
) -> Tuple[np.ndarray, str, str]:
    logging.info(
        "Encoding %d text chunks with model=%s batch_size=%d",
        len(texts),
        model_name,
        batch_size,
    )
    try:
        model = SentenceTransformer(model_name, device=device)
    except Exception as exc:  # pragma: no cover - depends on environment
        if device != "cpu":
            logging.warning("Falling back to CPU device due to error: %s", exc)
            model = SentenceTransformer(model_name, device="cpu")
        else:
            raise

    embeddings: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        emb = model.encode(
            batch,
            batch_size=min(batch_size, len(batch)),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings.append(np.asarray(emb, dtype="float32"))

    embedding_matrix = np.vstack(embeddings).astype("float32")
    resolved_device = str(model.device)
    logging.info(
        "Built embeddings matrix with shape %s using device=%s",
        embedding_matrix.shape,
        resolved_device,
    )
    return embedding_matrix, model_name, resolved_device


def build_faiss_index(embeddings: np.ndarray, index_type: str, nlist: int) -> faiss.Index:
    dim = embeddings.shape[1]
    if index_type == "flat":
        logging.info("Building IndexFlatIP with %d vectors", embeddings.shape[0])
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index

    if index_type == "ivfflat":
        logging.info("Building IndexIVFFlat with nlist=%d over %d vectors", nlist, embeddings.shape[0])
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        return index

    raise ValueError(f"Unsupported index_type: {index_type}")


def write_metadata(metadata: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logging.info("Wrote metadata for %d chunks to %s", len(metadata), path)


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logging.info("Wrote manifest to %s", path)


def save_faiss_index(index: faiss.Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    logging.info("Saved FAISS index to %s", path)


def sanity_search(
    index: faiss.Index,
    texts: List[str],
    metadata: List[Dict[str, Any]],
    queries: Iterable[str],
    model_name: str,
    device: str,
    top_k: int = 5,
) -> None:
    queries = list(queries)
    if not queries:
        return

    logging.info("Running sanity search for %d queries", len(queries))
    embedder = SentenceTransformer(model_name, device=device)
    for query in queries:
        query_vec = embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)
        distances, indices = index.search(np.asarray(query_vec, dtype="float32"), top_k)
        print(f"\n=== Query: {query} ===")
        for rank, (score, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            if idx < 0 or idx >= len(texts):
                continue
            meta = metadata[idx]
            preview = texts[idx][:200].replace("\n", " ")
            print(
                f"{rank:2d}. score={score:.3f} url={meta.get('original_url','')} "
                f"title={meta.get('title','')[:80]}"
            )
            print("    ", preview, "...")


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    start_time = time.time()
    logging.info("Starting index build for %s", args.input)

    output_paths = build_output_paths(args.input, args.output_dir, args.index_type)

    apply_sanitize = args.sanitize or args.redact_pii
    if apply_sanitize:
        logging.info("Sanitization enabled (redact_pii=%s)", args.redact_pii)

    with open_error_log(args, output_paths) as bad_out:
        records, load_stats = load_records(
            args.input,
            min_chars=args.min_chars,
            sanitize=apply_sanitize,
            redact_pii=args.redact_pii,
            bad_out=bad_out,
        )

    all_texts: List[str] = []
    metadata: List[Dict[str, Any]] = []
    valid_docs = 0

    for doc_idx, record in enumerate(records):
        chunks = make_chunks(record, args.min_chars, args.max_chunk_chars)
        if not chunks:
            continue
        valid_docs += 1
        chunk_count = len(chunks)
        for chunk in chunks:
            all_texts.append(chunk.text)
            metadata.append(
                {
                    "doc_id": doc_idx,
                    "chunk_idx": chunk.chunk_idx,
                    "chunk_count": chunk_count,
                    "original_url": chunk.url,
                    "wayback_url": chunk.wayback_url,
                    "timestamp": chunk.timestamp,
                    "date_iso": chunk.date_iso,
                    "title": chunk.title,
                }
            )

    if not all_texts:
        logging.error("No valid text chunks created; aborting build")
        return 1

    logging.info("Prepared %d chunks from %d documents", len(all_texts), valid_docs)

    embeddings, resolved_model_name, resolved_device = build_embeddings(
        all_texts,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
    )

    index = build_faiss_index(embeddings, args.index_type, args.nlist)

    save_faiss_index(index, output_paths["index"])
    write_metadata(metadata, output_paths["metadata"])

    manifest = {
        "input_file": str(args.input),
        "input_sha256_prefix": output_paths["hash"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name": resolved_model_name,
        "embedding_dim": int(embeddings.shape[1]),
        "num_vectors": int(embeddings.shape[0]),
        "num_chunks": int(len(all_texts)),
        "num_valid_documents": int(valid_docs),
        "index_type": args.index_type,
        "index_path": str(output_paths["index"]),
        "metadata_path": str(output_paths["metadata"]),
        "manifest_path": str(output_paths["manifest"]),
        "index_params": {"nlist": args.nlist if args.index_type == "ivfflat" else None},
        "min_chars": args.min_chars,
        "max_chunk_chars": args.max_chunk_chars,
        "batch_size": args.batch_size,
        "device_requested": args.device,
        "device_used": resolved_device,
        "stats": load_stats,
        "duration_seconds": time.time() - start_time,
    }
    write_manifest(output_paths["manifest"], manifest)

    sanity_search(index, all_texts, metadata, args.sanity_query, args.model, resolved_device)

    logging.info(
        "Completed index build with %d vectors written to %s",
        embeddings.shape[0],
        output_paths["index"],
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
