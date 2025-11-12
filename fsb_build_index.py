#!/usr/bin/env python3
"""Build a FAISS index from FSB Wayback snapshots with rich observability."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\s\-()]{6,}\d")


class ExtractedPayload(BaseModel):
    title: Optional[str] = None
    text: str = Field(default="", min_length=1)
    headings: Optional[List[Dict[str, Any]]] = None

    model_config = {"extra": "allow"}

    @field_validator("text")
    @classmethod
    def trim_text(cls, value: str) -> str:
        return value.strip()


class SnapshotRecord(BaseModel):
    original_url: str = Field(default="")
    wayback_url: Optional[str] = None
    timestamp: Optional[str] = None
    date_iso: Optional[str] = None
    extracted: ExtractedPayload
    language: Optional[str] = Field(default=None, alias="lang")

    model_config = {"extra": "allow", "populate_by_name": True}

    @field_validator("original_url")
    @classmethod
    def require_url(cls, value: str) -> str:
        return value.strip()


class IngestPolicy(BaseModel):
    allowed_url_regex: List[str]
    date_window: Dict[str, str]
    sha256_allowlist: List[str]
    min_chars: int
    max_boilerplate_pct: float
    lang_allowlist: List[str]
    dedup: Dict[str, Any]
    entity_outlier: Dict[str, Any]
    quarantine_bucket: str

    model_config = {"extra": "allow"}


def load_ingest_policy(policy_path: Path) -> IngestPolicy:
    raw = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    return IngestPolicy.model_validate(raw)


def _matches_regex(patterns: List[str], value: str) -> bool:
    return any(re.compile(p).search(value) for p in patterns)


def _parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d%H%M%S", "%Y%m%d"):
        try:
            return datetime.strptime(value[: len(fmt)], fmt)
        except ValueError:
            continue
    return None


def _estimate_lang(text: str) -> str:
    cyrillic = sum(1 for ch in text if "\u0400" <= ch <= "\u04FF")
    latin = sum(1 for ch in text if "A" <= ch <= "z")
    if cyrillic > latin:
        return "ru"
    return "en"


def _boilerplate_ratio(text: str) -> float:
    words = [w.lower() for w in re.findall(r"\w+", text)]
    if not words:
        return 1.0
    unique = len(set(words))
    return 1 - (unique / len(words))


def enforce_policy(
    record: SnapshotRecord,
    policy: IngestPolicy,
    seen_hashes: Dict[str, str],
) -> Optional[str]:
    url = record.original_url
    if not url or not _matches_regex(policy.allowed_url_regex, url):
        return "url_out_of_scope"

    window = policy.date_window
    since = _parse_date(window.get("since"))
    until = _parse_date(window.get("until"))
    stamp = _parse_date(record.date_iso or record.timestamp)
    if stamp is None or (since and stamp < since) or (until and stamp > until):
        return "timestamp_out_of_scope"

    text = record.extracted.text
    if len(text) < policy.min_chars:
        return "too_short"

    if _boilerplate_ratio(text) > policy.max_boilerplate_pct:
        return "too_boilerplate"

    lang = record.language or _estimate_lang(text)
    if policy.lang_allowlist and lang.lower() not in {code.lower() for code in policy.lang_allowlist}:
        return "lang_not_allowed"

    sha_prefix = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if policy.sha256_allowlist and sha_prefix not in policy.sha256_allowlist:
        return "hash_not_allowlisted"

    dedup_key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    if dedup_key in seen_hashes:
        return "duplicate_content"
    seen_hashes[dedup_key] = url

    return None


def quarantine_record(record: Dict[str, Any], quarantine_dir: Path, reason: str) -> None:
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    payload = {"record": record, "reason": reason}
    path = quarantine_dir / f"quarantine_{int(time.time()*1000)}_{random.randint(0, 9999):04d}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("fsb_wayback/ingest_policy.yaml"),
        help="Path to zero-trust ingest policy",
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
    policy: IngestPolicy,
    sanitize: bool,
    redact_pii: bool,
    bad_out,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    total = malformed = sanitized_count = 0
    policy_failures: Dict[str, int] = {}
    records: List[Dict[str, Any]] = []
    seen_hashes: Dict[str, str] = {}
    quarantine_dir = Path(policy.quarantine_bucket)
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

        try:
            record_model = SnapshotRecord.model_validate(obj)
        except ValidationError as exc:
            reason = "schema_validation_failed"
            policy_failures[reason] = policy_failures.get(reason, 0) + 1
            quarantine_record({"record": obj, "error": exc.errors()}, quarantine_dir, reason)
            continue

        failure = enforce_policy(record_model, policy, seen_hashes)
        if failure:
            policy_failures[failure] = policy_failures.get(failure, 0) + 1
            quarantine_record(record_model.model_dump(mode="python"), quarantine_dir, failure)
            continue

        records.append(record_model.model_dump(mode="python"))

    stats = {
        "total_lines": total,
        "malformed": malformed,
        "policy_failures": policy_failures,
        "sanitized": sanitized_count,
    }
    logging.info(
        "Loaded %d records (malformed=%d, policy_failures=%s, sanitized=%d)",
        len(records),
        malformed,
        policy_failures,
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


def append_merkle_manifest(entries: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logging.info("Appended %d entries to %s", len(entries), path)


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

    policy = load_ingest_policy(args.policy)
    logging.info("Loaded ingest policy from %s", args.policy)

    with open_error_log(args, output_paths) as bad_out:
        records, load_stats = load_records(
            args.input,
            policy=policy,
            sanitize=apply_sanitize,
            redact_pii=args.redact_pii,
            bad_out=bad_out,
        )

    all_texts: List[str] = []
    metadata: List[Dict[str, Any]] = []
    chunk_manifest_entries: List[Dict[str, Any]] = []
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
                    "chunk_text": chunk.text,
                    "hash_text": None,
                    "hash_raw": None,
                    "hash_embed": None,
                }
            )
            raw_body = extract_body_text(record) or ""
            chunk_manifest_entries.append(
                {
                    "doc_id": doc_idx,
                    "chunk_idx": chunk.chunk_idx,
                    "hash_raw": hashlib.sha256(raw_body.encode("utf-8")).hexdigest(),
                    "hash_text": hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
                    "hash_embed": None,
                    "source_url": chunk.url,
                    "wayback_ts": chunk.timestamp or chunk.date_iso,
                    "scraper_version": record.get("scraper_version", "unknown"),
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

    for idx, vector in enumerate(embeddings):
        chunk_manifest_entries[idx]["hash_embed"] = hashlib.sha256(vector.tobytes()).hexdigest()
        metadata[idx]["hash_embed"] = chunk_manifest_entries[idx]["hash_embed"]
        metadata[idx]["hash_text"] = chunk_manifest_entries[idx]["hash_text"]
        metadata[idx]["hash_raw"] = chunk_manifest_entries[idx]["hash_raw"]

    merkle_concat = "".join(entry["hash_embed"] for entry in chunk_manifest_entries)
    merkle_root = hashlib.sha256(merkle_concat.encode("utf-8")).hexdigest()

    index = build_faiss_index(embeddings, args.index_type, args.nlist)

    save_faiss_index(index, output_paths["index"])
    write_metadata(metadata, output_paths["metadata"])
    append_merkle_manifest(chunk_manifest_entries, Path("merkle/manifest.jsonl"))

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
        "merkle_root": merkle_root,
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
