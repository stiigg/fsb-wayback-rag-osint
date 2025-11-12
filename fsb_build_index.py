#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Dict, Any

import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def load_snapshots(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                items.append(obj)
            except json.JSONDecodeError:
                # Skip malformed
                continue
    return items


def make_text(obj: Dict[str, Any]) -> str:
    """Build the text used for embeddings from a snapshot record."""
    extracted = obj.get("extracted") or {}
    title = extracted.get("title") or ""
    headings = extracted.get("headings") or []
    heading_texts = " | ".join(h.get("text", "") for h in headings[:10])
    body = extracted.get("text") or ""

    parts = []
    if title:
        parts.append(title)
    if heading_texts:
        parts.append(heading_texts)
    if body:
        parts.append(body)
    return "\n\n".join(parts)


def build_index(
    snapshots: List[Dict[str, Any]],
    model_name: str,
    batch_size: int = 32,
):
    model = SentenceTransformer(model_name)
    texts = []
    metas: List[Dict[str, Any]] = []

    for obj in snapshots:
        text = make_text(obj)
        if not text or len(text.strip()) < 50:
            # Skip trivially small pages
            continue

        meta = {
            "original_url": obj.get("original_url") or obj.get("original") or "",
            "wayback_url": obj.get("wayback_url", ""),
            "timestamp": obj.get("timestamp", ""),
            "date_iso": obj.get("date_iso", ""),
            "fear_score": obj.get("fear_score", 0),
            "is_change": obj.get("is_change", False),
            "title": (obj.get("extracted") or {}).get("title", ""),
        }
        texts.append(text)
        metas.append(meta)

    if not texts:
        raise RuntimeError("No valid texts found to index.")

    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding snapshots"):
        batch = texts[i:i + batch_size]
        embs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        embeddings.append(embs)

    embs_all = np.vstack(embeddings).astype("float32")
    dim = embs_all.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embs_all)

    return index, metas


def save_index(index, index_path: str) -> None:
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    faiss.write_index(index, index_path)


def save_meta(meta, meta_path: str) -> None:
    os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="Build FAISS index from FSB Wayback JSONL.")
    p.add_argument("--input-jsonl", required=True, help="Input JSONL from scraper/diff pipeline")
    p.add_argument("--index-out", required=True, help="Output FAISS index path")
    p.add_argument("--meta-out", required=True, help="Output metadata JSONL path")
    p.add_argument(
        "--model-name",
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformers model name",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[+] Loading snapshots from {args.input_jsonl}")
    snaps = load_snapshots(args.input_jsonl)
    print(f"[+] Loaded {len(snaps)} raw records")

    print(f"[+] Building index using model={args.model_name}")
    index, meta = build_index(snaps, model_name=args.model_name)

    print(f"[+] Saving FAISS index to {args.index_out}")
    save_index(index, args.index_out)

    print(f"[+] Saving metadata to {args.meta_out} ({len(meta)} entries)")
    save_meta(meta, args.meta_out)

    print("[+] Done.")


if __name__ == "__main__":
    main()
