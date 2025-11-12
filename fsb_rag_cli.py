#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import openai


def load_meta(meta_path: str) -> List[Dict[str, Any]]:
    items = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                items.append(obj)
            except json.JSONDecodeError:
                continue
    return items


def load_index(index_path: str):
    return faiss.read_index(index_path)


def embed_model(model_name: str):
    model = SentenceTransformer(model_name)

    def _embed(text: str) -> np.ndarray:
        emb = model.encode([text], normalize_embeddings=True)
        return emb.astype("float32")
    return _embed


def build_context(hit_metas: List[Dict[str, Any]]) -> str:
    blocks = []
    for m in hit_metas:
        title = m.get("title") or ""
        url = m.get("original_url") or ""
        wb = m.get("wayback_url") or ""
        ts = m.get("date_iso") or m.get("timestamp") or ""
        fear = m.get("fear_score", 0)
        ch = m.get("is_change", False)
        block = (
            f"[{ts} | fear={fear} | change={ch}]
"
            f"URL: {url}
"
            f"Wayback: {wb}
"
            f"TITLE: {title}
"
        )
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)


def call_llm(openai_model: str, system_prompt: str, context: str, question: str) -> str:
    client = openai.OpenAI()
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": (
                "ARCHIVE CONTEXT:\n"
                + context
                + "\n\nQUESTION:\n"
                + question
            ),
        },
    ]
    resp = client.chat.completions.create(
        model=openai_model,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def parse_args():
    p = argparse.ArgumentParser(description="Interactive RAG CLI over FSB Wayback archive.")
    p.add_argument("--index", required=True, help="FAISS index file")
    p.add_argument("--meta", required=True, help="Metadata JSONL file")
    p.add_argument(
        "--model-name",
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformers model name",
    )
    p.add_argument(
        "--openai-model",
        default="gpt-4o-mini",
        help="OpenAI (or compatible) chat model",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top documents to retrieve for context",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("[!] OPENAI_API_KEY environment variable is not set.")
        print("    Set it before running, e.g.: export OPENAI_API_KEY='sk-...'")
        return

    print(f"[+] Loading FAISS index from {args.index}")
    index = load_index(args.index)

    print(f"[+] Loading metadata from {args.meta}")
    metas = load_meta(args.meta)

    if index.ntotal != len(metas):
        print(f"[!] Index size ({index.ntotal}) != metadata length ({len(metas)}).")
        print("    Make sure you used the same files when building.")
        return

    print(f"[+] Loading embedding model {args.model_name}")
    embed = embed_model(args.model_name)

    system_prompt = (
        "You are a digital forensics and OSINT analyst specialising in Russian "
        "state security narratives. You are given snippets of metadata from archived "
        "fsb.ru pages (timestamps, URLs, titles, some scores). Your tasks:\n"
        "1. Answer the user's question using ONLY what can be inferred from the context.\n"
        "2. Focus on narrative shifts, fear/threat framing, and semantic changes over time.\n"
        "3. When you refer to evidence, explicitly mention timestamps and URLs.\n"
        "4. If the context is insufficient, say so clearly and suggest what additional "
        "data would be needed.\n"
    )

    print("[+] RAG CLI ready. Type a question, or :q to quit.")
    while True:
        try:
            q = input("\n[?] Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[+] Bye.")
            break

        if not q:
            continue
        if q.lower() in {":q", ":quit", ":exit"}:
            print("[+] Exiting.")
            break

        # Embed query
        q_emb = embed(q)
        D, I = index.search(q_emb, args.top_k)
        idxs = I[0].tolist()

        hit_metas = [metas[i] for i in idxs if 0 <= i < len(metas)]
        context = build_context(hit_metas)

        print("\n[+] Retrieved context from top-k snapshots. Querying LLM...")
        answer = call_llm(args.openai_model, system_prompt, context, q)

        print("\n=== ANSWER ===")
        print(answer)
        print("\n=== SOURCES ===")
        for m in hit_metas:
            print(f"- {m.get('date_iso') or m.get('timestamp')} :: {m.get('wayback_url')}")


if __name__ == "__main__":
    main()
