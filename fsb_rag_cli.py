#!/usr/bin/env python3
"""Secure RAG CLI with consensus and provenance enforcement."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from fsb_wayback.audit_tools import generate_audit_report
from fsb_wayback.retrieval_policies import (
    consensus_overlap,
    jitter,
    masked_sources,
    should_answer,
    snippet_only,
    subsampled_topk,
)
from fsb_wayback.runtime import (
    configure_runtime,
    consensus_failure_message,
    derive_runtime_config,
    provenance_failure_message,
    refusal_message,
)


@dataclass
class DualIndex:
    metadata: Sequence[Dict[str, Any]]
    seed: int

    def search(self, query: str, top_k: int = 16) -> List[Tuple[int, float]]:
        scores: List[Tuple[int, float]] = []
        for idx, _meta in enumerate(self.metadata):
            base = 1.0 / (1.0 + idx)
            jitter_val = ((hash((self.seed, idx, query)) % 1000) / 1000.0) * 0.05
            score = max(0.0, base - jitter_val)
            scores.append((idx, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def load_manifest_index(path: Path) -> Dict[Tuple[int, int], Dict[str, Any]]:
    if not path.exists():
        return {}
    entries = load_jsonl(path)
    index: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for entry in entries:
        key = (entry.get("doc_id"), entry.get("chunk_idx"))
        index[key] = entry
    return index


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Secure OSINT RAG CLI")
    parser.add_argument("--meta", type=Path, required=True, help="Metadata JSONL with chunk_text")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("merkle/manifest.jsonl"),
        help="Merkle manifest JSONL",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Maximum chunks to surface")
    parser.add_argument("--mock-llm", action="store_true", help="Use offline summariser")
    parser.add_argument("--reveal-sources", action="store_true", help="Show raw source IDs")
    parser.add_argument("--audit", action="store_true", help="Emit audit report and exit")
    for flag in ("read_only", "airgapped", "no_trace", "anti_mia", "robust_answer", "verify_provenance"):
        parser.add_argument(
            f"--{flag.replace('_', '-')}",
            default=True,
            action=argparse.BooleanOptionalAction,
            help=f"Toggle {flag.replace('_', ' ')} mode",
        )
    return parser.parse_args(argv)


def _summarise_offline(chunks: Sequence[Dict[str, Any]], question: str) -> str:
    highlights = []
    for meta in chunks:
        snippet = meta.get("chunk_text", "")
        snippet = snippet_only(jitter(snippet))
        highlights.append(f"[{meta.get('date_iso') or meta.get('timestamp')}] {snippet}")
    answer = "\n".join(highlights) or "No context available."
    return answer


def build_answer(
    *,
    query: str,
    hits: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    if args.mock_llm:
        return _summarise_offline(hits, query)
    return refusal_message("offline mode enforced; no LLM available")


def verify_chunks(
    *,
    hits: Sequence[Dict[str, Any]],
    manifest_index: Dict[Tuple[int, int], Dict[str, Any]],
) -> bool:
    for meta in hits:
        key = (meta.get("doc_id"), meta.get("chunk_idx"))
        manifest_entry = manifest_index.get(key)
        if not manifest_entry:
            return False
        expected = manifest_entry.get("hash_text")
        if not expected:
            return False
        actual = meta.get("hash_text")
        if actual != expected:
            return False
        if manifest_entry.get("hash_embed") != meta.get("hash_embed"):
            return False
    return True


def format_sources(hits: Sequence[Dict[str, Any]], reveal: bool) -> List[str]:
    if reveal:
        return [meta.get("wayback_url") or meta.get("original_url") or "unknown" for meta in hits]
    return masked_sources(hits)


def execute_cli(args: argparse.Namespace) -> str:
    metadata = load_jsonl(args.meta)
    if not metadata:
        return refusal_message("metadata not available")
    manifest_index = load_manifest_index(args.manifest)

    retriever_a = DualIndex(metadata, seed=1)
    retriever_b = DualIndex(metadata, seed=2)

    runtime_config = derive_runtime_config(args)
    outputs: List[str] = []
    consensus_trace: List[float] = []

    with configure_runtime(runtime_config) as limiter:
        print("[+] Secure RAG CLI ready. Type a question, or :q to quit.")
        while True:
            try:
                question = input("\n[?] Question> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[+] Bye.")
                break
            if not question:
                continue
            if question.lower() in {":q", ":quit", ":exit"}:
                print("[+] Exiting.")
                break

            limiter.acquire()
            hits_a = retriever_a.search(question, top_k=args.top_k * 2)
            hits_b = retriever_b.search(question, top_k=args.top_k * 2)
            overlap_ids = consensus_overlap(hits_a, hits_b)
            consensus_trace.append(len(overlap_ids))
            if args.robust_answer and not should_answer(overlap_ids):
                refusal = consensus_failure_message()
                print(refusal)
                outputs.append(refusal)
                continue

            indices = [idx for idx, _score in hits_a if idx in overlap_ids]
            selected = [metadata[i].copy() for i in indices[: args.top_k]]
            if args.anti_mia:
                selected = subsampled_topk(selected, k=args.top_k)
                for meta in selected:
                    meta["chunk_text"] = jitter(meta.get("chunk_text", ""))
            selected = [
                {
                    **meta,
                    "chunk_text": snippet_only(meta.get("chunk_text", "")),
                }
                for meta in selected
            ]

            if args.verify_provenance and not verify_chunks(hits=selected, manifest_index=manifest_index):
                refusal = provenance_failure_message()
                print(refusal)
                outputs.append(refusal)
                continue

            answer = build_answer(query=question, hits=selected, args=args)
            print("\n=== ANSWER ===")
            print(answer)
            print("\n=== SOURCES ===")
            for source in format_sources(selected, args.reveal_sources):
                print(f"- {source}")
            outputs.append(answer)

    if args.audit:
        flags = {
            "read_only": args.read_only,
            "airgapped": args.airgapped,
            "no_trace": args.no_trace,
            "anti_mia": args.anti_mia,
            "robust_answer": args.robust_answer,
            "verify_provenance": args.verify_provenance,
        }
        kb_stats = {"indexed": len(metadata), "quarantined": 0}
        generate_audit_report(
            flags=flags,
            manifest_path=args.manifest,
            kb_stats=kb_stats,
            consensus_trace=consensus_trace,
            redteam_status_path=Path("tests/redteam/.last_run"),
        )
    return "\n".join(outputs)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    execute_cli(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
