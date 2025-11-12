#!/usr/bin/env python3
"""RAG CLI stripped of security theater."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from fsb_wayback.retrieval import overlap, snippet, top_k
from fsb_wayback.utils import RuntimeConfig, load_defaults, runtime_guard, write_audit_report

CONFIG_PATH = Path(__file__).resolve().parent / "fsb_wayback" / "config.yaml"


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
    if not path.exists():
        return items
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
    parser = argparse.ArgumentParser(description="Lean OSINT retrieval shell")
    parser.add_argument("--meta", type=Path, help="Metadata JSONL with chunk_text", default=None)
    parser.add_argument("--manifest", type=Path, help="Manifest JSONL to sanity check chunks", default=None)
    parser.add_argument("--top-k", type=int, default=None, help="Chunks per question")
    parser.add_argument("--mock-llm", action="store_true", help="Offline summary mode")
    parser.add_argument("--reveal-sources", action="store_true", help="Show raw URLs")
    parser.add_argument("--audit", action="store_true", help="Emit audit report and exit")
    parser.add_argument("--no-airgap", action="store_true", help="Keep proxies instead of purging them")
    parser.add_argument("--qps", type=float, default=0.5, help="Rate limit per second")
    parser.add_argument("--burst", type=int, default=2, help="Burst size for the limiter")
    return parser.parse_args(argv)


def resolve_runtime(args: argparse.Namespace) -> RuntimeConfig:
    airgapped = not args.no_airgap
    return RuntimeConfig(airgapped=airgapped, qps=args.qps, burst=args.burst)


def refusal(reason: str) -> str:
    return f"Refused: {reason}."


def _summarise_offline(chunks: Sequence[Dict[str, Any]]) -> str:
    highlights = []
    for meta in chunks:
        highlights.append(f"[{meta.get('date_iso') or meta.get('timestamp')}] {snippet(meta.get('chunk_text', ''))}")
    return "\n".join(highlights) or "No context available."


def build_answer(*, query: str, hits: Sequence[Dict[str, Any]], args: argparse.Namespace) -> str:
    if args.mock_llm:
        return _summarise_offline(hits)
    return refusal("LLM disabled")


def verify_chunks(*, hits: Sequence[Dict[str, Any]], manifest_index: Dict[Tuple[int, int], Dict[str, Any]]) -> bool:
    if not manifest_index:
        return True
    for meta in hits:
        key = (meta.get("doc_id"), meta.get("chunk_idx"))
        manifest_entry = manifest_index.get(key)
        if not manifest_entry:
            return False
        expected = manifest_entry.get("hash_text")
        actual = meta.get("hash_text")
        if expected and actual and expected != actual:
            return False
    return True


def format_sources(hits: Sequence[Dict[str, Any]], reveal: bool) -> List[str]:
    if reveal:
        return [meta.get("wayback_url") or meta.get("original_url") or "unknown" for meta in hits]
    return [f"archival-snippet-{idx:02d}" for idx, _meta in enumerate(hits, 1)]


def apply_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = load_defaults(CONFIG_PATH)
    meta_env = os.getenv("FSB_METADATA")
    manifest_env = os.getenv("FSB_MANIFEST")
    topk_env = os.getenv("FSB_TOP_K")

    if args.meta is None:
        path = meta_env or defaults.get("metadata_path")
        if path:
            args.meta = Path(path)
    if args.manifest is None:
        path = manifest_env or defaults.get("manifest_path")
        if path:
            args.manifest = Path(path)
    if args.top_k is None:
        value = topk_env or defaults.get("max_hits")
        if value is not None:
            args.top_k = int(value)
    if args.top_k is None:
        args.top_k = 6
    return args


def execute_cli(args: argparse.Namespace) -> str:
    args = apply_defaults(args)
    if args.meta is None:
        return refusal("metadata path missing")

    metadata = load_jsonl(args.meta)
    if not metadata:
        return refusal("metadata not available")
    manifest_index = load_manifest_index(args.manifest) if args.manifest else {}

    retriever_a = DualIndex(metadata, seed=1)
    retriever_b = DualIndex(metadata, seed=2)

    runtime_config = resolve_runtime(args)
    outputs: List[str] = []
    overlap_trace: List[int] = []

    with runtime_guard(runtime_config) as limiter:
        print("[+] Wayback RAG ready. :q quits.")
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
            shared = overlap(hits_a, hits_b)
            overlap_trace.append(len(shared))
            if not shared:
                refusal_msg = refusal("no overlapping hits")
                print(refusal_msg)
                outputs.append(refusal_msg)
                continue

            indices = [idx for idx, _score in hits_a if idx in shared]
            selected = [metadata[i].copy() for i in indices]
            selected = top_k(selected, args.top_k)
            for meta in selected:
                meta["chunk_text"] = snippet(meta.get("chunk_text", ""))

            if not verify_chunks(hits=selected, manifest_index=manifest_index):
                refusal_msg = refusal("manifest mismatch")
                print(refusal_msg)
                outputs.append(refusal_msg)
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
            "airgapped": runtime_config.airgapped,
            "qps": runtime_config.qps,
            "burst": runtime_config.burst,
        }
        kb_stats = {"indexed": len(metadata)}
        write_audit_report(
            flags=flags,
            manifest_path=args.manifest or Path("manifest.jsonl"),
            kb_stats=kb_stats,
            overlap_trace=overlap_trace,
        )
    return "\n".join(outputs)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    execute_cli(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
