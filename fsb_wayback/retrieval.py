"""Retrieval stripped to the studs."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


def top_k(hits: Sequence[dict], k: int) -> List[dict]:
    """Return the first k hits and stop whining."""

    if k <= 0:
        return []
    return list(hits[: min(k, len(hits))])


def snippet(text: str, max_chars: int = 320) -> str:
    """Trim a block to something a human can skim."""

    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def overlap(hits_a: Iterable[Tuple[int, float]], hits_b: Iterable[Tuple[int, float]]) -> List[int]:
    """Return shared hit ids without ceremony."""

    ids_a = {hid for hid, _ in hits_a}
    ids_b = {hid for hid, _ in hits_b}
    return sorted(ids_a & ids_b)


__all__ = ["overlap", "snippet", "top_k"]
