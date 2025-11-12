"""Retrieval hardening primitives for the FSB Wayback RAG CLI."""

from __future__ import annotations

import math
import random
import re
from typing import Iterable, List, Sequence, Tuple

STOPWORD_RE = re.compile(r"\b(the|a|an|and|or|но|и|а|на|что|это)\b", re.IGNORECASE)


def subsampled_topk(hits: Sequence[dict], k: int = 8, oversample: int = 2) -> List[dict]:
    """Return a stochastic subset of hits to frustrate MIA probes.

    The caller is expected to pass in ranked hits. We oversample, shuffle, and
    then take the first *k* entries to avoid deterministic leakage while keeping
    quality high.
    """

    if k <= 0:
        return []
    if not hits:
        return []

    oversampled = list(hits[: min(len(hits), k * max(1, oversample))])
    random.SystemRandom().shuffle(oversampled)
    return oversampled[: min(k, len(oversampled))]


def jitter(text: str) -> str:
    """Remove obvious boilerplate/stopwords to reduce deterministic echo."""

    if not text:
        return ""
    cleaned = STOPWORD_RE.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def snippet_only(text: str, max_chars: int = 420) -> str:
    """Trim output to short snippets for disclosure control."""

    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def consensus_overlap(hits_a: Iterable[Tuple[int, float]], hits_b: Iterable[Tuple[int, float]]):
    """Return the overlapping chunk identifiers between two hit sets."""

    ids_a = {hid for hid, _ in hits_a}
    ids_b = {hid for hid, _ in hits_b}
    overlap = sorted(ids_a & ids_b)
    return overlap


def should_answer(overlap_ids: Sequence[int], min_overlap: int = 3, min_chunks: int = 3) -> bool:
    """Decide whether we have enough consensus evidence to answer."""

    overlap_count = len(overlap_ids)
    if overlap_count < min_overlap:
        return False
    # Basic sanity check to avoid extremely small contexts
    return overlap_count >= min_chunks


def masked_sources(hits: Sequence[dict]) -> List[str]:
    """Render masked source strings for display when sources are hidden."""

    masked = []
    for idx, _hit in enumerate(hits, 1):
        masked.append(f"archival-snippet-{idx:02d}")
    return masked


def overlap_ratio(hits_a: Iterable[Tuple[int, float]], hits_b: Iterable[Tuple[int, float]]) -> float:
    """Compute ratio of overlap for telemetry and audits."""

    list_a = list(hits_a)
    list_b = list(hits_b)
    if not list_a or not list_b:
        return 0.0
    overlap = consensus_overlap(list_a, list_b)
    denom = math.sqrt(len(list_a) * len(list_b))
    if denom == 0:
        return 0.0
    return len(overlap) / denom

