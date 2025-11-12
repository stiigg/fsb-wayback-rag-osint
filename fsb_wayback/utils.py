"""Runtime hacks and audit crumbs."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional


@dataclass
class RuntimeConfig:
    """Bare runtime knobs that actually matter."""

    airgapped: bool = True
    quiet: bool = True
    qps: float = 0.5
    burst: int = 2


class RateLimiter:
    """Token bucket that says no when you spam."""

    def __init__(self, qps: float, burst: int) -> None:
        self.qps = max(qps, 0.0001)
        self.burst = max(1, burst)
        self._tokens = float(self.burst)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        delta = now - self._last
        self._last = now
        self._tokens = min(self.burst, self._tokens + delta * self.qps)

    def acquire(self) -> None:
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                needed = (1.0 - self._tokens) / self.qps
            time.sleep(max(needed, 0.01))


def _strip_proxies() -> None:
    for key in list(os.environ):
        if key.lower().endswith("_proxy"):
            os.environ.pop(key, None)


def _mute_logs() -> None:
    logging.getLogger().setLevel(logging.CRITICAL)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@contextmanager
def runtime_guard(config: RuntimeConfig) -> Iterator[RateLimiter]:
    """Apply the bare minimum runtime duct tape."""

    if config.airgapped:
        _strip_proxies()
    if config.quiet:
        _mute_logs()
    limiter = RateLimiter(config.qps, config.burst)
    yield limiter


def _read_latest_manifest(manifest_path: Path) -> Optional[Dict[str, Any]]:
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]
    if not lines:
        return None
    return lines[-1]


def write_audit_report(
    *,
    flags: Dict[str, Any],
    manifest_path: Path,
    kb_stats: Dict[str, Any],
    overlap_trace: Iterable[int] = (),
    output_path: Path | None = None,
) -> Dict[str, Any]:
    """Dump a blunt audit bundle for whoever insists."""

    manifest = _read_latest_manifest(manifest_path) or {}
    report = {
        "flags": flags,
        "knowledge_base": kb_stats,
        "manifest": manifest,
        "overlap_trace": list(overlap_trace),
    }
    target = output_path or Path("audit_report.json")
    target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def load_defaults(config_path: Path) -> Dict[str, Any]:
    """Load the simple config blob or fall back to emptiness."""

    if not config_path.exists():
        return {}
    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        return {}
    return data


__all__ = [
    "RateLimiter",
    "RuntimeConfig",
    "load_defaults",
    "runtime_guard",
    "write_audit_report",
]
