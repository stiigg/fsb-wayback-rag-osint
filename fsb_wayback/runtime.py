"""Runtime hardening primitives for the secure CLI."""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator


@dataclass
class RuntimeConfig:
    read_only: bool = True
    airgapped: bool = True
    no_trace: bool = True
    anti_mia: bool = True
    robust_answer: bool = True
    verify_provenance: bool = True
    qps: float = 0.33
    burst: int = 2


class RateLimiter:
    """Simple token bucket limiter used to throttle answer generation."""

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

    def wrap(self, func: Callable[..., str]) -> Callable[..., str]:
        def wrapped(*args, **kwargs):
            self.acquire()
            return func(*args, **kwargs)

        return wrapped


def _clear_proxies() -> None:
    for key in list(os.environ):
        if key.lower().endswith("_proxy"):
            os.environ.pop(key, None)


def _configure_logging(no_trace: bool) -> None:
    if no_trace:
        logging.getLogger().setLevel(logging.CRITICAL)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@contextmanager
def configure_runtime(config: RuntimeConfig) -> Iterator[RateLimiter]:
    """Apply runtime hardening and yield a QPS limiter."""

    if config.airgapped:
        _clear_proxies()
    _configure_logging(config.no_trace)
    limiter = RateLimiter(config.qps, config.burst)
    try:
        yield limiter
    finally:
        # No teardown required now, placeholder for future hooks
        pass


def derive_runtime_config(args) -> RuntimeConfig:
    return RuntimeConfig(
        read_only=args.read_only,
        airgapped=args.airgapped,
        no_trace=args.no_trace,
        anti_mia=args.anti_mia,
        robust_answer=args.robust_answer,
        verify_provenance=args.verify_provenance,
    )


def refusal_message(reason: str) -> str:
    return f"Refusal: {reason}."


def provenance_failure_message() -> str:
    return refusal_message("provenance verification failed")


def consensus_failure_message() -> str:
    return refusal_message("insufficient cross-index consensus")

