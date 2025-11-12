"""Auditor tooling for the secure FSB Wayback CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _read_manifest(manifest_path: Path) -> Optional[Dict[str, Any]]:
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]
    if not lines:
        return None
    latest = lines[-1]
    return latest


def generate_audit_report(
    *,
    flags: Dict[str, Any],
    manifest_path: Path,
    kb_stats: Dict[str, Any],
    consensus_trace: Optional[Iterable[float]] = None,
    redteam_status_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compile an auditor friendly report."""

    manifest = _read_manifest(manifest_path)
    redteam_status = None
    if redteam_status_path and redteam_status_path.exists():
        redteam_status = redteam_status_path.read_text(encoding="utf-8").strip()

    report = {
        "active_flags": flags,
        "knowledge_base": kb_stats,
        "manifest": manifest or {},
        "consensus_trace": list(consensus_trace or []),
        "redteam_last_run": redteam_status or "unknown",
    }
    output_path = Path("audit_report.json")
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


__all__ = ["generate_audit_report"]

