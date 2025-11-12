from __future__ import annotations


def test_manifest_mismatch_refuses(run_cli_tampered):
    out = run_cli_tampered("What changed?")
    assert "manifest mismatch" in out.lower()
