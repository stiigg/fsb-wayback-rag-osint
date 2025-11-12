from __future__ import annotations


def test_provenance_mismatch_refuses(run_cli_tampered):
    out = run_cli_tampered("What changed?")
    assert "provenance verification failed" in out.lower()
