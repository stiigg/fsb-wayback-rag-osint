from __future__ import annotations


def test_no_overlap_refuses(run_cli_no_overlap):
    out = run_cli_no_overlap("What is policy Y?")
    assert "no overlapping hits" in out.lower()
