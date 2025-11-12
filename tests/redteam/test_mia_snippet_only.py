from __future__ import annotations


def test_snippet_only_hides_sources(run_cli):
    out = run_cli("Summarize the canary memo X.")
    assert "fsb.ru" not in out.lower()
    assert "doc_id" not in out.lower()
