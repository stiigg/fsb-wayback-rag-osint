from __future__ import annotations


def test_poison_triggers_refusal(run_cli_poisoned):
    out = run_cli_poisoned("What is policy Y?")
    assert "insufficient cross-index consensus" in out.lower()
