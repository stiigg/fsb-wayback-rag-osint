import hashlib
import io
import json
import sys
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsb_rag_cli import execute_cli, parse_args


@pytest.fixture
def sample_metadata(tmp_path: Path) -> dict:
    records: List[dict] = []
    manifest_entries: List[dict] = []
    for doc_id in range(5):
        text = f"Sample archival snippet {doc_id} referencing memo security patterns."
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        embed_hash = hashlib.sha256(f"embed-{doc_id}".encode("utf-8")).hexdigest()
        record = {
            "doc_id": doc_id,
            "chunk_idx": 0,
            "chunk_count": 1,
            "original_url": f"https://fsb.ru/doc/{doc_id}",
            "wayback_url": f"https://web.archive.org/fake/{doc_id}",
            "timestamp": "20200101000000",
            "date_iso": "2020-01-01",
            "title": f"Memo {doc_id}",
            "chunk_text": text,
            "hash_text": digest,
            "hash_raw": digest,
            "hash_embed": embed_hash,
        }
        manifest_entry = {
            "doc_id": doc_id,
            "chunk_idx": 0,
            "hash_text": digest,
            "hash_raw": digest,
            "hash_embed": embed_hash,
            "source_url": record["original_url"],
            "wayback_ts": record["timestamp"],
            "scraper_version": "test-fixture",
        }
        records.append(record)
        manifest_entries.append(manifest_entry)

    meta_path = tmp_path / "meta.jsonl"
    manifest_path = tmp_path / "manifest.jsonl"
    meta_path.write_text("\n".join([json.dumps(r) for r in records]) + "\n", encoding="utf-8")
    manifest_path.write_text("\n".join([json.dumps(m) for m in manifest_entries]) + "\n", encoding="utf-8")
    return {"meta": meta_path, "manifest": manifest_path}


@pytest.fixture
def run_cli(monkeypatch, sample_metadata):
    def _runner(question: str, *extra_args: str) -> str:
        inputs = iter([question, ":q"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        buffer = io.StringIO()
        monkeypatch.setattr(sys, "stdout", buffer)
        args = [
            "--meta",
            str(sample_metadata["meta"]),
            "--manifest",
            str(sample_metadata["manifest"]),
            "--mock-llm",
        ]
        args.extend(extra_args)
        namespace = parse_args(args)
        execute_cli(namespace)
        return buffer.getvalue()

    return _runner


@pytest.fixture
def run_cli_no_overlap(monkeypatch, run_cli):
    import fsb_rag_cli

    monkeypatch.setattr(fsb_rag_cli, "overlap", lambda *_, **__: [])

    def _runner(*args: str) -> str:
        return run_cli(*args)

    return _runner


@pytest.fixture
def run_cli_tampered(monkeypatch, sample_metadata):
    tampered_manifest = sample_metadata["manifest"].with_name("tampered_manifest.jsonl")
    tampered_manifest.write_text("{}\n", encoding="utf-8")

    def _runner(question: str, *extra_args: str) -> str:
        inputs = iter([question, ":q"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        buffer = io.StringIO()
        monkeypatch.setattr(sys, "stdout", buffer)
        args = [
            "--meta",
            str(sample_metadata["meta"]),
            "--manifest",
            str(tampered_manifest),
            "--mock-llm",
        ]
        args.extend(extra_args)
        namespace = parse_args(args)
        execute_cli(namespace)
        return buffer.getvalue()

    return _runner
