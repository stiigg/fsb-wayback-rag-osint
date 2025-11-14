import json
import requests

from fsb_wayback import scraper


def test_iso_date_from_timestamp():
    assert scraper.iso_date_from_timestamp("20240102112233") == "2024-01-02"
    assert scraper.iso_date_from_timestamp("bogus") is None


def test_extract_payload_fallback(monkeypatch):
    html = """
    <html>
      <head><title>Archive</title></head>
      <body>
        <h1>Main Heading</h1>
        <p>Body content</p>
      </body>
    </html>
    """
    monkeypatch.setattr(scraper.trafilatura, "extract", lambda *_, **__: None)
    payload = scraper.extract_payload(html)
    assert payload["title"] == "Archive"
    assert "Body content" in payload["text"]
    assert payload["headings"] == [{"tag": "h1", "text": "Main Heading"}]


class FakeResponse:
    def __init__(self, *, status_code=200, json_data=None, text="", headers=None):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text
        self.headers = headers or {}

    def json(self):
        if self._json_data is None:
            raise ValueError("No JSON data")
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requested_urls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, timeout=30):
        self.requested_urls.append(url)
        if not self._responses:
            raise AssertionError("No more fake responses available")
        return self._responses.pop(0)


class DummyLimiter:
    def acquire(self):
        return None


def fake_runtime_guard(_config):
    class _Guard:
        def __enter__(self_inner):
            return DummyLimiter()

        def __exit__(self_inner, exc_type, exc, tb):
            return False

    return _Guard()


def test_run_scraper_produces_snapshot(monkeypatch, tmp_path):
    cdx_data = [
        ["urlkey", "timestamp", "original", "mimetype", "statuscode"],
        [
            "ru,fsb)/page",
            "20200101120000",
            "https://fsb.ru/page",
            "text/html",
            "200",
        ],
    ]

    def fake_requests_get(url, params=None, timeout=30):
        assert url == scraper.CDX_ENDPOINT
        return FakeResponse(json_data=cdx_data)

    html = """
    <html>
      <head><title>FSB Snapshot</title></head>
      <body>
        <h1>Header</h1>
        <p>Archive body text</p>
      </body>
    </html>
    """
    html_response = FakeResponse(
        text=html,
        headers={"Content-Type": "text/html; charset=utf-8"},
    )

    monkeypatch.setattr(scraper.requests, "get", fake_requests_get)
    monkeypatch.setattr(scraper.requests, "Session", lambda: FakeSession([html_response]))
    monkeypatch.setattr(scraper, "runtime_guard", fake_runtime_guard)
    monkeypatch.setattr(scraper, "current_git_commit", lambda: "deadbeef")
    monkeypatch.setattr(
        scraper.trafilatura,
        "extract",
        lambda *_args, **_kwargs: json.dumps({"title": "FSB Snapshot", "text": "Archive body text"}),
    )

    output_path = tmp_path / "snapshots.jsonl"
    manifest_path = tmp_path / "manifest.jsonl"
    audit_path = tmp_path / "audit.json"

    args = scraper.parse_args(
        [
            "--domain",
            "fsb.ru",
            "--from-year",
            "2020",
            "--to-year",
            "2020",
            "--output",
            str(output_path),
            "--manifest",
            str(manifest_path),
            "--audit-report",
            str(audit_path),
            "--limit",
            "1",
            "--qps",
            "10",
        ]
    )

    result = scraper.run_scraper(args)

    lines = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
    record = lines[0]
    assert record["original_url"] == "https://fsb.ru/page"
    assert record["wayback_url"].startswith("https://web.archive.org/web/20200101120000id_/")
    assert record["extracted"]["text"] == "Archive body text"

    manifest_lines = manifest_path.read_text(encoding="utf-8").splitlines()
    assert len(manifest_lines) == 1
    manifest_entry = json.loads(manifest_lines[0])
    assert manifest_entry["hashes"]["config_sha256"]
    assert manifest_entry["stats"]["snapshots_extracted"] == 1

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["knowledge_base"]["records"] == 1

    assert result["records"] == 1
    assert result["stats"]["snapshots_extracted"] == 1
