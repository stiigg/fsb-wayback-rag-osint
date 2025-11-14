"""Wayback scraper that feeds the ingestion pipeline with receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests

try:  # pragma: no cover - exercised in integration scenarios
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - fallback logic tested separately
    BeautifulSoup = None  # type: ignore

try:  # pragma: no cover - exercised in integration scenarios
    import trafilatura  # type: ignore
except ImportError:  # pragma: no cover - provide a stub for offline environments
    class _TrafilaturaStub:  # pragma: no cover - behaviour asserted via tests
        @staticmethod
        def extract(*_args, **_kwargs) -> None:
            return None

    trafilatura = _TrafilaturaStub()  # type: ignore

from .utils import RuntimeConfig, runtime_guard, write_audit_report

CDX_ENDPOINT = "https://web.archive.org/cdx/search/cdx"

logger = logging.getLogger(__name__)


@dataclass
class CdxRow:
    """Minimal CDX row we actually care about."""

    timestamp: str
    original_url: str
    statuscode: str
    mimetype: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Wayback snapshots into SnapshotRecord JSONL.",
    )
    parser.add_argument("--domain", required=True, help="Base domain, e.g. fsb.ru")
    parser.add_argument("--from-year", type=int, default=2010)
    parser.add_argument("--to-year", type=int, default=datetime.utcnow().year)
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/fsb_snapshots_manifest.jsonl"),
        help="Manifest JSONL tracking scraper runs.",
    )
    parser.add_argument(
        "--audit-report",
        type=Path,
        default=Path("scraper_audit.json"),
        help="Path for the generated audit report JSON.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Cap CDX rows (0 = no cap)")
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Keep every Nth CDX row after filtering (1 = keep all).",
    )
    parser.add_argument(
        "--qps",
        type=float,
        default=0.2,
        help="Wayback fetches per second (respect the archive).",
    )
    parser.add_argument(
        "--burst",
        type=int,
        default=1,
        help="Token bucket burst size for the rate limiter.",
    )
    parser.add_argument(
        "--match-type",
        choices=["prefix", "domain"],
        default="prefix",
        help="Wayback matchType for CDX queries.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity for the scraper run.",
    )
    return parser.parse_args(argv)


def cdx_query_params(domain: str, since: int, until: int, match_type: str) -> Dict[str, Any]:
    url = domain
    if match_type == "prefix" and not domain.endswith("/*"):
        url = f"{domain}/*"
    params: Dict[str, Any] = {
        "url": url,
        "output": "json",
        "filter": ["statuscode:200", "mimetype:text/html"],
        "collapse": "digest",
        "from": str(since),
        "to": str(until),
    }
    if match_type == "domain":
        params["matchType"] = "domain"
    return params


def fetch_cdx_rows(params: Dict[str, Any], limit: int = 0) -> List[CdxRow]:
    resp = requests.get(CDX_ENDPOINT, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return []
    header, *rows = data
    idx_map = {name: header.index(name) for name in ("timestamp", "original", "statuscode", "mimetype")}
    results: List[CdxRow] = []
    for row in rows:
        results.append(
            CdxRow(
                timestamp=row[idx_map["timestamp"]],
                original_url=row[idx_map["original"]],
                statuscode=row[idx_map["statuscode"]],
                mimetype=row[idx_map["mimetype"]],
            )
        )
        if limit and len(results) >= limit:
            break
    return results


def wayback_url(timestamp: str, original_url: str) -> str:
    return f"https://web.archive.org/web/{timestamp}id_/{original_url}"


def iso_date_from_timestamp(timestamp: str) -> Optional[str]:
    if not timestamp:
        return None
    try:
        dt = datetime.strptime(timestamp[:8], "%Y%m%d")
    except ValueError:
        return None
    return dt.date().isoformat()


def fetch_html(session: requests.Session, url: str, limiter) -> Optional[str]:
    limiter.acquire()
    resp = session.get(url, timeout=30)
    if resp.status_code != 200:
        logger.debug("Skipping %s due to HTTP %s", url, resp.status_code)
        return None
    ctype = resp.headers.get("Content-Type", "")
    if "text/html" not in ctype:
        logger.debug("Skipping %s due to content type %s", url, ctype)
        return None
    return resp.text


def _extract_with_trafilatura(html: str) -> Optional[Dict[str, Any]]:
    if not hasattr(trafilatura, "extract"):
        return None
    extracted = trafilatura.extract(html, output="json", favor_recall=True)
    if not extracted:
        return None
    try:
        payload = json.loads(extracted)
    except json.JSONDecodeError:
        return None
    text = (payload.get("text") or "").strip()
    title = (payload.get("title") or "").strip() or None
    if not text:
        return None
    headings: Optional[Iterable[Dict[str, Any]]] = payload.get("headings")
    if headings:
        cleaned = []
        for entry in headings:
            tag = entry.get("tag")
            text_value = (entry.get("text") or "").strip()
            if tag and text_value:
                cleaned.append({"tag": tag, "text": text_value})
        headings = cleaned or None
    return {"title": title, "text": text, "headings": list(headings) if headings else None}


class _SimpleHTMLExtractor:
    """Lightweight fallback when BeautifulSoup is unavailable."""

    def __init__(self) -> None:
        from html.parser import HTMLParser

        class _Parser(HTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self.title_parts: List[str] = []
                self.text_parts: List[str] = []
                self.headings: List[Dict[str, str]] = []
                self._in_title = False
                self._heading_tag: Optional[str] = None
                self._heading_parts: List[str] = []

            def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
                if tag.lower() == "title":
                    self._in_title = True
                    self.title_parts = []
                if tag.lower() in {"h1", "h2", "h3"}:
                    self._heading_tag = tag.lower()
                    self._heading_parts = []

            def handle_endtag(self, tag: str) -> None:
                lower = tag.lower()
                if lower == "title":
                    self._in_title = False
                if lower in {"h1", "h2", "h3"} and self._heading_tag == lower:
                    text_value = " ".join(self._heading_parts).strip()
                    if text_value:
                        self.headings.append({"tag": lower, "text": text_value})
                    self._heading_tag = None
                    self._heading_parts = []

            def handle_data(self, data: str) -> None:
                text = data.strip()
                if not text:
                    return
                if self._in_title:
                    self.title_parts.append(text)
                if self._heading_tag:
                    self._heading_parts.append(text)
                self.text_parts.append(text)

        self._parser = _Parser()

    def parse(self, html: str) -> Dict[str, Any]:
        self._parser.feed(html)
        title = " ".join(self._parser.title_parts).strip() or None
        text = " ".join(self._parser.text_parts).strip()
        headings = self._parser.headings or None
        return {"title": title, "text": text, "headings": headings}


def extract_payload(html: str) -> Dict[str, Any]:
    payload = _extract_with_trafilatura(html)
    if payload:
        return payload
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "lxml")
        title = soup.title.get_text(strip=True) if soup.title else None
        text = soup.get_text(" ", strip=True)
        headings: List[Dict[str, str]] = []
        for tag in ("h1", "h2", "h3"):
            for node in soup.find_all(tag):
                text_value = node.get_text(" ", strip=True)
                if text_value:
                    headings.append({"tag": tag, "text": text_value})
        return {"title": title, "text": text, "headings": headings or None}

    extractor = _SimpleHTMLExtractor()
    return extractor.parse(html)


def current_git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def append_manifest(manifest_path: Path, entry: Dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def serialise_flags(namespace: argparse.Namespace) -> Dict[str, Any]:
    flags: Dict[str, Any] = {}
    for key, value in vars(namespace).items():
        if isinstance(value, Path):
            flags[key] = str(value)
        else:
            flags[key] = value
    return flags


def run_scraper(args: argparse.Namespace) -> Dict[str, Any]:
    logging.basicConfig(level=getattr(logging, args.log_level))
    params = cdx_query_params(args.domain, args.from_year, args.to_year, args.match_type)
    rows = fetch_cdx_rows(params, limit=args.limit)
    logger.info("Fetched %d CDX rows for %s", len(rows), args.domain)

    rows_iter = enumerate(rows, start=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    stats = {
        "cdx_rows": len(rows),
        "snapshots_sampled": 0,
        "snapshots_fetched": 0,
        "snapshots_extracted": 0,
        "total_chars": 0,
    }

    config = RuntimeConfig(airgapped=False, quiet=True, qps=args.qps, burst=args.burst)
    with runtime_guard(config) as limiter, requests.Session() as session:
        for idx, row in rows_iter:
            if args.sample_every > 1 and (idx - 1) % args.sample_every != 0:
                continue
            stats["snapshots_sampled"] += 1
            wb_url = wayback_url(row.timestamp, row.original_url)
            html = fetch_html(session, wb_url, limiter)
            if html is None:
                continue
            stats["snapshots_fetched"] += 1
            payload = extract_payload(html)
            text = (payload.get("text") or "").strip()
            if not text:
                continue
            stats["snapshots_extracted"] += 1
            stats["total_chars"] += len(text)
            record = {
                "original_url": row.original_url,
                "wayback_url": wb_url,
                "timestamp": row.timestamp,
                "date_iso": iso_date_from_timestamp(row.timestamp),
                "extracted": payload,
            }
            records.append(record)

    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    output_sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
    config_blob = json.dumps(serialise_flags(args), sort_keys=True).encode("utf-8")
    config_sha = hashlib.sha256(config_blob).hexdigest()

    manifest_entry = {
        "run_id": f"{datetime.utcnow().isoformat(timespec='seconds')}Z",
        "tool": "fsb_wayback.scraper",
        "domain": args.domain,
        "git_commit": current_git_commit(),
        "cdx_params": params,
        "stats": stats,
        "hashes": {
            "output_jsonl_sha256": output_sha,
            "config_sha256": config_sha,
        },
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    append_manifest(args.manifest, manifest_entry)

    audit = write_audit_report(
        flags=serialise_flags(args),
        manifest_path=args.manifest,
        kb_stats={"records": len(records), "chars": stats["total_chars"]},
        overlap_trace=[],
        output_path=args.audit_report,
    )
    return {"manifest": manifest_entry, "audit": audit, "records": len(records), "stats": stats}


def main(argv: Sequence[str] | None = None) -> Dict[str, Any]:
    args = parse_args(argv)
    return run_scraper(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
