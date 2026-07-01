"""Groww fund page fetcher — HTTP with Playwright fallback and hash-based skip."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml
from bs4 import BeautifulSoup

from src.constants import ALLOWED_DOMAIN, ALLOWLISTED_GROWW_URLS, CORPUS_CONFIG_PATH, RAW_DATA_DIR

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT_SECONDS = 30.0
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 1.5
RATE_LIMIT_SECONDS = 1.0
MIN_HTML_BYTES = 5_000
MIN_TEXT_CHARS = 800
MANIFEST_FILENAME = "fetch_manifest.json"


@dataclass(frozen=True)
class SchemeConfig:
    scheme_id: str
    scheme_name: str
    url: str
    source_type: str


@dataclass
class FetchResult:
    scheme_id: str
    url: str
    html_path: Path
    content_hash: str
    fetched_at: str
    fetch_method: str
    skipped: bool
    bytes_written: int


def load_schemes(config_path: Path | None = None) -> list[SchemeConfig]:
    """Load and validate scheme entries from corpus.yaml."""
    path = config_path or CORPUS_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    schemes: list[SchemeConfig] = []
    for entry in data.get("schemes", []):
        scheme = SchemeConfig(
            scheme_id=entry["scheme_id"],
            scheme_name=entry["scheme_name"],
            url=entry["url"],
            source_type=entry["source_type"],
        )
        _validate_scheme(scheme)
        schemes.append(scheme)
    return schemes


def _validate_scheme(scheme: SchemeConfig) -> None:
    if scheme.source_type != "groww_fund_page":
        raise ValueError(f"{scheme.scheme_id}: source_type must be groww_fund_page")
    if ALLOWED_DOMAIN not in scheme.url:
        raise ValueError(f"{scheme.scheme_id}: URL must be on {ALLOWED_DOMAIN}")
    if scheme.url not in ALLOWLISTED_GROWW_URLS:
        raise ValueError(f"{scheme.scheme_id}: URL not in allowlist")


def compute_content_hash(content: bytes | str) -> str:
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _extract_visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ", strip=True).split())


def is_minimal_content(html: str) -> bool:
    """True when static HTML looks like an empty shell needing JS rendering."""
    if len(html.encode("utf-8")) < MIN_HTML_BYTES:
        return True
    text = _extract_visible_text(html)
    if len(text) < MIN_TEXT_CHARS:
        return True
    lowered = text.lower()
    signals = ("expense ratio", "nav", "sip", "exit load", "benchmark", "mutual fund")
    return not any(signal in lowered for signal in signals)


def _manifest_path(raw_dir: Path | None = None) -> Path:
    return (raw_dir or RAW_DATA_DIR) / MANIFEST_FILENAME


def _load_manifest(raw_dir: Path | None = None) -> dict[str, Any]:
    path = _manifest_path(raw_dir)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_manifest(manifest: dict[str, Any], raw_dir: Path | None = None) -> None:
    path = _manifest_path(raw_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def _html_path(scheme_id: str, raw_dir: Path | None = None) -> Path:
    return (raw_dir or RAW_DATA_DIR) / f"{scheme_id}.html"


def fetch_http(url: str, *, client: httpx.Client | None = None) -> str:
    """HTTP GET with retries, timeout, and User-Agent."""
    headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if client is None:
                with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=True) as c:
                    response = c.get(url, headers=headers)
            else:
                response = client.get(url, headers=headers)
            response.raise_for_status()
            response.encoding = response.encoding or "utf-8"
            return response.text
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                sleep_for = RETRY_BACKOFF_SECONDS * attempt
                logger.warning("HTTP fetch attempt %s/%s failed for %s: %s", attempt, MAX_RETRIES, url, exc)
                time.sleep(sleep_for)

    assert last_error is not None
    raise RuntimeError(f"HTTP fetch failed for {url} after {MAX_RETRIES} attempts") from last_error


def fetch_playwright(url: str) -> str:
    """Fetch rendered HTML via Playwright (JS fallback)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is required for JS-rendered Groww pages") from exc

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page(user_agent=DEFAULT_USER_AGENT)
            page.goto(url, wait_until="networkidle", timeout=int(REQUEST_TIMEOUT_SECONDS * 1000))
            html = page.content()
        finally:
            browser.close()
    return html


def fetch_page_html(url: str, *, client: httpx.Client | None = None) -> tuple[str, str]:
    """Fetch HTML; fall back to Playwright when static content is minimal."""
    html = fetch_http(url, client=client)
    if is_minimal_content(html):
        logger.info("Minimal static HTML for %s — using Playwright fallback", url)
        html = fetch_playwright(url)
        if is_minimal_content(html):
            raise RuntimeError(f"Fetched content still minimal after Playwright for {url}")
        return html, "playwright"
    return html, "httpx"


def fetch_scheme(
    scheme: SchemeConfig,
    *,
    force_refresh: bool = False,
    raw_dir: Path | None = None,
    client: httpx.Client | None = None,
) -> FetchResult:
    """Fetch one Groww page and persist raw HTML when content changes."""
    out_dir = raw_dir or RAW_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    html_path = _html_path(scheme.scheme_id, out_dir)
    manifest = _load_manifest(out_dir)
    stored = manifest.get(scheme.scheme_id, {})
    stored_hash = stored.get("content_hash")

    html, fetch_method = fetch_page_html(scheme.url, client=client)
    content_hash = compute_content_hash(html)
    fetched_at = datetime.now(UTC).replace(microsecond=0).isoformat()

    if stored_hash == content_hash and not force_refresh and html_path.exists():
        logger.info("Skipping write for %s — content hash unchanged", scheme.scheme_id)
        return FetchResult(
            scheme_id=scheme.scheme_id,
            url=scheme.url,
            html_path=html_path,
            content_hash=content_hash,
            fetched_at=stored.get("fetched_at", fetched_at),
            fetch_method=stored.get("fetch_method", fetch_method),
            skipped=True,
            bytes_written=0,
        )

    html_path.write_text(html, encoding="utf-8")
    manifest[scheme.scheme_id] = {
        "scheme_id": scheme.scheme_id,
        "scheme_name": scheme.scheme_name,
        "url": scheme.url,
        "content_hash": content_hash,
        "fetched_at": fetched_at,
        "fetch_method": fetch_method,
    }
    _save_manifest(manifest, out_dir)

    return FetchResult(
        scheme_id=scheme.scheme_id,
        url=scheme.url,
        html_path=html_path,
        content_hash=content_hash,
        fetched_at=fetched_at,
        fetch_method=fetch_method,
        skipped=False,
        bytes_written=len(html.encode("utf-8")),
    )


def fetch_all(
    *,
    force_refresh: bool = False,
    scheme_id: str | None = None,
    raw_dir: Path | None = None,
) -> list[FetchResult]:
    """Fetch all configured schemes (or one scheme) with rate limiting."""
    schemes = load_schemes()
    if scheme_id is not None:
        schemes = [s for s in schemes if s.scheme_id == scheme_id]
        if not schemes:
            known = {s.scheme_id for s in load_schemes()}
            raise ValueError(f"Unknown scheme_id: {scheme_id!r}. Valid: {sorted(known)}")

    results: list[FetchResult] = []
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=True) as client:
        for index, scheme in enumerate(schemes):
            if index > 0:
                time.sleep(RATE_LIMIT_SECONDS)
            logger.info("Fetching %s (%s)", scheme.scheme_id, scheme.url)
            results.append(
                fetch_scheme(scheme, force_refresh=force_refresh, raw_dir=raw_dir, client=client)
            )
    return results


def fetch_result_to_dict(result: FetchResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["html_path"] = str(result.html_path)
    return payload
