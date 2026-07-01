"""Unit tests for src.ingestion.fetcher."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.ingestion.fetcher import (
    FetchResult,
    SchemeConfig,
    compute_content_hash,
    fetch_all,
    fetch_http,
    fetch_page_html,
    fetch_scheme,
    is_minimal_content,
    load_schemes,
)

RICH_HTML = """
<html><body>
<h1>HDFC Large Cap Fund Direct Growth</h1>
<p>Expense ratio 1.04%. Minimum SIP is Rs 100.</p>
<p>Exit load applies. Benchmark NIFTY 100 TRI. NAV updated daily.</p>
<p>Mutual fund investment objective and riskometer details for investors.</p>
""" + ("Lorem ipsum dolor sit amet. " * 200) + """
</body></html>
"""

MINIMAL_HTML = "<html><body><div id='root'></div><script>boot()</script></body></html>"


@pytest.fixture
def scheme() -> SchemeConfig:
    return SchemeConfig(
        scheme_id="large_cap",
        scheme_name="HDFC Large Cap Fund Direct Growth",
        url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
        source_type="groww_fund_page",
    )


@pytest.fixture
def corpus_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.yaml"
    path.write_text(
        """
schemes:
  - scheme_id: large_cap
    scheme_name: HDFC Large Cap Fund Direct Growth
    url: https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth
    source_type: groww_fund_page
""",
        encoding="utf-8",
    )
    return path


def test_load_schemes(corpus_yaml: Path):
    schemes = load_schemes(corpus_yaml)
    assert len(schemes) == 1
    assert schemes[0].scheme_id == "large_cap"


def test_compute_content_hash_is_deterministic():
    assert compute_content_hash("abc") == compute_content_hash("abc")
    assert compute_content_hash("abc") != compute_content_hash("abcd")


def test_is_minimal_content():
    assert is_minimal_content(MINIMAL_HTML) is True
    assert is_minimal_content(RICH_HTML) is False


def test_fetch_http_retries_then_succeeds():
    response_ok = httpx.Response(200, text=RICH_HTML, request=httpx.Request("GET", "https://groww.in/x"))
    client = MagicMock()
    client.get.side_effect = [
        httpx.TimeoutException("timeout"),
        response_ok,
    ]

    with patch("src.ingestion.fetcher.time.sleep"):
        html = fetch_http("https://groww.in/x", client=client)

    assert "Expense ratio" in html
    assert client.get.call_count == 2


def test_fetch_page_html_uses_playwright_on_minimal_static_html():
    client = MagicMock()
    client.get.return_value = httpx.Response(
        200, text=MINIMAL_HTML, request=httpx.Request("GET", "https://groww.in/x")
    )

    with patch("src.ingestion.fetcher.fetch_playwright", return_value=RICH_HTML) as pw:
        html, method = fetch_page_html("https://groww.in/x", client=client)

    assert method == "playwright"
    assert "Expense ratio" in html
    pw.assert_called_once()


def test_fetch_scheme_skips_write_when_hash_unchanged(scheme: SchemeConfig, tmp_path: Path):
    html_path = tmp_path / "large_cap.html"
    html_path.write_text(RICH_HTML, encoding="utf-8")
    content_hash = compute_content_hash(RICH_HTML)
    manifest = {
        "large_cap": {
            "content_hash": content_hash,
            "fetched_at": "2026-01-01T00:00:00+00:00",
            "fetch_method": "httpx",
        }
    }
    (tmp_path / "fetch_manifest.json").write_text(__import__("json").dumps(manifest), encoding="utf-8")

    with patch("src.ingestion.fetcher.fetch_page_html", return_value=(RICH_HTML, "httpx")):
        result = fetch_scheme(scheme, force_refresh=False, raw_dir=tmp_path)

    assert isinstance(result, FetchResult)
    assert result.skipped is True
    assert result.bytes_written == 0
    assert result.content_hash == content_hash


def test_fetch_scheme_writes_on_force_refresh(scheme: SchemeConfig, tmp_path: Path):
    html_path = tmp_path / "large_cap.html"
    html_path.write_text("old", encoding="utf-8")
    old_hash = compute_content_hash("old")
    manifest = {"large_cap": {"content_hash": old_hash, "fetched_at": "2026-01-01T00:00:00+00:00"}}
    (tmp_path / "fetch_manifest.json").write_text(__import__("json").dumps(manifest), encoding="utf-8")

    with patch("src.ingestion.fetcher.fetch_page_html", return_value=(RICH_HTML, "httpx")):
        result = fetch_scheme(scheme, force_refresh=True, raw_dir=tmp_path)

    assert result.skipped is False
    assert result.bytes_written > 0
    assert html_path.read_text(encoding="utf-8") == RICH_HTML


def test_fetch_all_single_scheme_filter(scheme: SchemeConfig, tmp_path: Path):
    with patch("src.ingestion.fetcher.load_schemes", return_value=[scheme]):
        with patch(
            "src.ingestion.fetcher.fetch_scheme",
            return_value=FetchResult(
                scheme_id="large_cap",
                url=scheme.url,
                html_path=tmp_path / "large_cap.html",
                content_hash="abc",
                fetched_at="2026-01-01T00:00:00+00:00",
                fetch_method="httpx",
                skipped=False,
                bytes_written=100,
            ),
        ) as fetch_one:
            results = fetch_all(scheme_id="large_cap", raw_dir=tmp_path)

    assert len(results) == 1
    fetch_one.assert_called_once()


def test_fetch_all_unknown_scheme_raises():
    with patch("src.ingestion.fetcher.load_schemes", return_value=[]):
        with pytest.raises(ValueError, match="Unknown scheme_id"):
            fetch_all(scheme_id="missing")
