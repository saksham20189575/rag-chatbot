"""Unit tests for src.ingestion.parser."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.constants import RAW_DATA_DIR
from src.ingestion.parser import (
    ParsedPage,
    TextBlock,
    _build_text_blocks_from_json,
    _dedupe_text_blocks,
    extract_mf_server_side_data,
    parse_all,
    parse_html,
    parse_raw_file,
    save_parsed_pages,
)

SAMPLE_MF = {
    "scheme_name": "HDFC Large Cap Fund Direct Growth",
    "nav": 1217.44,
    "nav_date": "25-Jun-2026",
    "expense_ratio": 1.04,
    "min_sip_investment": 100,
    "min_investment_amount": 100,
    "aum": 37808.3057,
    "nfo_risk": "Moderately High Riskometer",
    "exit_load": "Exit load of 1% if redeemed within 1 year",
    "benchmark_name": "NIFTY 100 Total Return Index",
    "description": "The scheme seeks long-term capital appreciation in Large-Cap companies.",
    "fund_manager": "Prashant Jain",
    "category": "Equity",
    "sub_category": "Large Cap",
    "plan_type": "Direct",
    "scheme_type": "Growth",
}


def test_build_text_blocks_from_json_key_fields():
    blocks = _build_text_blocks_from_json(SAMPLE_MF)
    titles = {b.section_title for b in blocks}
    assert "Key fund facts" in titles
    assert "Exit load" in titles
    assert "Benchmark" in titles
    assert "Investment objective" in titles

    combined = " ".join(b.text for b in blocks)
    assert "1.04%" in combined
    assert "₹100" in combined
    assert "NIFTY 100 Total Return Index" in combined


def test_dedupe_text_blocks_keeps_longer():
    blocks = _dedupe_text_blocks(
        [
            TextBlock("Exit load", "short"),
            TextBlock("Exit load", "much longer exit load description"),
        ]
    )
    assert len(blocks) == 1
    assert blocks[0].text.startswith("much longer")


def test_parse_raw_large_cap():
    if not (RAW_DATA_DIR / "large_cap.html").exists():
        pytest.skip("raw HTML not present — run fetcher first")

    parsed = parse_raw_file("large_cap")
    assert isinstance(parsed, ParsedPage)
    assert parsed.scheme_id == "large_cap"
    assert parsed.nav_date
    assert parsed.fetched_at
    assert len(parsed.text_blocks) >= 4

    combined = " ".join(b.text for b in parsed.text_blocks)
    assert "1.04" in combined
    assert "100" in combined


@pytest.mark.parametrize(
    ("scheme_id", "needles"),
    [
        ("mid_cap", ["NIFTY Midcap 150", "0.75"]),
        ("small_cap", ["Small-Cap", "₹100"]),
        ("gold_fof", ["15 days", "Domestic Price of Gold"]),
        ("silver_fof", ["Very High", "Domestic Price of Silver"]),
    ],
)
def test_parse_all_schemes(scheme_id: str, needles: list[str]):
    html_path = RAW_DATA_DIR / f"{scheme_id}.html"
    if not html_path.exists():
        pytest.skip("raw HTML not present — run fetcher first")

    parsed = parse_raw_file(scheme_id)
    combined = " ".join(b.text for b in parsed.text_blocks).lower()
    for needle in needles:
        assert needle.lower() in combined, f"missing {needle!r} in {scheme_id}"


def test_parse_html_missing_json_raises_when_empty():
    html = "<html><body><p>unrelated</p></body></html>"
    with pytest.raises(ValueError, match="No fund content"):
        parse_html(html, "large_cap", fetched_at="2026-01-01T00:00:00+00:00")


def test_extract_mf_server_side_data_from_fixture():
    html = Path(RAW_DATA_DIR / "large_cap.html").read_text(encoding="utf-8") if (
        RAW_DATA_DIR / "large_cap.html"
    ).exists() else ""
    if not html:
        pytest.skip("raw HTML not present")

    from bs4 import BeautifulSoup

    mf = extract_mf_server_side_data(BeautifulSoup(html, "html.parser"))
    assert mf is not None
    assert float(mf["expense_ratio"]) == 1.04


def test_parse_all_returns_five_when_raw_present():
    if not list(RAW_DATA_DIR.glob("*.html")):
        pytest.skip("raw HTML not present")

    pages = parse_all()
    assert len(pages) == 5


def test_save_parsed_pages_writes_review_files(tmp_path: Path):
    page = ParsedPage(
        scheme_id="large_cap",
        text_blocks=[TextBlock("Key fund facts", "Expense ratio: 1.04%.")],
        nav_date="25-Jun-2026",
        fetched_at="2026-06-28T00:00:00+00:00",
    )
    paths = save_parsed_pages([page], parsed_dir=tmp_path)
    assert len(paths) == 1
    assert (tmp_path / "large_cap.json").exists()
    assert (tmp_path / "parse_manifest.json").exists()

    payload = json.loads((tmp_path / "large_cap.json").read_text(encoding="utf-8"))
    assert payload["scheme_id"] == "large_cap"
    assert payload["scheme_name"]
    assert payload["source_url"].startswith("https://groww.in/")
    assert payload["text_blocks"][0]["section_title"] == "Key fund facts"
