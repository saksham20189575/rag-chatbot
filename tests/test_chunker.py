"""Unit tests for src.ingestion.chunker."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.constants import CHUNK_OVERLAP_TOKENS, RAW_DATA_DIR
from src.ingestion.fetcher import SchemeConfig
from src.ingestion.parser import ParsedPage, TextBlock
from src.ingestion.chunker import (
    Chunk,
    chunk_all,
    chunk_page,
    clean_fund_management,
    estimate_tokens,
    normalize_last_updated,
    save_chunks,
    _split_with_overlap,
)

LARGE_CAP = SchemeConfig(
    scheme_id="large_cap",
    scheme_name="HDFC Large Cap Fund Direct Growth",
    url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
    source_type="groww_fund_page",
)

FUND_MGMT_NOISE = (
    "RB Rahul Baijal Jul 2022 - Present View details Education Mr. Baijal has done PGDM. "
    "Experience Prior to joining HDFC Mutual Fund, he worked with Sundaram Mutual Fund. "
    "Also manages these schemes HDFC Mid Cap Fund Direct Growth HDFC Small Cap Fund Direct Growth "
    "HDFC Business Cycle Fund Direct Growth "
    "DM Dhruv Muchhal Jun 2023 - Present View details Education Mr. Dhruv has done CA and CFA. "
    "Experience Prior to joining HDFC MF he worked with Goldman Sachs. "
    "Also manages these schemes HDFC Flexi Cap Direct Plan Growth HDFC Value Fund Direct Plan Growth"
)


def _sample_page() -> ParsedPage:
    return ParsedPage(
        scheme_id="large_cap",
        text_blocks=[
            TextBlock("Key fund facts", "Expense ratio: 1.04%. Minimum SIP investment: ₹100."),
            TextBlock("Exit load", "Exit load of 1% if redeemed within 1 year"),
            TextBlock("Benchmark", "Fund benchmark: NIFTY 100 Total Return Index."),
            TextBlock("Investment objective", "Long-term capital appreciation in Large-Cap companies."),
            TextBlock("Fund management", FUND_MGMT_NOISE),
            TextBlock("Minimum investments", "Min. for SIP ₹100"),
        ],
        nav_date="25-Jun-2026",
        fetched_at="2026-06-28T05:55:41+00:00",
    )


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("one two three") >= 3


def test_normalize_last_updated_from_nav_date():
    assert normalize_last_updated("25-Jun-2026", None) == "2026-06-25"
    assert normalize_last_updated("1-Jan-2026", None) == "2026-01-01"


def test_normalize_last_updated_fallback_to_fetched_at():
    assert normalize_last_updated(None, "2026-06-28T05:55:41+00:00") == "2026-06-28"
    assert normalize_last_updated("garbage", "2026-06-28T00:00:00+00:00") == "2026-06-28"


def test_clean_fund_management_strips_noise_and_keeps_bios():
    cleaned = clean_fund_management(FUND_MGMT_NOISE)
    assert "Also manages these schemes" not in cleaned
    assert "View details" not in cleaned
    assert "Rahul Baijal" in cleaned
    assert "Dhruv Muchhal" in cleaned
    assert "Sundaram Mutual Fund" in cleaned


def test_clean_fund_management_removes_cross_scheme_names():
    """Other-scheme names must not survive in the large_cap chunk (RT-05)."""
    cleaned = clean_fund_management(FUND_MGMT_NOISE)
    assert "HDFC Mid Cap Fund" not in cleaned
    assert "HDFC Small Cap Fund" not in cleaned
    assert "HDFC Flexi Cap" not in cleaned


def test_split_with_overlap_small_text_single_chunk():
    assert _split_with_overlap("short text here") == ["short text here"]


def test_split_with_overlap_large_text_splits():
    text = " ".join(f"word{i}" for i in range(2000))
    pieces = _split_with_overlap(text, max_tokens=100, overlap_tokens=20)
    assert len(pieces) > 1
    # Overlap: end of piece 0 should reappear at start of piece 1.
    tail = pieces[0].split()[-5:]
    assert any(w in pieces[1].split()[: CHUNK_OVERLAP_TOKENS] for w in tail)


def test_chunk_page_metadata_and_ids():
    chunks = chunk_page(_sample_page(), scheme=LARGE_CAP)
    assert len(chunks) == 6  # one per section

    first = chunks[0]
    assert first.chunk_id == "large_cap_groww_0000"
    assert first.scheme_id == "large_cap"
    assert first.scheme_name == "HDFC Large Cap Fund Direct Growth"
    assert first.document_type == "groww_fund_page"
    assert first.source_domain == "groww.in"
    assert first.source_url.startswith("https://groww.in/")
    assert first.last_updated == "2026-06-25"
    assert first.section_title == "Key fund facts"

    # chunk_ids are unique and zero-padded sequential
    ids = [c.chunk_id for c in chunks]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)


def test_chunk_page_embed_text_has_scheme_context():
    chunks = chunk_page(_sample_page(), scheme=LARGE_CAP)
    exit_load = next(c for c in chunks if c.section_title == "Exit load")
    assert exit_load.embed_text.startswith("HDFC Large Cap Fund Direct Growth | Exit load:")
    # Stored answer text stays clean (no scheme prefix)
    assert exit_load.text == "Exit load of 1% if redeemed within 1 year"


def test_chunk_page_fund_management_cleaned():
    chunks = chunk_page(_sample_page(), scheme=LARGE_CAP)
    fm = next(c for c in chunks if c.section_title == "Fund management")
    assert "Also manages these schemes" not in fm.text
    assert "HDFC Mid Cap Fund" not in fm.text


def test_chunk_metadata_excludes_embed_text():
    chunk = chunk_page(_sample_page(), scheme=LARGE_CAP)[0]
    meta = chunk.metadata()
    assert "embed_text" not in meta
    assert "text" not in meta
    assert meta["chunk_id"] == "large_cap_groww_0000"


def test_save_chunks_writes_review_files(tmp_path: Path):
    chunks = chunk_page(_sample_page(), scheme=LARGE_CAP)
    paths = save_chunks(chunks, chunks_dir=tmp_path)
    assert len(paths) == 1
    assert (tmp_path / "large_cap.json").exists()
    assert (tmp_path / "chunk_manifest.json").exists()

    payload = json.loads((tmp_path / "large_cap.json").read_text(encoding="utf-8"))
    assert payload["scheme_id"] == "large_cap"
    assert payload["chunk_count"] == len(chunks)
    assert payload["chunks"][0]["chunk_id"] == "large_cap_groww_0000"


def test_chunk_all_real_corpus_meets_eval_gates():
    if not list(RAW_DATA_DIR.glob("*.html")):
        pytest.skip("raw HTML not present — run fetcher first")

    chunks = chunk_all()

    # eval.md gates: ≥20 total, ≥3 per scheme, 5 schemes, only groww.in URLs.
    assert len(chunks) >= 20
    by_scheme: dict[str, int] = {}
    for chunk in chunks:
        by_scheme[chunk.scheme_id] = by_scheme.get(chunk.scheme_id, 0) + 1
        assert "groww.in" in chunk.source_url
        assert chunk.document_type == "groww_fund_page"
        assert ".pdf" not in chunk.source_url
    assert len(by_scheme) == 5
    assert all(count >= 3 for count in by_scheme.values())


def test_chunk_all_no_cross_scheme_contamination():
    if not list(RAW_DATA_DIR.glob("*.html")):
        pytest.skip("raw HTML not present — run fetcher first")

    other_names = {
        "large_cap": "HDFC Mid Cap Fund",
        "gold_fof": "HDFC Silver ETF",
    }
    chunks = chunk_all()
    for scheme_id, foreign in other_names.items():
        fm = [
            c
            for c in chunks
            if c.scheme_id == scheme_id and c.section_title == "Fund management"
        ]
        for chunk in fm:
            assert foreign not in chunk.text, f"{foreign} leaked into {scheme_id} fund mgmt"
