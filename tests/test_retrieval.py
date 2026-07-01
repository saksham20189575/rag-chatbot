"""Tests for Phase 2 retrieval: query utils and scheme-aware retriever."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.constants import CHUNKS_DATA_DIR, DEFAULT_BGE_MODEL, INDEX_DIR
from src.ingestion.chunker import chunk_page, save_chunks
from src.ingestion.fetcher import SchemeConfig
from src.ingestion.indexer import build_index, index_chunks
from src.ingestion.parser import ParsedPage, TextBlock
from src.rag.query_utils import detect_scheme_id, expand_abbreviations, normalize_query
from src.rag.retriever import (
    Retriever,
    _keyword_boost,
    _target_sections,
    assemble_context,
)

LARGE_CAP = SchemeConfig(
    scheme_id="large_cap",
    scheme_name="HDFC Large Cap Fund Direct Growth",
    url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
    source_type="groww_fund_page",
)

MID_CAP = SchemeConfig(
    scheme_id="mid_cap",
    scheme_name="HDFC Mid Cap Fund Direct Growth",
    url="https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
    source_type="groww_fund_page",
)

GOLD_FOF = SchemeConfig(
    scheme_id="gold_fof",
    scheme_name="HDFC Gold ETF Fund of Fund Direct Plan Growth",
    url="https://groww.in/mutual-funds/hdfc-gold-etf-fund-of-fund-direct-plan-growth",
    source_type="groww_fund_page",
)

SMALL_CAP = SchemeConfig(
    scheme_id="small_cap",
    scheme_name="HDFC Small Cap Fund Direct Growth",
    url="https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
    source_type="groww_fund_page",
)

SILVER_FOF = SchemeConfig(
    scheme_id="silver_fof",
    scheme_name="HDFC Silver ETF FoF Direct Growth",
    url="https://groww.in/mutual-funds/hdfc-silver-etf-fof-direct-growth",
    source_type="groww_fund_page",
)

ALL_SCHEMES = [LARGE_CAP, MID_CAP, SMALL_CAP, GOLD_FOF, SILVER_FOF]


class FakeEmbedder:
    """Deterministic 384-d vectors keyed by embed_text / query content."""

    def encode(self, texts, **kwargs):
        del kwargs
        rows = []
        for text in texts:
            lowered = text.lower()
            seed = sum(ord(c) for c in text) % 997
            row = np.zeros(384, dtype=np.float32)
            row[0] = float(seed)
            row[1] = float(len(text))
            if "exit load" in lowered:
                row[2] = 50.0
            elif "expense ratio" in lowered or "key fund facts" in lowered:
                row[2] = 40.0
            elif "benchmark" in lowered:
                row[2] = 30.0
            elif "minimum investments" in lowered or "minimum sip" in lowered:
                row[2] = 20.0
            elif "riskometer" in lowered:
                row[2] = 35.0
            norm = np.linalg.norm(row) or 1.0
            rows.append(row / norm)
        return np.vstack(rows)


def _full_page(scheme_id: str) -> ParsedPage:
    return ParsedPage(
        scheme_id=scheme_id,
        text_blocks=[
            TextBlock(
                "Key fund facts",
                "Expense ratio: 1.04%. Minimum SIP investment: ₹100. Riskometer: Moderately High Riskometer.",
            ),
            TextBlock("Exit load", "Exit load of 1% if redeemed within 1 year"),
            TextBlock("Benchmark", "Fund benchmark: NIFTY 100 Total Return Index."),
            TextBlock("Investment objective", "To provide long-term capital appreciation."),
            TextBlock("Stamp duty and tax", "Stamp duty 0.005% on investment. STT not applicable."),
            TextBlock("Fund management", "Fund managed by Example Manager since Jan 2013."),
            TextBlock("Minimum investments", "Minimum SIP ₹100. Minimum lumpsum ₹100."),
        ],
        nav_date="25-Jun-2026",
        fetched_at="2026-06-28T05:55:41+00:00",
    )


def _build_test_index(tmp_path: Path) -> Retriever:
    chunks = []
    for scheme in ALL_SCHEMES:
        chunks.extend(chunk_page(_full_page(scheme.scheme_id), scheme=scheme))
    index_chunks(chunks, model=FakeEmbedder(), index_dir=tmp_path)  # type: ignore[arg-type]
    return Retriever(index_dir=tmp_path, model=FakeEmbedder())  # type: ignore[arg-type]


# --- query_utils ---


def test_expand_abbreviations_ter_and_fof():
    assert "expense ratio" in expand_abbreviations("What is the TER?").lower()
    assert "fund of fund" in expand_abbreviations("Gold ETF FoF exit load").lower()


def test_detect_scheme_id_aliases():
    assert detect_scheme_id("expense ratio of HDFC Mid Cap Fund") == "mid_cap"
    assert detect_scheme_id("exit load for Gold ETF FoF") == "gold_fof"
    assert detect_scheme_id("gold fund exit load") == "gold_fof"
    assert detect_scheme_id("riskometer Silver ETF FoF") == "silver_fof"
    assert detect_scheme_id("What is the expense ratio?") is None


def test_normalize_query_expands_and_detects():
    expanded, scheme_id = normalize_query("TER of HDFC Large Cap")
    assert "expense ratio" in expanded.lower()
    assert scheme_id == "large_cap"


def test_target_sections_maps_expense_ratio():
    assert "Key fund facts" in _target_sections("expense ratio of mid cap")


def test_keyword_boost_applies_for_matching_section():
    boost = _keyword_boost("Exit load", _target_sections("exit load for gold fof"))
    assert boost > 0
    assert _keyword_boost("Benchmark", _target_sections("exit load")) == 0.0


def test_assemble_context_format():
    from src.rag.retriever import RetrievedChunk

    chunk = RetrievedChunk(
        chunk_id="large_cap_groww_0001",
        scheme_id="large_cap",
        scheme_name="HDFC Large Cap Fund Direct Growth",
        section_title="Exit load",
        text="Exit load of 1% if redeemed within 1 year",
        source_url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
        last_updated="2026-06-25",
        score=0.9,
    )
    context = assemble_context([chunk])
    assert "[https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth | 2026-06-25 | Exit load]" in context
    assert "Exit load of 1%" in context


# --- retriever gate queries (eval.md Phase 2) ---

GATE_QUERIES = [
    ("expense ratio of HDFC Mid Cap Fund", "mid_cap"),
    ("exit load for Gold ETF FoF", "gold_fof"),
    ("minimum SIP HDFC Small Cap", "small_cap"),
    ("benchmark HDFC Large Cap Fund", "large_cap"),
    ("riskometer Silver ETF FoF", "silver_fof"),
]

P1_QUERIES = [
    ("TER of HDFC Large Cap", "large_cap"),
    ("gold fund exit load", "gold_fof"),
]


@pytest.fixture
def retriever(tmp_path: Path) -> Retriever:
    return _build_test_index(tmp_path)


@pytest.mark.parametrize("query,expected_scheme", GATE_QUERIES)
def test_retrieval_gate_scheme_at_1(retriever: Retriever, query: str, expected_scheme: str):
    result = retriever.retrieve(query, min_similarity=0.0)
    assert result.scheme_id == expected_scheme
    assert result.has_results
    assert result.chunks[0].scheme_id == expected_scheme


@pytest.mark.parametrize("query,expected_scheme", P1_QUERIES)
def test_retrieval_alias_scheme_at_1(retriever: Retriever, query: str, expected_scheme: str):
    result = retriever.retrieve(query, min_similarity=0.0)
    assert result.scheme_id == expected_scheme
    assert result.chunks[0].scheme_id == expected_scheme


def test_retrieval_no_scheme_returns_empty(retriever: Retriever):
    result = retriever.retrieve("What is the expense ratio?")
    assert result.scheme_id is None
    assert not result.has_results
    assert result.context == ""


def test_retrieval_section_keyword_boost_exit_load(retriever: Retriever):
    result = retriever.retrieve("exit load for Gold ETF FoF", min_similarity=0.0)
    assert result.chunks[0].section_title == "Exit load"


def test_retrieval_section_keyword_boost_expense_ratio(retriever: Retriever):
    result = retriever.retrieve("expense ratio of HDFC Mid Cap Fund", min_similarity=0.0)
    assert result.chunks[0].section_title == "Key fund facts"


def test_retrieval_ter_expansion_hits_key_fund_facts(retriever: Retriever):
    result = retriever.retrieve("TER of HDFC Large Cap", min_similarity=0.0)
    assert result.chunks[0].section_title == "Key fund facts"


def test_retrieval_embedding_model_mismatch_raises(tmp_path: Path):
    chunks_dir = tmp_path / "chunks"
    save_chunks(chunk_page(_full_page("large_cap"), scheme=LARGE_CAP), chunks_dir=chunks_dir)
    with patch("src.ingestion.indexer.load_embedding_model", return_value=FakeEmbedder()):
        build_index(chunks_dir=chunks_dir, index_dir=tmp_path)

    with patch.dict("os.environ", {"BGE_MODEL_NAME": "BAAI/bge-large-en-v1.5"}):
        with pytest.raises(ValueError, match="Index was built with"):
            Retriever(index_dir=tmp_path, model=FakeEmbedder())  # type: ignore[arg-type]


@pytest.mark.skipif(
    not (INDEX_DIR / "index_manifest.json").is_file(),
    reason="real index not built",
)
def test_retrieval_real_index_gate_queries():
    """Slow integration test against the live BGE index when present."""
    retriever = Retriever()
    for query, expected_scheme in GATE_QUERIES + P1_QUERIES:
        result = retriever.retrieve(query)
        assert result.scheme_id == expected_scheme, query
        assert result.has_results, query
        assert result.chunks[0].scheme_id == expected_scheme, query


@pytest.mark.skipif(
    not (CHUNKS_DATA_DIR / "chunk_manifest.json").is_file(),
    reason="chunk review files not present",
)
def test_retrieval_real_corpus_with_fake_embedder(tmp_path: Path):
    with patch("src.ingestion.indexer.load_embedding_model", return_value=FakeEmbedder()):
        build_index(chunks_dir=CHUNKS_DATA_DIR, index_dir=tmp_path)
    retriever = Retriever(index_dir=tmp_path, model=FakeEmbedder())  # type: ignore[arg-type]
    result = retriever.retrieve("exit load for Gold ETF FoF", min_similarity=0.0)
    assert result.scheme_id == "gold_fof"
    assert result.chunks[0].scheme_id == "gold_fof"
