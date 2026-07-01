"""Tests for the Phase 1.5 build pipeline and CLI orchestration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import argparse

import pytest

from src.constants import CHUNKS_DATA_DIR, RAW_DATA_DIR
from src.ingestion.chunker import chunk_all
from src.ingestion.indexer import chunks_fingerprint, read_index_manifest
from src.ingestion.pipeline import (
    BuildOptions,
    count_chunks_per_scheme,
    format_build_summary,
    run_build,
    should_skip_indexing,
    validate_build_options,
    validate_corpus,
)


def test_validate_build_options_rejects_conflicting_flags():
    with pytest.raises(ValueError, match="--parse-only or --index-only"):
        validate_build_options(BuildOptions(parse_only=True, index_only=True))


def test_should_skip_indexing_when_fingerprint_matches(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.ingestion.pipeline.INDEX_DIR", tmp_path)
    monkeypatch.setattr("src.ingestion.indexer.INDEX_DIR", tmp_path)

    if not list(RAW_DATA_DIR.glob("*.html")):
        pytest.skip("raw HTML not present")

    chunks = chunk_all()
    fingerprint = chunks_fingerprint(chunks)

    manifest_path = tmp_path / "index_manifest.json"
    manifest_path.write_text(
        f'{{"chunks_fingerprint": "{fingerprint}", "chunk_count": 35}}\n',
        encoding="utf-8",
    )

    assert should_skip_indexing(chunks, options=BuildOptions()) is True
    assert should_skip_indexing(chunks, options=BuildOptions(force_refresh=True)) is False
    assert should_skip_indexing(chunks, options=BuildOptions(scheme_id="large_cap")) is False


def test_validate_corpus_full_build_passes():
    if not list(RAW_DATA_DIR.glob("*.html")):
        pytest.skip("raw HTML not present")

    chunks = chunk_all()
    errors = validate_corpus(chunks)
    assert errors == []
    assert len(chunks) == 35
    assert count_chunks_per_scheme(chunks)["large_cap"] == 7


def test_validate_corpus_detects_missing_large_cap_fact():
    if not list(RAW_DATA_DIR.glob("*.html")):
        pytest.skip("raw HTML not present")

    chunks = chunk_all()
    chunks[0].text = "Expense ratio: 9.99%."
    errors = validate_corpus(chunks)
    assert any("1.04%" in error for error in errors)


def test_run_build_parse_only_skips_fetch_and_index(monkeypatch, tmp_path: Path):
    if not list(RAW_DATA_DIR.glob("*.html")):
        pytest.skip("raw HTML not present")

    monkeypatch.setattr("src.ingestion.pipeline.RAW_DATA_DIR", RAW_DATA_DIR)
    monkeypatch.setattr("src.ingestion.pipeline.PARSED_DATA_DIR", tmp_path / "parsed")
    monkeypatch.setattr("src.ingestion.pipeline.CHUNKS_DATA_DIR", tmp_path / "chunks")
    monkeypatch.setattr("src.ingestion.pipeline.INDEX_DIR", tmp_path / "index")

    with patch("src.ingestion.pipeline.fetch_all") as fetch_mock, patch(
        "src.ingestion.pipeline.build_index"
    ) as index_mock:
        result = run_build(BuildOptions(parse_only=True))

    fetch_mock.assert_not_called()
    index_mock.assert_not_called()
    assert result.parsed_scheme_count == 5
    assert len(result.chunks) == 35
    assert result.validation_errors == []


def test_run_build_index_only_uses_saved_chunks(monkeypatch, tmp_path: Path):
    if not (CHUNKS_DATA_DIR / "chunk_manifest.json").is_file():
        pytest.skip("chunk review files not present")

    monkeypatch.setattr("src.ingestion.pipeline.INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr("src.ingestion.indexer.INDEX_DIR", tmp_path / "index")

    class FakeEmbedder:
        def encode(self, texts, **kwargs):
            import numpy as np

            del kwargs
            return np.vstack([np.ones(384, dtype=np.float32) for _ in texts])

    with patch("src.ingestion.indexer.load_embedding_model", return_value=FakeEmbedder()):
        result = run_build(BuildOptions(index_only=True))

    assert result.index_result is not None
    assert result.index_result.chunk_count == 35
    assert read_index_manifest(tmp_path / "index") is not None


def test_format_build_summary_includes_counts():
    from dataclasses import dataclass, field

    @dataclass
    class FakeResult:
        fetch_results: list = field(default_factory=list)
        parsed_scheme_count: int = 5
        chunks: list = field(default_factory=lambda: [object()] * 35)
        chunks_per_scheme: dict = field(default_factory=lambda: {"large_cap": 7, "mid_cap": 7})
        index_result: object | None = None
        index_skipped: bool = False
        validation_errors: list = field(default_factory=list)

    summary = format_build_summary(FakeResult())
    assert "Chunk: 35 chunk(s)" in summary
    assert "large_cap: 7 chunks" in summary


def test_build_script_main_success(monkeypatch):
    from scripts import build_index as cli

    class FakeResult:
        fetch_results = []
        parsed_scheme_count = 5
        chunks = []
        chunks_per_scheme = {}
        index_result = None
        index_skipped = False
        validation_errors = []

    monkeypatch.setattr(cli, "parse_args", lambda: argparse.Namespace(
        force_refresh=False, scheme=None, parse_only=False, index_only=False
    ))
    monkeypatch.setattr(cli, "run_build", lambda options: FakeResult())
    monkeypatch.setattr(cli, "load_schemes", lambda: [object()] * 5)
    assert cli.main() == 0


def test_build_script_main_validation_failure(monkeypatch):
    from scripts import build_index as cli

    class FakeResult:
        fetch_results = []
        parsed_scheme_count = 0
        chunks = []
        chunks_per_scheme = {}
        index_result = None
        index_skipped = False
        validation_errors = ["boom"]

    monkeypatch.setattr(cli, "parse_args", lambda: argparse.Namespace(
        force_refresh=False, scheme=None, parse_only=False, index_only=False
    ))
    monkeypatch.setattr(cli, "run_build", lambda options: FakeResult())
    monkeypatch.setattr(cli, "load_schemes", lambda: [object()] * 5)
    assert cli.main() == 1
