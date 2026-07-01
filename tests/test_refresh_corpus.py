"""Tests for Phase 6 corpus refresh (scripts/refresh_corpus.py)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.fetcher import FetchResult
from src.ingestion.refresh import (
    REFRESH_MANIFEST_PATH,
    RefreshOptions,
    RefreshResult,
    read_refresh_manifest,
    refresh_exit_code,
    refresh_status,
    run_refresh,
    write_refresh_manifest,
)


def _fetch_result(scheme_id: str, *, skipped: bool) -> FetchResult:
    return FetchResult(
        scheme_id=scheme_id,
        url=f"https://groww.in/mutual-funds/{scheme_id}",
        html_path=Path(f"data/raw/{scheme_id}.html"),
        content_hash="abc123",
        fetched_at="2026-06-29T00:00:00+00:00",
        fetch_method="httpx",
        skipped=skipped,
        bytes_written=0 if skipped else 12000,
    )


def test_refresh_exit_codes():
    assert refresh_exit_code(RefreshResult()) == 0
    assert refresh_exit_code(RefreshResult(errors=["fetch failed"])) == 1
    assert refresh_exit_code(RefreshResult(fatal=True)) == 2


def test_refresh_status_labels():
    assert refresh_status(RefreshResult()) == "success"
    assert refresh_status(RefreshResult(errors=["x"])) == "partial_failure"
    assert refresh_status(RefreshResult(fatal=True)) == "fatal"


def test_write_refresh_manifest_preserves_last_success_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("src.ingestion.refresh.REFRESH_MANIFEST_PATH", tmp_path / "refresh_manifest.json")
    prior = {
        "last_success_at": "2026-06-28T10:00:00+05:30",
        "status": "success",
    }
    (tmp_path / "refresh_manifest.json").write_text(json.dumps(prior), encoding="utf-8")

    result = RefreshResult(schemes_checked=5, schemes_skipped=5, errors=["boom"])
    write_refresh_manifest(result, options=RefreshOptions(), status="partial_failure")

    manifest = read_refresh_manifest(tmp_path / "refresh_manifest.json")
    assert manifest["last_success_at"] == "2026-06-28T10:00:00+05:30"
    assert manifest["status"] == "partial_failure"


def test_dry_run_skips_parse_and_index(monkeypatch, tmp_path):
    monkeypatch.setattr("src.ingestion.refresh.RAW_DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr("src.ingestion.refresh.PARSED_DATA_DIR", tmp_path / "parsed")
    monkeypatch.setattr("src.ingestion.refresh.CHUNKS_DATA_DIR", tmp_path / "chunks")
    monkeypatch.setattr("src.ingestion.refresh.INDEX_DIR", tmp_path / "index")

    fetch_results = [_fetch_result("large_cap", skipped=True)]

    with patch("src.ingestion.refresh.fetch_all", return_value=fetch_results) as fetch_mock, patch(
        "src.ingestion.refresh.parse_all"
    ) as parse_mock, patch("src.ingestion.refresh.build_index") as index_mock:
        result = run_refresh(RefreshOptions(dry_run=True))

    fetch_mock.assert_called_once()
    parse_mock.assert_not_called()
    index_mock.assert_not_called()
    assert result.schemes_skipped == 1
    assert result.schemes_updated == 0


def test_unchanged_hash_skips_parse_and_index(monkeypatch, tmp_path):
    monkeypatch.setattr("src.ingestion.refresh.RAW_DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr("src.ingestion.refresh.PARSED_DATA_DIR", tmp_path / "parsed")
    monkeypatch.setattr("src.ingestion.refresh.CHUNKS_DATA_DIR", tmp_path / "chunks")
    monkeypatch.setattr("src.ingestion.refresh.INDEX_DIR", tmp_path / "index")

    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    (chunks_dir / "large_cap.json").write_text(
        json.dumps(
            {
                "scheme_id": "large_cap",
                "scheme_name": "HDFC Large Cap",
                "source_url": "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
                "last_updated": "2026-06-28",
                "chunk_count": 1,
                "chunks": [
                    {
                        "chunk_id": "large_cap_groww_0001",
                        "scheme_id": "large_cap",
                        "scheme_name": "HDFC Large Cap",
                        "document_type": "groww_fund_page",
                        "source_url": "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
                        "source_domain": "groww.in",
                        "section_title": "Key fund facts",
                        "last_updated": "2026-06-28",
                        "text": "Expense ratio 1.04%. Minimum SIP ₹100.",
                        "embed_text": "HDFC Large Cap | Key fund facts | Expense ratio 1.04%.",
                        "token_estimate": 10,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    fetch_results = [_fetch_result("large_cap", skipped=True)]

    with patch("src.ingestion.refresh.fetch_all", return_value=fetch_results), patch(
        "src.ingestion.refresh.parse_all"
    ) as parse_mock, patch("src.ingestion.refresh.index_chunks") as index_mock, patch(
        "src.ingestion.refresh.build_index"
    ) as build_mock, patch(
        "src.ingestion.refresh.get_index_stats",
        return_value={
            "total_chunks": 1,
            "chunks_per_scheme": {"large_cap": 1},
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "collection_name": "hdfc_mf_groww_corpus",
        },
    ), patch(
        "src.ingestion.refresh.read_index_manifest",
        return_value={"embedded_at": "2026-06-28", "chunks_fingerprint": "fp"},
    ), patch(
        "src.ingestion.refresh.should_skip_indexing",
        return_value=True,
    ):
        result = run_refresh(RefreshOptions(scheme_id="large_cap"))

    parse_mock.assert_not_called()
    index_mock.assert_not_called()
    build_mock.assert_not_called()
    assert result.index_skipped is True
    assert result.schemes_updated == 0


def test_changed_hash_reindexes_scheme(monkeypatch, tmp_path):
    monkeypatch.setattr("src.ingestion.refresh.RAW_DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr("src.ingestion.refresh.PARSED_DATA_DIR", tmp_path / "parsed")
    monkeypatch.setattr("src.ingestion.refresh.CHUNKS_DATA_DIR", tmp_path / "chunks")
    monkeypatch.setattr("src.ingestion.refresh.INDEX_DIR", tmp_path / "index")

    from src.ingestion.chunker import Chunk

    fake_chunk = Chunk(
        chunk_id="large_cap_groww_0001",
        scheme_id="large_cap",
        scheme_name="HDFC Large Cap",
        document_type="groww_fund_page",
        source_url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
        source_domain="groww.in",
        section_title="Key fund facts",
        last_updated="2026-06-29",
        text="Expense ratio 1.04%. Minimum SIP ₹100.",
        embed_text="HDFC Large Cap | Key fund facts | Expense ratio 1.04%.",
        token_estimate=10,
    )

    fetch_results = [_fetch_result("large_cap", skipped=False)]
    index_result = MagicMock(chunk_count=1, scheme_ids=["large_cap"])

    with patch("src.ingestion.refresh.fetch_all", return_value=fetch_results), patch(
        "src.ingestion.refresh.parse_all",
        return_value=[MagicMock(scheme_id="large_cap")],
    ), patch("src.ingestion.refresh.save_parsed_pages"), patch(
        "src.ingestion.refresh.chunk_all",
        return_value=[fake_chunk],
    ), patch("src.ingestion.refresh.save_chunks"), patch(
        "src.ingestion.refresh.load_saved_chunks",
        return_value=[fake_chunk],
    ), patch("src.ingestion.refresh.should_skip_indexing", return_value=False), patch(
        "src.ingestion.refresh.read_index_manifest",
        return_value={"embedded_at": "2026-06-28"},
    ), patch(
        "src.ingestion.refresh.index_chunks",
        return_value=index_result,
    ) as index_mock, patch(
        "src.ingestion.refresh.validate_corpus",
        return_value=[],
    ):
        result = run_refresh(RefreshOptions(scheme_id="large_cap"))

    index_mock.assert_called_once()
    assert result.schemes_updated == 1
    assert result.schemes_reindexed == 1


def test_refresh_cli_writes_manifest(monkeypatch, tmp_path):
    import argparse

    from scripts import refresh_corpus as cli

    monkeypatch.setattr("src.ingestion.refresh.REFRESH_MANIFEST_PATH", tmp_path / "refresh_manifest.json")

    fake_result = RefreshResult(schemes_checked=5, schemes_skipped=5, chunk_count=35)

    monkeypatch.setattr(cli, "parse_args", lambda: argparse.Namespace(
        force_refresh=False,
        scheme=None,
        dry_run=False,
        trigger="manual",
        workflow_run_id=None,
    ))
    monkeypatch.setattr(cli, "run_refresh", lambda options: fake_result)

    assert cli.main() == 0
    manifest = json.loads((tmp_path / "refresh_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "success"
    assert manifest["schemes_checked"] == 5
