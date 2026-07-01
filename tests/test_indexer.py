"""Unit tests for src.ingestion.indexer."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.constants import BGE_QUERY_PREFIX, CHROMA_COLLECTION_NAME, DEFAULT_BGE_MODEL
from src.ingestion.chunker import Chunk, chunk_page, save_chunks
from src.ingestion.fetcher import SchemeConfig
from src.ingestion.indexer import (
    INDEX_MANIFEST_FILENAME,
    build_index,
    embed_query_text,
    index_chunks,
    load_saved_chunks,
    read_index_manifest,
    resolve_bge_model_name,
)
from src.ingestion.parser import ParsedPage, TextBlock

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


class FakeEmbedder:
    """Deterministic 384-d vectors for tests (no model download)."""

    def encode(self, texts, **kwargs):
        del kwargs
        rows = []
        for text in texts:
            seed = sum(ord(c) for c in text) % 997
            row = np.zeros(384, dtype=np.float32)
            row[0] = float(seed)
            row[1] = float(len(text))
            norm = np.linalg.norm(row) or 1.0
            rows.append(row / norm)
        return np.vstack(rows)


def _sample_page(scheme_id: str = "large_cap") -> ParsedPage:
    return ParsedPage(
        scheme_id=scheme_id,
        text_blocks=[
            TextBlock("Key fund facts", "Expense ratio: 1.04%. Minimum SIP investment: ₹100."),
            TextBlock("Exit load", "Exit load of 1% if redeemed within 1 year"),
        ],
        nav_date="25-Jun-2026",
        fetched_at="2026-06-28T05:55:41+00:00",
    )


def _sample_chunks(scheme: SchemeConfig) -> list[Chunk]:
    return chunk_page(_sample_page(scheme.scheme_id), scheme=scheme)


def test_resolve_bge_model_name_default():
    with patch.dict("os.environ", {}, clear=True):
        assert resolve_bge_model_name() == DEFAULT_BGE_MODEL


def test_resolve_bge_model_name_from_env():
    with patch.dict("os.environ", {"BGE_MODEL_NAME": "BAAI/bge-large-en-v1.5"}):
        assert resolve_bge_model_name() == "BAAI/bge-large-en-v1.5"


def test_embed_query_text_uses_bge_prefix():
    model = FakeEmbedder()
    plain = embed_query_text(model, "expense ratio")  # type: ignore[arg-type]
    prefixed = model.encode([f"{BGE_QUERY_PREFIX}expense ratio"], normalize_embeddings=True)
    np.testing.assert_allclose(plain, prefixed[0].tolist())


def test_load_saved_chunks_round_trip(tmp_path: Path):
    chunks = _sample_chunks(LARGE_CAP)
    save_chunks(chunks, chunks_dir=tmp_path)
    loaded = load_saved_chunks(chunks_dir=tmp_path)
    assert len(loaded) == len(chunks)
    assert loaded[0].chunk_id == chunks[0].chunk_id
    assert loaded[0].embed_text == chunks[0].embed_text


def test_index_chunks_writes_chroma_and_manifest(tmp_path: Path):
    chunks = _sample_chunks(LARGE_CAP)
    result = index_chunks(
        chunks,
        model=FakeEmbedder(),  # type: ignore[arg-type]
        index_dir=tmp_path,
    )

    assert result.chunk_count == len(chunks)
    assert result.embedding_model == DEFAULT_BGE_MODEL
    assert result.scheme_ids == ["large_cap"]

    manifest = read_index_manifest(tmp_path)
    assert manifest is not None
    assert manifest["chunk_count"] == len(chunks)
    assert manifest["embedding_model"] == DEFAULT_BGE_MODEL
    assert manifest["collection_name"] == CHROMA_COLLECTION_NAME

    import chromadb

    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_collection(CHROMA_COLLECTION_NAME)
    stored = collection.get(include=["documents", "metadatas", "embeddings"])
    assert len(stored["ids"]) == len(chunks)
    assert stored["documents"][0] == chunks[0].text
    assert stored["metadatas"][0]["section_title"] == "Key fund facts"
    assert stored["metadatas"][0]["text"] == chunks[0].text
    assert len(stored["embeddings"][0]) == 384


def test_index_chunks_embeds_embed_text_not_raw_text(tmp_path: Path):
    chunks = _sample_chunks(LARGE_CAP)
    seen: list[str] = []

    class CaptureEmbedder(FakeEmbedder):
        def encode(self, texts, **kwargs):
            seen.extend(texts)
            return super().encode(texts, **kwargs)

    index_chunks(chunks, model=CaptureEmbedder(), index_dir=tmp_path)  # type: ignore[arg-type]
    assert seen == [c.embed_text for c in chunks]
    assert all("| Key fund facts:" in t or "| Exit load:" in t for t in seen)


def test_index_chunks_single_scheme_preserves_other_schemes(tmp_path: Path):
    large = _sample_chunks(LARGE_CAP)
    mid = _sample_chunks(MID_CAP)
    index_chunks(large + mid, model=FakeEmbedder(), index_dir=tmp_path)  # type: ignore[arg-type]

    updated_large = [
        Chunk(
            **{
                **large[0].to_dict(),
                "text": "Expense ratio: 1.05%. Minimum SIP investment: ₹100.",
                "embed_text": "HDFC Large Cap Fund Direct Growth | Key fund facts: Expense ratio: 1.05%. Minimum SIP investment: ₹100.",
            }
        ),
        large[1],
    ]
    index_chunks(
        updated_large,
        model=FakeEmbedder(),  # type: ignore[arg-type]
        index_dir=tmp_path,
        scheme_id="large_cap",
    )

    import chromadb

    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_collection(CHROMA_COLLECTION_NAME)
    assert collection.count() == len(large) + len(mid)

    large_docs = collection.get(where={"scheme_id": "large_cap"}, include=["documents"])
    assert any("1.05%" in doc for doc in large_docs["documents"])


def test_index_chunks_rejects_model_mismatch_on_partial_update(tmp_path: Path):
    chunks = _sample_chunks(LARGE_CAP)
    index_chunks(chunks, model=FakeEmbedder(), index_dir=tmp_path)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Index was built with"):
        index_chunks(
            chunks,
            model=FakeEmbedder(),  # type: ignore[arg-type]
            index_dir=tmp_path,
            scheme_id="large_cap",
            model_name="BAAI/bge-large-en-v1.5",
        )


def test_build_index_from_saved_chunks(tmp_path: Path):
    save_chunks(_sample_chunks(LARGE_CAP), chunks_dir=tmp_path)
    with patch("src.ingestion.indexer.load_embedding_model", return_value=FakeEmbedder()):
        result = build_index(chunks_dir=tmp_path, index_dir=tmp_path)
    assert result.chunk_count == 2
    assert (tmp_path / INDEX_MANIFEST_FILENAME).is_file()


def test_build_index_real_corpus(tmp_path: Path):
    from src.constants import CHUNKS_DATA_DIR

    if not (CHUNKS_DATA_DIR / "chunk_manifest.json").is_file():
        pytest.skip("chunk review files not present")

    with patch("src.ingestion.indexer.load_embedding_model", return_value=FakeEmbedder()):
        result = build_index(chunks_dir=CHUNKS_DATA_DIR, index_dir=tmp_path)
    assert result.chunk_count == 35
    assert len(result.scheme_ids) == 5

    manifest = json.loads((tmp_path / INDEX_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["chunk_count"] == 35
