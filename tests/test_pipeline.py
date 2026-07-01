"""Tests for src.rag.pipeline orchestration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.rag.groq_client import GroqGenerationError, GroqQuotaExceeded
from src.rag.models import DraftAnswer
from src.rag.pipeline import PipelineStats, answer
from src.rag.retriever import RetrievedChunk, RetrievalResult


def _chunk(scheme_id: str = "mid_cap") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{scheme_id}_groww_0000",
        scheme_id=scheme_id,
        scheme_name="HDFC Mid Cap Fund Direct Growth",
        section_title="Key fund facts",
        text="Expense ratio: 1.23%. Minimum SIP investment: ₹100.",
        source_url="https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
        last_updated="2026-06-25",
        score=0.9,
    )


def _retrieval(scheme_id: str = "mid_cap") -> RetrievalResult:
    chunk = _chunk(scheme_id)
    return RetrievalResult(
        query="expense ratio",
        expanded_query="expense ratio",
        scheme_id=scheme_id,
        chunks=[chunk],
        context=f"[{chunk.source_url} | {chunk.last_updated} | {chunk.section_title}]\n{chunk.text}",
    )


def test_advisory_query_skips_retrieval_and_groq():
    stats = PipelineStats()
    with patch("src.rag.pipeline.Retriever") as mock_retriever_cls:
        response = answer("Should I invest in HDFC Gold FoF?", stats=stats)
    mock_retriever_cls.assert_not_called()
    assert stats.groq_calls == 0
    assert response.type == "refusal"


def test_performance_query_skips_groq():
    stats = PipelineStats()
    with patch("src.rag.pipeline.Retriever") as mock_retriever_cls:
        response = answer("3-year return of Mid Cap", stats=stats)
    mock_retriever_cls.assert_not_called()
    assert stats.groq_calls == 0
    assert response.type == "refusal"
    assert "%" not in response.text


def test_empty_retrieval_returns_refusal_without_groq():
    stats = PipelineStats()
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = RetrievalResult(
        query="What is the expense ratio?",
        expanded_query="What is the expense ratio?",
        scheme_id=None,
        chunks=[],
        context="",
    )
    response = answer("What is the expense ratio?", retriever=mock_retriever, stats=stats)
    assert stats.groq_calls == 0
    assert response.type == "refusal"


def test_factual_query_calls_groq_once_and_returns_answer():
    stats = PipelineStats()
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = _retrieval("mid_cap")
    mock_groq = MagicMock()
    mock_groq.complete.return_value = DraftAnswer(
        text="The expense ratio is 1.23%. Minimum SIP is ₹100.",
        citation_url="https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
        last_updated="2026-06-25",
    )

    with patch("src.rag.pipeline.generate_answer", return_value=mock_groq.complete.return_value):
        with patch("src.rag.pipeline.GroqClient", return_value=mock_groq):
            response = answer(
                "What is expense ratio of Mid Cap?",
                retriever=mock_retriever,
                groq_client=mock_groq,
                stats=stats,
            )

    assert stats.groq_calls == 1
    assert response.type == "answer"
    assert "groww.in/mutual-funds/hdfc-mid-cap" in response.citation_url


def test_groq_quota_exceeded_returns_refusal():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = _retrieval()
    with patch("src.rag.pipeline.generate_answer", side_effect=GroqQuotaExceeded()):
        response = answer(
            "What is expense ratio of Mid Cap?",
            retriever=mock_retriever,
            groq_client=MagicMock(),
        )
    assert response.type == "refusal"


def test_groq_error_after_retry_returns_refusal():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = _retrieval()
    with patch("src.rag.pipeline.generate_answer", side_effect=GroqGenerationError("429")):
        response = answer(
            "What is expense ratio of Mid Cap?",
            retriever=mock_retriever,
            groq_client=MagicMock(),
        )
    assert response.type == "refusal"


def test_validator_failure_does_not_recall_groq():
    stats = PipelineStats()
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = _retrieval()
    bad_draft = DraftAnswer(
        text="I recommend this fund. It returned 21%. Great pick. Another sentence.",
        citation_url="https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
        last_updated="2026-06-25",
    )
    with patch("src.rag.pipeline.generate_answer", return_value=bad_draft):
        response = answer(
            "What is expense ratio of Mid Cap?",
            retriever=mock_retriever,
            groq_client=MagicMock(),
            stats=stats,
        )
    assert stats.groq_calls == 1
    assert response.type == "refusal"
