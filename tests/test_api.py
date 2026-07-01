"""Tests for the Phase 4 FastAPI chat API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.rate_limit import ApiRateLimiter
from src.constants import DISCLAIMER
from src.rag.models import ChatResponse


@asynccontextmanager
async def _test_lifespan(app):
    app.state.retriever = MagicMock()
    app.state.groq_client = MagicMock()
    app.state.groq_client.get_quota_status.return_value = {
        "rpm_remaining": 25,
        "tpm_remaining": 10000,
    }
    app.state.allowlisted_urls = frozenset()
    app.state.rate_limiter = ApiRateLimiter(limit=25)
    app.state.index_ready = True
    yield


@pytest.fixture
def client():
    app = create_app(lifespan_override=_test_lifespan)
    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_index_quota_and_refresh(client):
    with patch("src.api.main.get_index_stats") as mock_stats, patch(
        "src.api.main.read_refresh_manifest"
    ) as mock_refresh:
        mock_stats.return_value = {
            "total_chunks": 35,
            "embedded_at": "2026-06-28T06:28:32+00:00",
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "collection_name": "hdfc_mf_groww_corpus",
        }
        mock_refresh.return_value = {
            "last_success_at": "2026-06-29T10:30:00+05:30",
            "status": "success",
        }
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["index"]["chunk_count"] == 35
    assert payload["last_refreshed_at"] == "2026-06-29T10:30:00+05:30"
    assert payload["refresh_status"] == "success"
    assert payload["groq_quota"]["rpm_remaining"] == 25


def test_health_returns_503_when_index_not_ready(client):
    client.app.state.index_ready = False
    response = client.get("/health")
    assert response.status_code == 503


def test_chat_rejects_empty_message(client):
    response = client.post("/chat", json={"message": "   "})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_chat_rejects_overlong_message(client):
    response = client.post("/chat", json={"message": "x" * 501})
    assert response.status_code == 400
    assert "500" in response.json()["detail"]


def test_chat_strips_html(client):
    with patch("src.api.main.answer") as mock_answer:
        mock_answer.return_value = ChatResponse(
            type="answer",
            text="Minimum SIP is ₹100.",
            citation_url="https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
            last_updated="2026-06-28",
            disclaimer=DISCLAIMER,
        )
        response = client.post(
            "/chat",
            json={"message": "<script>alert(1)</script>What is min SIP for <b>Small Cap</b>?"},
        )

    assert response.status_code == 200
    called_query = mock_answer.call_args[0][0]
    assert "<" not in called_query
    assert "script" not in called_query.lower()


def test_chat_pii_returns_refusal_without_pipeline(client):
    with patch("src.api.main.answer") as mock_answer:
        response = client.post("/chat", json={"message": "My PAN is ABCDE1234F for Large Cap"})

    mock_answer.assert_not_called()
    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "refusal"
    assert "personal information" in payload["text"].lower()


def test_chat_factual_query_returns_pipeline_response(client):
    expected = ChatResponse(
        type="answer",
        text="The minimum SIP is ₹100.",
        citation_url="https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
        last_updated="2026-06-28",
        disclaimer=DISCLAIMER,
    )
    with patch("src.api.main.answer", return_value=expected) as mock_answer:
        response = client.post(
            "/chat",
            json={"message": "What is the min SIP for HDFC Small Cap Fund?"},
        )

    mock_answer.assert_called_once()
    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "answer"
    assert payload["disclaimer"] == DISCLAIMER
    assert "groww.in" in payload["citation_url"]


def test_chat_rate_limit_returns_429(client):
    client.app.state.rate_limiter = ApiRateLimiter(limit=2)
    for _ in range(2):
        with patch("src.api.main.answer") as mock_answer:
            mock_answer.return_value = ChatResponse(
                type="refusal",
                text="ok",
                citation_url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
                last_updated="2026-06-28",
                disclaimer=DISCLAIMER,
            )
            assert client.post("/chat", json={"message": "What is expense ratio of Large Cap?"}).status_code == 200

    response = client.post("/chat", json={"message": "What is expense ratio of Large Cap?"})
    assert response.status_code == 429


def test_chat_returns_503_when_index_not_ready(client):
    client.app.state.index_ready = False
    response = client.post("/chat", json={"message": "What is expense ratio of Large Cap?"})
    assert response.status_code == 503


def test_chat_internal_error_returns_generic_500(client):
    with patch("src.api.main.answer", side_effect=RuntimeError("boom")):
        response = client.post("/chat", json={"message": "What is expense ratio of Large Cap?"})

    assert response.status_code == 500
    assert response.json()["detail"] == "An internal error occurred while processing your request."
    assert "boom" not in response.json()["detail"]
