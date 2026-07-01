"""Tests for the Streamlit UI API client."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from src.ui.api_client import ChatApiError, get_health, post_chat


def test_post_chat_returns_payload():
    mock_response = httpx.Response(
        200,
        json={
            "type": "answer",
            "text": "Minimum SIP is ₹100.",
            "citation_url": "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
            "last_updated": "2026-06-25",
            "disclaimer": "Facts-only. No investment advice.",
        },
        request=httpx.Request("POST", "http://testserver/chat"),
    )
    with patch("src.ui.api_client.httpx.post", return_value=mock_response):
        payload = post_chat("What is min SIP?", base_url="http://testserver")

    assert payload["type"] == "answer"
    assert "₹100" in payload["text"]


def test_post_chat_raises_on_429():
    mock_response = httpx.Response(
        429,
        json={"detail": "Rate limit exceeded."},
        request=httpx.Request("POST", "http://testserver/chat"),
    )
    with patch("src.ui.api_client.httpx.post", return_value=mock_response):
        with pytest.raises(ChatApiError) as exc_info:
            post_chat("hello", base_url="http://testserver")

    assert exc_info.value.status_code == 429


def test_get_health_returns_index_stats():
    mock_response = httpx.Response(
        200,
        json={
            "status": "ok",
            "index": {"chunk_count": 35, "embedded_at": "2026-06-28T06:28:32+00:00"},
        },
        request=httpx.Request("GET", "http://testserver/health"),
    )
    with patch("src.ui.api_client.httpx.get", return_value=mock_response):
        payload = get_health("http://testserver")

    assert payload["index"]["chunk_count"] == 35


def test_get_health_raises_when_unreachable():
    with patch(
        "src.ui.api_client.httpx.get",
        side_effect=httpx.ConnectError("Connection refused"),
    ):
        with pytest.raises(ChatApiError, match="Cannot reach API"):
            get_health("http://testserver")
