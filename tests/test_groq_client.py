"""Tests for Groq quota limiter and client retry behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from groq import APIStatusError

from src.rag.groq_client import (
    GroqClient,
    GroqConfig,
    GroqGenerationError,
    GroqQuotaExceeded,
    GroqQuotaLimiter,
)


def test_estimate_tokens_over_estimates_short_prompt():
    assert GroqQuotaLimiter.estimate_tokens("abcd") >= 1


def test_limiter_blocks_when_rpm_exceeded():
    config = GroqConfig(rpm_limit=1, tpm_limit=10_000, rpd_limit=10, tpd_limit=10_000)
    limiter = GroqQuotaLimiter(config)
    limiter.record_usage(100)
    with pytest.raises(GroqQuotaExceeded):
        limiter.check_quota(50)


def test_limiter_blocks_when_tpm_exceeded():
    config = GroqConfig(rpm_limit=100, tpm_limit=100, rpd_limit=100, tpd_limit=100_000)
    limiter = GroqQuotaLimiter(config)
    limiter.record_usage(90)
    with pytest.raises(GroqQuotaExceeded):
        limiter.check_quota(20)


def test_get_quota_status_reports_remaining():
    limiter = GroqQuotaLimiter(GroqConfig(rpm_limit=5, tpm_limit=1000))
    status = limiter.get_quota_status()
    assert status["rpm_limit"] == 5
    assert status["rpm_remaining"] == 5


def _api_status_error(status_code: int, *, retry_after: str | None = None) -> APIStatusError:
    headers = {"retry-after": retry_after} if retry_after else {}
    request = SimpleNamespace()
    response = SimpleNamespace(status_code=status_code, headers=headers, request=request)
    return APIStatusError("rate limited", response=response, body=None)


def test_groq_client_retries_once_on_429_then_succeeds():
    mock_groq = MagicMock()
    success = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Answer text."))],
        usage=SimpleNamespace(total_tokens=120),
    )
    mock_groq.chat.completions.create.side_effect = [
        _api_status_error(429, retry_after="0"),
        success,
    ]
    sleeps: list[float] = []

    client = GroqClient(
        client=mock_groq,
        config=GroqConfig(rpm_limit=10, tpm_limit=10_000, rpd_limit=100, tpd_limit=100_000),
        api_key="test-key",
        sleeper=sleeps.append,
    )
    result = client.complete(system_prompt="sys", user_prompt="user question")
    assert result.text == "Answer text."
    assert mock_groq.chat.completions.create.call_count == 2
    assert sleeps == [0.0]


def test_groq_client_raises_after_second_429():
    mock_groq = MagicMock()
    mock_groq.chat.completions.create.side_effect = [
        _api_status_error(429),
        _api_status_error(429),
    ]
    client = GroqClient(
        client=mock_groq,
        config=GroqConfig(rpm_limit=10, tpm_limit=10_000, rpd_limit=100, tpd_limit=100_000),
        api_key="test-key",
        sleeper=lambda _: None,
    )
    with pytest.raises(GroqGenerationError):
        client.complete(system_prompt="sys", user_prompt="user question")
    assert mock_groq.chat.completions.create.call_count == 2
