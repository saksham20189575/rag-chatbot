"""Groq SDK wrapper with sliding-window quota limits and 429 retry."""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from groq import APIStatusError, Groq

from src.constants import (
    DEFAULT_GROQ_MAX_TOKENS,
    DEFAULT_GROQ_MODEL,
    DEFAULT_GROQ_REQUEST_TIMEOUT,
    DEFAULT_GROQ_RPD_LIMIT,
    DEFAULT_GROQ_RPM_LIMIT,
    DEFAULT_GROQ_TEMPERATURE,
    DEFAULT_GROQ_TPD_LIMIT,
    DEFAULT_GROQ_TPM_LIMIT,
)

logger = logging.getLogger(__name__)

MINUTE_WINDOW = 60.0
DAY_WINDOW = 86_400.0


class GroqQuotaExceeded(Exception):
    """Raised when a client-side quota window would be exceeded."""


class GroqGenerationError(Exception):
    """Raised when Groq fails after retry (429, timeout, 5xx)."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class GroqConfig:
    model: str = DEFAULT_GROQ_MODEL
    max_tokens: int = DEFAULT_GROQ_MAX_TOKENS
    temperature: float = DEFAULT_GROQ_TEMPERATURE
    timeout: float = DEFAULT_GROQ_REQUEST_TIMEOUT
    rpm_limit: int = DEFAULT_GROQ_RPM_LIMIT
    tpm_limit: int = DEFAULT_GROQ_TPM_LIMIT
    rpd_limit: int = DEFAULT_GROQ_RPD_LIMIT
    tpd_limit: int = DEFAULT_GROQ_TPD_LIMIT

    @classmethod
    def from_env(cls) -> GroqConfig:
        def _int(name: str, default: int) -> int:
            raw = os.getenv(name)
            return int(raw) if raw else default

        def _float(name: str, default: float) -> float:
            raw = os.getenv(name)
            return float(raw) if raw else default

        return cls(
            model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            max_tokens=_int("GROQ_MAX_TOKENS", DEFAULT_GROQ_MAX_TOKENS),
            temperature=_float("GROQ_TEMPERATURE", DEFAULT_GROQ_TEMPERATURE),
            timeout=_float("GROQ_REQUEST_TIMEOUT", DEFAULT_GROQ_REQUEST_TIMEOUT),
            rpm_limit=_int("GROQ_RPM_LIMIT", DEFAULT_GROQ_RPM_LIMIT),
            tpm_limit=_int("GROQ_TPM_LIMIT", DEFAULT_GROQ_TPM_LIMIT),
            rpd_limit=_int("GROQ_RPD_LIMIT", DEFAULT_GROQ_RPD_LIMIT),
            tpd_limit=_int("GROQ_TPD_LIMIT", DEFAULT_GROQ_TPD_LIMIT),
        )


@dataclass
class _QuotaWindow:
    request_events: deque[float] = field(default_factory=deque)
    token_events: deque[tuple[float, int]] = field(default_factory=deque)

    def _prune(self, now: float, window: float) -> None:
        cutoff = now - window
        while self.request_events and self.request_events[0] <= cutoff:
            self.request_events.popleft()
        while self.token_events and self.token_events[0][0] <= cutoff:
            self.token_events.popleft()

    def request_count(self, now: float, window: float) -> int:
        self._prune(now, window)
        return len(self.request_events)

    def token_count(self, now: float, window: float) -> int:
        self._prune(now, window)
        return sum(tokens for _, tokens in self.token_events)

    def would_exceed(
        self,
        *,
        now: float,
        window: float,
        request_limit: int,
        token_limit: int,
        estimated_tokens: int,
    ) -> bool:
        self._prune(now, window)
        if len(self.request_events) + 1 > request_limit:
            return True
        current_tokens = sum(tokens for _, tokens in self.token_events)
        return current_tokens + estimated_tokens > token_limit

    def record(self, now: float, tokens: int) -> None:
        self.request_events.append(now)
        self.token_events.append((now, tokens))


class GroqQuotaLimiter:
    """In-process sliding-window limiter with headroom under Groq free-tier caps."""

    def __init__(self, config: GroqConfig | None = None) -> None:
        self.config = config or GroqConfig.from_env()
        self.minute = _QuotaWindow()
        self.day = _QuotaWindow()
        self.last_429_at: float | None = None

    @staticmethod
    def estimate_tokens(*texts: str) -> int:
        combined = "\n".join(texts)
        return max(1, int(len(combined) / 4 * 1.1))

    def check_quota(self, estimated_tokens: int) -> None:
        now = time.monotonic()
        if self.minute.would_exceed(
            now=now,
            window=MINUTE_WINDOW,
            request_limit=self.config.rpm_limit,
            token_limit=self.config.tpm_limit,
            estimated_tokens=estimated_tokens,
        ):
            raise GroqQuotaExceeded("Groq per-minute quota would be exceeded")
        if self.day.would_exceed(
            now=now,
            window=DAY_WINDOW,
            request_limit=self.config.rpd_limit,
            token_limit=self.config.tpd_limit,
            estimated_tokens=estimated_tokens,
        ):
            raise GroqQuotaExceeded("Groq daily quota would be exceeded")

    def record_usage(self, tokens: int) -> None:
        now = time.monotonic()
        self.minute.record(now, tokens)
        self.day.record(now, tokens)

    def get_quota_status(self) -> dict[str, Any]:
        now = time.monotonic()
        minute_requests = self.minute.request_count(now, MINUTE_WINDOW)
        minute_tokens = self.minute.token_count(now, MINUTE_WINDOW)
        day_requests = self.day.request_count(now, DAY_WINDOW)
        day_tokens = self.day.token_count(now, DAY_WINDOW)
        return {
            "rpm_used": minute_requests,
            "rpm_limit": self.config.rpm_limit,
            "rpm_remaining": max(0, self.config.rpm_limit - minute_requests),
            "tpm_used": minute_tokens,
            "tpm_limit": self.config.tpm_limit,
            "tpm_remaining": max(0, self.config.tpm_limit - minute_tokens),
            "rpd_used": day_requests,
            "rpd_limit": self.config.rpd_limit,
            "rpd_remaining": max(0, self.config.rpd_limit - day_requests),
            "tpd_used": day_tokens,
            "tpd_limit": self.config.tpd_limit,
            "tpd_remaining": max(0, self.config.tpd_limit - day_tokens),
            "last_429_at": self.last_429_at,
        }


@dataclass
class GroqCompletionResult:
    text: str
    total_tokens: int


class GroqClient:
    """Quota-aware Groq chat client with a single 429 retry."""

    def __init__(
        self,
        *,
        client: Groq | None = None,
        config: GroqConfig | None = None,
        limiter: GroqQuotaLimiter | None = None,
        api_key: str | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or GroqConfig.from_env()
        self.limiter = limiter or GroqQuotaLimiter(self.config)
        resolved_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = client if client is not None else Groq(api_key=resolved_key)
        self._sleeper = sleeper

    def get_quota_status(self) -> dict[str, Any]:
        return self.limiter.get_quota_status()

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> GroqCompletionResult:
        estimated = self.limiter.estimate_tokens(
            system_prompt,
            user_prompt,
            " " * self.config.max_tokens,
        )
        self.limiter.check_quota(estimated)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        last_error: Exception | None = None

        for attempt in range(2):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout,
                )
                text = (response.choices[0].message.content or "").strip()
                usage = getattr(response, "usage", None)
                total_tokens = estimated
                if usage is not None and getattr(usage, "total_tokens", None):
                    total_tokens = int(usage.total_tokens)
                self.limiter.record_usage(total_tokens)
                return GroqCompletionResult(text=text, total_tokens=total_tokens)
            except APIStatusError as exc:
                last_error = exc
                if exc.status_code == 429 and attempt == 0:
                    self.limiter.last_429_at = time.monotonic()
                    retry_after = 2.0
                    if exc.response is not None:
                        header = exc.response.headers.get("retry-after")
                        if header:
                            try:
                                retry_after = float(header)
                            except ValueError:
                                retry_after = 2.0
                    logger.warning("Groq 429 — retrying once after %.1fs", retry_after)
                    self._sleeper(retry_after)
                    continue
                raise GroqGenerationError(str(exc), status_code=exc.status_code) from exc
            except Exception as exc:
                raise GroqGenerationError(str(exc)) from exc

        assert last_error is not None
        status_code = last_error.status_code if isinstance(last_error, APIStatusError) else None
        raise GroqGenerationError(str(last_error), status_code=status_code)