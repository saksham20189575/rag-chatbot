"""Per-client request rate limiting aligned with Groq RPM guardrails."""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque

from src.constants import DEFAULT_GROQ_RPM_LIMIT

MINUTE_WINDOW = 60.0


class RateLimitExceeded(Exception):
    """Raised when a client exceeds the configured requests-per-minute cap."""


class ApiRateLimiter:
    """In-process sliding-window limiter keyed by client identifier (e.g. IP)."""

    def __init__(
        self,
        *,
        limit: int | None = None,
        window: float = MINUTE_WINDOW,
    ) -> None:
        raw = os.getenv("API_RPM_LIMIT") or os.getenv("GROQ_RPM_LIMIT")
        self.limit = limit if limit is not None else int(raw or DEFAULT_GROQ_RPM_LIMIT)
        self.window = window
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def check(self, client_key: str) -> None:
        now = time.monotonic()
        events = self._events[client_key]
        cutoff = now - self.window
        while events and events[0] <= cutoff:
            events.popleft()
        if len(events) >= self.limit:
            raise RateLimitExceeded(
                f"Rate limit exceeded: {self.limit} requests per {int(self.window)} seconds."
            )
        events.append(now)

    def remaining(self, client_key: str) -> int:
        now = time.monotonic()
        events = self._events[client_key]
        cutoff = now - self.window
        while events and events[0] <= cutoff:
            events.popleft()
        return max(0, self.limit - len(events))
