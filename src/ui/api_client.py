"""HTTP client for the Phase 4 chat API."""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_API_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 30.0


class ChatApiError(Exception):
    """Raised when the chat API returns an error or is unreachable."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def resolve_api_base_url() -> str:
    return os.getenv("CHAT_API_URL", DEFAULT_API_BASE_URL).rstrip("/")


def get_health(base_url: str | None = None, *, timeout: float = 5.0) -> dict[str, Any]:
    """Return /health payload or raise ChatApiError."""
    url = f"{(base_url or resolve_api_base_url())}/health"
    try:
        response = httpx.get(url, timeout=timeout)
    except httpx.RequestError as exc:
        raise ChatApiError(f"Cannot reach API at {url}: {exc}") from exc

    if response.status_code == 503:
        raise ChatApiError("API is running but the vector index is not available.", status_code=503)
    if response.status_code != 200:
        raise ChatApiError(
            f"Health check failed with status {response.status_code}.",
            status_code=response.status_code,
        )
    return response.json()


def post_chat(
    message: str,
    *,
    base_url: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Send a message to POST /chat and return the JSON response."""
    url = f"{(base_url or resolve_api_base_url())}/chat"
    try:
        response = httpx.post(url, json={"message": message}, timeout=timeout)
    except httpx.RequestError as exc:
        raise ChatApiError(f"Cannot reach API at {url}: {exc}") from exc

    if response.status_code == 429:
        detail = response.json().get("detail", "Rate limit exceeded.")
        raise ChatApiError(detail, status_code=429)

    if response.status_code >= 400:
        detail = response.json().get("detail", response.text)
        raise ChatApiError(str(detail), status_code=response.status_code)

    return response.json()
