"""Sanitize and validate chat messages before the RAG pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape

from src.rag.classifier import contains_pii

MAX_MESSAGE_LENGTH = 500

_SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style)[^>]*>.*?</\1>")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class MessageValidationError:
    code: str
    detail: str


def strip_html(text: str) -> str:
    """Remove script/style blocks and HTML tags from user input."""
    cleaned = _SCRIPT_STYLE_RE.sub("", text)
    cleaned = _HTML_TAG_RE.sub(" ", cleaned)
    return unescape(cleaned)


def normalize_message(text: str) -> str:
    """Strip HTML and collapse whitespace."""
    return " ".join(strip_html(text).split())


def validate_message(text: str) -> MessageValidationError | None:
    """Return a validation error, or None when the message is acceptable."""
    if not text:
        return MessageValidationError(code="empty_message", detail="Message must not be empty.")
    if len(text) > MAX_MESSAGE_LENGTH:
        return MessageValidationError(
            code="message_too_long",
            detail=f"Message must be at most {MAX_MESSAGE_LENGTH} characters.",
        )
    return None
