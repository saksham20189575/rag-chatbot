"""Compliance checks on generated answers before returning to callers."""

from __future__ import annotations

import re

from src.constants import ALLOWLISTED_GROWW_URLS, DISCLAIMER
from src.rag.classifier import contains_pii
from src.rag.models import ChatResponse, DraftAnswer, ValidationResult
from src.rag.refusal import build_refusal

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_URL_PATTERN = re.compile(r"https?://[^\s\])>]+", re.IGNORECASE)
_ADVICE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\byou should\b", re.IGNORECASE),
    re.compile(r"\bi recommend\b", re.IGNORECASE),
    re.compile(r"\bwe recommend\b", re.IGNORECASE),
    re.compile(r"\bbetter fund\b", re.IGNORECASE),
    re.compile(r"\bshould invest\b", re.IGNORECASE),
)
_RETURN_PERCENT = re.compile(
    r"(?:return|cagr|yield|performance|annual(?:ized)?)[\s\w,]{0,25}\d+(?:\.\d+)?\s*%"
    r"|\d+(?:\.\d+)?\s*%\s*(?:return|cagr|yield|annual(?:ized)?|p\.?a\.?)",
    re.IGNORECASE,
)


def count_sentences(text: str) -> int:
    cleaned = " ".join(text.split())
    if not cleaned:
        return 0
    parts = [part for part in _SENTENCE_SPLIT.split(cleaned) if part.strip()]
    return len(parts)


def _normalize_url(url: str) -> str:
    return url.rstrip("/")


def validate_answer(draft: DraftAnswer) -> ValidationResult:
    """Validate a draft answer; return safe refusal on any compliance failure."""
    if count_sentences(draft.text) > 3:
        return ValidationResult(ok=False, reason="too_many_sentences")

    citation = _normalize_url(draft.citation_url)
    if citation not in {_normalize_url(url) for url in ALLOWLISTED_GROWW_URLS}:
        return ValidationResult(ok=False, reason="citation_not_allowlisted")

    if _matches_advice(draft.text):
        return ValidationResult(ok=False, reason="advice_language")

    if _RETURN_PERCENT.search(draft.text):
        return ValidationResult(ok=False, reason="return_percentage")

    if contains_pii(draft.text):
        return ValidationResult(ok=False, reason="pii_echo")

    urls_in_text = [_normalize_url(match) for match in _URL_PATTERN.findall(draft.text)]
    if len(urls_in_text) > 1:
        return ValidationResult(ok=False, reason="multiple_urls")

    for url in urls_in_text:
        if url not in {_normalize_url(allowed) for allowed in ALLOWLISTED_GROWW_URLS}:
            return ValidationResult(ok=False, reason="non_allowlisted_url_in_text")

    response = ChatResponse(
        type="answer",
        text=draft.text.strip(),
        citation_url=citation,
        last_updated=draft.last_updated,
        disclaimer=DISCLAIMER,
    )
    return ValidationResult(ok=True, response=response)


def validate_or_refuse(
    draft: DraftAnswer,
    *,
    scheme_id: str | None = None,
) -> ChatResponse:
    """Validate draft answer or return a safe refusal without re-calling Groq."""
    result = validate_answer(draft)
    if result.ok and result.response is not None:
        return result.response
    return build_refusal("validation_failed", scheme_id=scheme_id)


def _matches_advice(text: str) -> bool:
    return any(pattern.search(text) for pattern in _ADVICE_PATTERNS)
