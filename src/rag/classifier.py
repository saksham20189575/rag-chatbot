"""Rule-based query routing before retrieval or Groq generation."""

from __future__ import annotations

import re

from src.rag.models import ClassificationResult
from src.rag.query_utils import detect_scheme_id

# PII patterns (conservative — false positives refuse rather than leak).
_PAN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
_AADHAAR = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b")
_OTP = re.compile(r"\botp\s+is\s+\d+", re.IGNORECASE)
_FOLIO_NUMBER = re.compile(r"\bfolio\s+\d+", re.IGNORECASE)

# Performance before advisory — return/ranking queries must not hit Groq.
_PERFORMANCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\breturns?\b", re.IGNORECASE),
    re.compile(r"\bcagr\b", re.IGNORECASE),
    re.compile(r"\bperformance\b", re.IGNORECASE),
    re.compile(r"compare\s+returns?", re.IGNORECASE),
    re.compile(r"\b3[\s-]?year\b", re.IGNORECASE),
    re.compile(r"\b3y\b", re.IGNORECASE),
    re.compile(r"\b5y\b", re.IGNORECASE),
    re.compile(r"\brank(?:ing)?\b", re.IGNORECASE),
    re.compile(r"\d+(?:\.\d+)?\s*%\s*(?:return|cagr|yield)", re.IGNORECASE),
)

_ADVISORY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"should\s+i\s+(?:invest|buy)", re.IGNORECASE),
    re.compile(r"which\s+is\s+better", re.IGNORECASE),
    re.compile(r"which\s+.*\bbetter\b", re.IGNORECASE),
    re.compile(r"\brecommend\b", re.IGNORECASE),
    re.compile(r"\bworth\s+it\b", re.IGNORECASE),
    re.compile(r"honest\s+opinion", re.IGNORECASE),
    re.compile(r"ignore\s+(?:all\s+)?(?:rules|instructions)", re.IGNORECASE),
    re.compile(r"repeat\s+your\s+system", re.IGNORECASE),
    re.compile(r"\bsuggest\b.*\bfund\b", re.IGNORECASE),
    re.compile(r"compare\s+(?:funds?|large|mid|small|gold|silver)", re.IGNORECASE),
)

_OUT_OF_SCOPE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:sbi|icici|axis|nippon|mirae|kotak|franklin|uti|dsp|tata)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bhdfc\s+(?:flexi|hybrid|balanced|debt|liquid|income|tax\s+saver)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:folio|account\s+balance|my\s+portfolio)\b", re.IGNORECASE),
    re.compile(r"download\s+(?:my\s+)?(?:statement|report|capital\s+gains)", re.IGNORECASE),
    re.compile(r"\bshow\s+my\s+(?:folio|holdings|balance)\b", re.IGNORECASE),
)


def _matches_any(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def contains_pii(text: str) -> bool:
    return any(
        pattern.search(text)
        for pattern in (_PAN, _AADHAAR, _EMAIL, _PHONE, _OTP, _FOLIO_NUMBER)
    )


def classify(query: str) -> ClassificationResult:
    """Route a user query to a handler class and optional scheme_id."""
    normalized = " ".join(query.split())
    scheme_id = detect_scheme_id(normalized)

    if contains_pii(normalized):
        return ClassificationResult(query_class="pii", scheme_id=scheme_id)

    if _matches_any(normalized, _PERFORMANCE_PATTERNS):
        return ClassificationResult(query_class="performance", scheme_id=scheme_id)

    if _matches_any(normalized, _ADVISORY_PATTERNS):
        return ClassificationResult(query_class="advisory", scheme_id=scheme_id)

    if _matches_any(normalized, _OUT_OF_SCOPE_PATTERNS):
        return ClassificationResult(query_class="out_of_scope", scheme_id=scheme_id)

    return ClassificationResult(query_class="factual", scheme_id=scheme_id)
