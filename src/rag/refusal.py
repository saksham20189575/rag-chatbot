"""Fixed refusal and link-only responses — no Groq generation."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from src.constants import (
    DEFAULT_CITATION_URL,
    DEFAULT_SCHEME_ID,
    DISCLAIMER,
    SCHEME_URLS,
)
from src.ingestion.indexer import read_index_manifest
from src.rag.models import ChatResponse

RefusalReason = Literal[
    "advisory",
    "pii",
    "performance",
    "out_of_scope",
    "insufficient_context",
    "validation_failed",
    "quota_exceeded",
    "groq_error",
]


def _resolve_citation_url(scheme_id: str | None) -> str:
    if scheme_id and scheme_id in SCHEME_URLS:
        return SCHEME_URLS[scheme_id]
    return DEFAULT_CITATION_URL


def _default_last_updated() -> str:
    manifest = read_index_manifest()
    if manifest and manifest.get("embedded_at"):
        return manifest["embedded_at"][:10]
    return datetime.now(UTC).date().isoformat()


def build_refusal(
    reason: RefusalReason,
    *,
    scheme_id: str | None = None,
    last_updated: str | None = None,
) -> ChatResponse:
    """Return a template refusal or link-only response for the given reason."""
    citation_url = _resolve_citation_url(scheme_id)
    updated = last_updated or _default_last_updated()

    templates: dict[RefusalReason, str] = {
        "advisory": (
            "I can only answer factual questions about the five listed HDFC schemes on Groww. "
            "I cannot provide investment advice, opinions, or fund comparisons."
        ),
        "pii": (
            "Please do not share personal information such as PAN, Aadhaar, phone numbers, or folio details. "
            "I can only answer general factual questions about the five listed HDFC schemes."
        ),
        "performance": (
            "I cannot state or discuss fund returns or performance figures here. "
            "For historical returns and performance data, please refer to the official Groww fund page."
        ),
        "out_of_scope": (
            "That question is outside the scope of this assistant. "
            "I can only answer factual questions about these five HDFC schemes: "
            "Large Cap, Mid Cap, Small Cap, Gold ETF FoF, and Silver ETF FoF."
        ),
        "insufficient_context": (
            "I could not find enough information to answer that question reliably. "
            "Please try rephrasing with a specific scheme name and fact "
            "(for example, expense ratio, exit load, or minimum SIP)."
        ),
        "validation_failed": (
            "I could not produce a compliant answer from the available sources. "
            "Please refer to the official Groww fund page for verified facts."
        ),
        "quota_exceeded": (
            "The assistant is temporarily unable to generate new answers due to usage limits. "
            "Please try again shortly or refer to the official Groww fund page."
        ),
        "groq_error": (
            "The assistant is temporarily unavailable. "
            "Please try again later or refer to the official Groww fund page for fund facts."
        ),
    }

    return ChatResponse(
        type="refusal",
        text=templates[reason],
        citation_url=citation_url,
        last_updated=updated,
        disclaimer=DISCLAIMER,
    )


def build_performance_response(
    *,
    scheme_id: str | None = None,
    last_updated: str | None = None,
) -> ChatResponse:
    """Link-only path for performance queries — no return figures."""
    return build_refusal(
        "performance",
        scheme_id=scheme_id or DEFAULT_SCHEME_ID,
        last_updated=last_updated,
    )
