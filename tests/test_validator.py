"""Tests for src.rag.validator."""

from __future__ import annotations

from src.constants import SCHEME_URLS
from src.rag.models import DraftAnswer
from src.rag.validator import count_sentences, validate_answer, validate_or_refuse


def test_count_sentences():
    assert count_sentences("One. Two. Three.") == 3
    assert count_sentences("Single sentence only.") == 1


def test_validate_passes_three_sentence_allowlisted_answer():
    draft = DraftAnswer(
        text="The expense ratio is 1.04%. Minimum SIP is ₹100. Source facts are from Groww.",
        citation_url=SCHEME_URLS["large_cap"],
        last_updated="2026-06-25",
    )
    result = validate_answer(draft)
    assert result.ok
    assert result.response is not None
    assert result.response.type == "answer"
    assert result.response.disclaimer == "Facts-only. No investment advice."


def test_validate_rejects_four_sentences():
    draft = DraftAnswer(
        text="One. Two. Three. Four.",
        citation_url=SCHEME_URLS["large_cap"],
        last_updated="2026-06-25",
    )
    assert validate_answer(draft).ok is False


def test_validate_rejects_non_allowlisted_citation():
    draft = DraftAnswer(
        text="Expense ratio is 1.04%.",
        citation_url="https://hdfcfund.com/example",
        last_updated="2026-06-25",
    )
    assert validate_answer(draft).ok is False


def test_validate_rejects_advice_language():
    draft = DraftAnswer(
        text="I recommend this fund for long-term investors.",
        citation_url=SCHEME_URLS["mid_cap"],
        last_updated="2026-06-25",
    )
    assert validate_answer(draft).ok is False


def test_validate_rejects_return_percentage():
    draft = DraftAnswer(
        text="The fund delivered 21.62% returns over 3 years.",
        citation_url=SCHEME_URLS["mid_cap"],
        last_updated="2026-06-25",
    )
    assert validate_answer(draft).ok is False


def test_validate_rejects_multiple_urls_in_text():
    draft = DraftAnswer(
        text=(
            "See https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth and "
            "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth."
        ),
        citation_url=SCHEME_URLS["large_cap"],
        last_updated="2026-06-25",
    )
    assert validate_answer(draft).ok is False


def test_validate_or_refuse_returns_refusal_without_regenerating():
    draft = DraftAnswer(
        text="I recommend investing now.",
        citation_url=SCHEME_URLS["large_cap"],
        last_updated="2026-06-25",
    )
    response = validate_or_refuse(draft, scheme_id="large_cap")
    assert response.type == "refusal"
