"""Groq-backed answer synthesis from retrieved context."""

from __future__ import annotations

from src.rag.groq_client import GroqClient, GroqGenerationError, GroqQuotaExceeded
from src.rag.models import DraftAnswer
from src.rag.retriever import RetrievalResult

SYSTEM_PROMPT = """You are a facts-only mutual fund FAQ assistant for five HDFC schemes on Groww.
Rules:
- Answer in at most 3 sentences.
- Use ONLY the provided context from Groww fund pages.
- Do not give investment advice, opinions, or fund comparisons.
- Do not state historical returns or performance percentages.
- Do not include URLs in your answer text."""


def build_user_prompt(query: str, context: str) -> str:
    return f"Context:\n{context}\n\nUser question: {query}"


def generate_answer(
    query: str,
    retrieval: RetrievalResult,
    *,
    groq_client: GroqClient | None = None,
) -> DraftAnswer:
    """Call Groq once to synthesize an answer from retrieved chunks."""
    if not retrieval.chunks:
        raise ValueError("Cannot generate without retrieved chunks")

    primary = retrieval.chunks[0]
    client = groq_client or GroqClient()
    result = client.complete(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_user_prompt(query, retrieval.context),
    )
    if not result.text:
        raise GroqGenerationError("Groq returned an empty response")

    return DraftAnswer(
        text=result.text,
        citation_url=primary.source_url,
        last_updated=primary.last_updated,
    )


__all__ = [
    "SYSTEM_PROMPT",
    "GroqGenerationError",
    "GroqQuotaExceeded",
    "build_user_prompt",
    "generate_answer",
]
