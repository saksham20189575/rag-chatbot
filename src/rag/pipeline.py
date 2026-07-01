"""End-to-end RAG orchestration: classify → retrieve → generate → validate."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.rag.classifier import classify
from src.rag.generator import generate_answer
from src.rag.groq_client import GroqClient, GroqGenerationError, GroqQuotaExceeded
from src.rag.models import ChatResponse, ClassificationResult
from src.rag.refusal import build_performance_response, build_refusal
from src.rag.retriever import Retriever, RetrievalResult
from src.rag.validator import validate_or_refuse

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    groq_calls: int = 0
    retrieval_calls: int = 0


def _handle_non_factual(
    classification: ClassificationResult,
) -> ChatResponse | None:
    scheme_id = classification.scheme_id
    if classification.query_class == "pii":
        return build_refusal("pii", scheme_id=scheme_id)
    if classification.query_class == "advisory":
        return build_refusal("advisory", scheme_id=scheme_id)
    if classification.query_class == "performance":
        return build_performance_response(scheme_id=scheme_id)
    if classification.query_class == "out_of_scope":
        return build_refusal("out_of_scope", scheme_id=scheme_id)
    return None


def answer(
    query: str,
    *,
    retriever: Retriever | None = None,
    groq_client: GroqClient | None = None,
    stats: PipelineStats | None = None,
) -> ChatResponse:
    """Run the full query pipeline and return a compliant chat response."""
    normalized = " ".join(query.split())
    if not normalized:
        raise ValueError("Query must not be empty")

    classification = classify(normalized)
    early = _handle_non_factual(classification)
    if early is not None:
        return early

    active_retriever = retriever or Retriever()
    if stats is not None:
        stats.retrieval_calls += 1
    retrieval = active_retriever.retrieve(
        normalized,
        scheme_id=classification.scheme_id,
    )
    if not retrieval.has_results:
        return build_refusal(
            "insufficient_context",
            scheme_id=classification.scheme_id,
        )

    return _answer_from_retrieval(
        normalized,
        retrieval,
        scheme_id=classification.scheme_id,
        groq_client=groq_client,
        stats=stats,
    )


def _answer_from_retrieval(
    query: str,
    retrieval: RetrievalResult,
    *,
    scheme_id: str | None,
    groq_client: GroqClient | None,
    stats: PipelineStats | None,
) -> ChatResponse:
    client = groq_client or GroqClient()
    try:
        draft = generate_answer(query, retrieval, groq_client=client)
    except GroqQuotaExceeded:
        return build_refusal("quota_exceeded", scheme_id=scheme_id)
    except GroqGenerationError:
        logger.exception("Groq generation failed")
        return build_refusal("groq_error", scheme_id=scheme_id)

    if stats is not None:
        stats.groq_calls += 1
    return validate_or_refuse(draft, scheme_id=scheme_id)
