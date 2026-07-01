"""FastAPI chat API — Phase 4.

Run locally::

    uvicorn src.api.main:app --reload

Example requests::

    curl -s http://localhost:8000/health | jq

    curl -s -X POST http://localhost:8000/chat \\
      -H 'Content-Type: application/json' \\
      -d '{"message":"What is the min SIP for HDFC Small Cap Fund?"}' | jq
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.api.input_utils import MAX_MESSAGE_LENGTH, normalize_message, validate_message
from src.api.rate_limit import ApiRateLimiter, RateLimitExceeded
from src.constants import ALLOWLISTED_GROWW_URLS, CORPUS_CONFIG_PATH, PROJECT_ROOT
from src.ingestion.indexer import get_index_stats, read_index_manifest
from src.ingestion.refresh import read_refresh_manifest
from src.rag.classifier import contains_pii
from src.rag.groq_client import GroqClient
from src.rag.models import ChatResponse
from src.rag.pipeline import answer
from src.rag.refusal import build_refusal
from src.rag.retriever import Retriever

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=MAX_MESSAGE_LENGTH + 100)


class ChatResponseModel(BaseModel):
    type: str
    text: str
    citation_url: str
    last_updated: str
    disclaimer: str


class ErrorResponse(BaseModel):
    detail: str


def load_allowlisted_urls() -> frozenset[str]:
    """Load Groww URLs from corpus config and verify against shared constants."""
    with open(CORPUS_CONFIG_PATH, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    urls = frozenset(scheme["url"] for scheme in payload["schemes"])
    unknown = urls - ALLOWLISTED_GROWW_URLS
    if unknown:
        raise RuntimeError(f"Corpus config contains non-allowlisted URLs: {sorted(unknown)}")
    return urls


def _chat_response_model(response: ChatResponse) -> ChatResponseModel:
    return ChatResponseModel(**response.to_dict())


def _client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client is not None:
        return request.client.host
    return "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is required. Copy .env.example to .env and set your Groq API key."
        )

    try:
        retriever = Retriever()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Vector index not found. Run: python scripts/build_index.py"
        ) from exc

    app.state.retriever = retriever
    app.state.groq_client = GroqClient(api_key=api_key)
    app.state.allowlisted_urls = load_allowlisted_urls()
    app.state.rate_limiter = ApiRateLimiter()
    app.state.index_ready = True
    manifest = read_index_manifest()
    logger.info(
        "API ready — index loaded (%s chunks)",
        manifest.get("chunk_count", "?") if manifest else "?",
    )
    yield


def create_app(*, lifespan_override: Any | None = None) -> FastAPI:
    app = FastAPI(
        title="Mutual Fund FAQ Assistant",
        description="Facts-only RAG chat API for five HDFC schemes on Groww.",
        version="0.4.0",
        lifespan=lifespan_override or lifespan,
    )
    _register_routes(app)
    _register_exception_handlers(app)
    return app


def _register_routes(app: FastAPI) -> None:
    @app.get("/health")
    def health(request: Request) -> JSONResponse:
        if not getattr(request.app.state, "index_ready", False):
            return JSONResponse(
                status_code=503,
                content={"status": "unavailable", "detail": "Vector index is not loaded."},
            )

        try:
            index_stats = get_index_stats()
        except Exception:
            logger.exception("Failed to read index stats")
            return JSONResponse(
                status_code=503,
                content={"status": "unavailable", "detail": "Vector index is not available."},
            )

        groq_client: GroqClient = request.app.state.groq_client
        refresh_manifest = read_refresh_manifest()
        payload = {
            "status": "ok",
            "index": {
                "chunk_count": index_stats.get("total_chunks", 0),
                "embedded_at": index_stats.get("embedded_at"),
                "embedding_model": index_stats.get("embedding_model"),
                "collection_name": index_stats.get("collection_name"),
            },
            "last_refreshed_at": (
                refresh_manifest.get("last_success_at") if refresh_manifest else None
            ),
            "refresh_status": refresh_manifest.get("status") if refresh_manifest else None,
            "groq_quota": groq_client.get_quota_status(),
        }
        return JSONResponse(status_code=200, content=payload)

    @app.post(
        "/chat",
        response_model=ChatResponseModel,
        responses={
            400: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    def chat(body: ChatRequest, request: Request) -> ChatResponseModel:
        if not getattr(request.app.state, "index_ready", False):
            raise HTTPException(status_code=503, detail="Service temporarily unavailable.")

        rate_limiter: ApiRateLimiter = request.app.state.rate_limiter
        client_key = _client_key(request)
        try:
            rate_limiter.check(client_key)
        except RateLimitExceeded as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc

        message = normalize_message(body.message)
        validation_error = validate_message(message)
        if validation_error is not None:
            raise HTTPException(status_code=400, detail=validation_error.detail)

        if contains_pii(message):
            return _chat_response_model(build_refusal("pii"))

        retriever: Retriever = request.app.state.retriever
        groq_client: GroqClient = request.app.state.groq_client
        try:
            response = answer(message, retriever=retriever, groq_client=groq_client)
        except Exception:
            logger.exception("Unhandled error while answering query")
            raise HTTPException(
                status_code=500,
                detail="An internal error occurred while processing your request.",
            ) from None

        return _chat_response_model(response)


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


app = create_app()
