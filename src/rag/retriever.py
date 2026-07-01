"""Scheme-aware dense retrieval over the local Chroma index."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from src.constants import INDEX_DIR
from src.ingestion.indexer import (
    embed_query_text,
    load_collection,
    load_embedding_model,
    resolve_bge_model_name,
    verify_embedding_model,
)
from src.rag.query_utils import normalize_query

logger = logging.getLogger(__name__)

RETRIEVAL_TOP_K = 3
RETRIEVAL_CONTEXT_CHUNKS = 2
MIN_SIMILARITY = 0.25
KEYWORD_BOOST = 0.12

# Query signals → section_title for within-scheme keyword reranking.
SECTION_QUERY_SIGNALS: dict[str, str] = {
    "expense ratio": "Key fund facts",
    "ter": "Key fund facts",
    "nav": "Key fund facts",
    "aum": "Key fund facts",
    "fund size": "Key fund facts",
    "riskometer": "Key fund facts",
    "risk": "Key fund facts",
    "exit load": "Exit load",
    "redemption charge": "Exit load",
    "benchmark": "Benchmark",
    "index": "Benchmark",
    "min sip": "Minimum investments",
    "minimum sip": "Minimum investments",
    "minimum investment": "Minimum investments",
    "minimum investments": "Minimum investments",
    "lumpsum": "Minimum investments",
    "stamp duty": "Stamp duty and tax",
    "tax": "Stamp duty and tax",
    "stt": "Stamp duty and tax",
    "investment objective": "Investment objective",
    "objective": "Investment objective",
    "fund manager": "Fund management",
    "fund management": "Fund management",
    "manager": "Fund management",
    "tenure": "Fund management",
}


@dataclass
class RetrievedChunk:
    chunk_id: str
    scheme_id: str
    scheme_name: str
    section_title: str
    text: str
    source_url: str
    last_updated: str
    score: float

    @classmethod
    def from_chroma(
        cls,
        *,
        chunk_id: str,
        document: str,
        metadata: dict[str, Any],
        score: float,
    ) -> RetrievedChunk:
        return cls(
            chunk_id=chunk_id,
            scheme_id=metadata["scheme_id"],
            scheme_name=metadata["scheme_name"],
            section_title=metadata["section_title"],
            text=document,
            source_url=metadata["source_url"],
            last_updated=metadata["last_updated"],
            score=score,
        )


@dataclass
class RetrievalResult:
    query: str
    expanded_query: str
    scheme_id: str | None
    chunks: list[RetrievedChunk]
    context: str

    @property
    def has_results(self) -> bool:
        return bool(self.chunks)


def _distance_to_similarity(distance: float) -> float:
    """Convert Chroma cosine distance to similarity (vectors are L2-normalized)."""
    return 1.0 - distance


def _target_sections(query: str) -> set[str]:
    lowered = query.lower()
    targets: set[str] = set()
    for signal, section_title in sorted(
        SECTION_QUERY_SIGNALS.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if signal in lowered:
            targets.add(section_title)
    return targets


def _keyword_boost(section_title: str, target_sections: set[str]) -> float:
    if not target_sections:
        return 0.0
    if section_title.casefold() in {title.casefold() for title in target_sections}:
        return KEYWORD_BOOST
    return 0.0


def assemble_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks for the LLM prompt."""
    blocks: list[str] = []
    for chunk in chunks:
        blocks.append(
            f"[{chunk.source_url} | {chunk.last_updated} | {chunk.section_title}]\n{chunk.text}"
        )
    return "\n\n---\n\n".join(blocks)


class Retriever:
    """Load the vector index and retrieve scheme-filtered chunks for a query."""

    def __init__(
        self,
        *,
        index_dir: Path | None = None,
        model: SentenceTransformer | None = None,
        model_name: str | None = None,
        collection: Collection | None = None,
    ) -> None:
        self.index_dir = index_dir or INDEX_DIR
        self.model_name = model_name or resolve_bge_model_name()
        self.collection = collection or load_collection(self.index_dir)
        verify_embedding_model(
            index_dir=self.index_dir,
            model_name=self.model_name,
            collection=self.collection,
        )
        self.model = model or load_embedding_model(self.model_name)

    def retrieve(
        self,
        query: str,
        *,
        scheme_id: str | None = None,
        top_k: int = RETRIEVAL_TOP_K,
        context_chunks: int = RETRIEVAL_CONTEXT_CHUNKS,
        min_similarity: float = MIN_SIMILARITY,
    ) -> RetrievalResult:
        expanded, detected_scheme = normalize_query(query)
        resolved_scheme = scheme_id or detected_scheme
        if resolved_scheme is None:
            logger.info("No scheme detected for query — returning empty retrieval")
            return RetrievalResult(
                query=query,
                expanded_query=expanded,
                scheme_id=None,
                chunks=[],
                context="",
            )

        query_vector = embed_query_text(self.model, expanded)
        raw = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, 7),
            where={"scheme_id": resolved_scheme},
            include=["documents", "metadatas", "distances"],
        )

        target_sections = _target_sections(expanded)
        ranked: list[RetrievedChunk] = []
        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            similarity = _distance_to_similarity(distance)
            boost = _keyword_boost(metadata["section_title"], target_sections)
            score = similarity + boost
            if score < min_similarity:
                continue
            ranked.append(
                RetrievedChunk.from_chroma(
                    chunk_id=chunk_id,
                    document=document,
                    metadata=metadata,
                    score=score,
                )
            )

        ranked.sort(key=lambda chunk: chunk.score, reverse=True)
        selected = ranked[:context_chunks]
        return RetrievalResult(
            query=query,
            expanded_query=expanded,
            scheme_id=resolved_scheme,
            chunks=selected,
            context=assemble_context(selected),
        )


def retrieve(
    query: str,
    *,
    index_dir: Path | None = None,
    scheme_id: str | None = None,
    retriever: Retriever | None = None,
) -> RetrievalResult:
    """Convenience wrapper that reuses a provided retriever or builds a default one."""
    active = retriever or Retriever(index_dir=index_dir)
    return active.retrieve(query, scheme_id=scheme_id)
