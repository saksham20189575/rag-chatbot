"""Embed section-atomic chunks with BGE and persist to a local Chroma index.

Embedding strategy (ImplementationPlan §1.4):
- Index ``embed_text`` (scheme + section prefix); store raw ``text`` as the document.
- Documents are embedded without the BGE query instruction prefix.
- Cosine similarity over L2-normalized BGE vectors.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from src.constants import (
    BGE_QUERY_PREFIX,
    CHROMA_COLLECTION_NAME,
    CHUNKS_DATA_DIR,
    DEFAULT_BGE_MODEL,
    INDEX_DIR,
)
from src.ingestion.chunker import CHUNK_MANIFEST_FILENAME, Chunk

logger = logging.getLogger(__name__)

INDEX_MANIFEST_FILENAME = "index_manifest.json"


def chunks_fingerprint(chunks: list[Chunk]) -> str:
    """Stable hash of chunk content for skip-re-embed detection (IX-05)."""
    payload = "|".join(sorted(f"{c.chunk_id}:{c.embed_text}" for c in chunks))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class IndexResult:
    chunk_count: int
    embedding_model: str
    embedded_at: str
    index_dir: Path
    collection_name: str
    scheme_ids: list[str]


def resolve_bge_model_name() -> str:
    """Return ``BGE_MODEL_NAME`` from the environment or the project default."""
    return os.getenv("BGE_MODEL_NAME", DEFAULT_BGE_MODEL).strip() or DEFAULT_BGE_MODEL


def load_embedding_model(model_name: str | None = None) -> SentenceTransformer:
    """Load the BGE sentence-transformers model."""
    name = model_name or resolve_bge_model_name()
    logger.info("Loading embedding model: %s", name)
    return SentenceTransformer(name)


def embed_document_texts(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """Embed corpus passages (no BGE query prefix)."""
    if not texts:
        return []
    vectors = model.encode(
        texts,
        batch_size=len(texts),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def embed_query_text(model: SentenceTransformer, query: str) -> list[float]:
    """Embed a user query with the BGE v1.5 retrieval instruction prefix."""
    prefixed = f"{BGE_QUERY_PREFIX}{query.strip()}"
    vector = model.encode(
        [prefixed],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vector[0].tolist()


def load_saved_chunks(
    *,
    chunks_dir: Path | None = None,
    scheme_id: str | None = None,
) -> list[Chunk]:
    """Load chunk review files written by ``save_chunks``."""
    root = chunks_dir or CHUNKS_DATA_DIR
    if not root.is_dir():
        raise FileNotFoundError(f"Chunks directory not found: {root}")

    paths = sorted(root.glob("*.json"))
    paths = [p for p in paths if p.name != CHUNK_MANIFEST_FILENAME]
    if scheme_id is not None:
        path = root / f"{scheme_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"No chunk file for scheme {scheme_id!r}: {path}")
        paths = [path]

    chunks: list[Chunk] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("chunks", []):
            chunks.append(
                Chunk(
                    chunk_id=item["chunk_id"],
                    scheme_id=item["scheme_id"],
                    scheme_name=item["scheme_name"],
                    document_type=item["document_type"],
                    source_url=item["source_url"],
                    source_domain=item["source_domain"],
                    section_title=item["section_title"],
                    last_updated=item["last_updated"],
                    text=item["text"],
                    embed_text=item["embed_text"],
                    token_estimate=item["token_estimate"],
                )
            )

    if not chunks:
        raise ValueError(f"No chunks found under {root}")
    return chunks


def read_index_manifest(index_dir: Path | None = None) -> dict[str, Any] | None:
    """Return persisted index metadata, or ``None`` if the index was never built."""
    path = (index_dir or INDEX_DIR) / INDEX_MANIFEST_FILENAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def verify_embedding_model(
    *,
    index_dir: Path | None = None,
    model_name: str | None = None,
    collection: Collection | None = None,
) -> str:
    """Ensure the configured BGE model matches the persisted index (RT-10).

    Returns the resolved model name on success.
    """
    expected = model_name or resolve_bge_model_name()
    root = index_dir or INDEX_DIR
    manifest = read_index_manifest(root)
    if manifest is None:
        raise FileNotFoundError(
            f"Vector index not found under {root}. Run scripts/build_index.py first."
        )

    manifest_model = manifest.get("embedding_model")
    if manifest_model and manifest_model != expected:
        raise ValueError(
            f"Index was built with {manifest_model!r} but {expected!r} is configured. "
            "Run a full rebuild or set BGE_MODEL_NAME to match."
        )

    if collection is not None:
        collection_model = (collection.metadata or {}).get("embedding_model")
        if collection_model and collection_model != expected:
            raise ValueError(
                f"Chroma collection metadata has {collection_model!r} but {expected!r} is configured."
            )

    return expected


def write_index_manifest(
    *,
    chunk_count: int,
    embedding_model: str,
    embedded_at: str,
    scheme_ids: list[str],
    index_dir: Path | None = None,
    chunks_fingerprint: str | None = None,
) -> Path:
    """Write index metadata for RT-10 embedding-model validation."""
    out_dir = index_dir or INDEX_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "embedding_model": embedding_model,
        "embedded_at": embedded_at,
        "chunk_count": chunk_count,
        "collection_name": CHROMA_COLLECTION_NAME,
        "scheme_ids": sorted(scheme_ids),
    }
    if chunks_fingerprint is not None:
        payload["chunks_fingerprint"] = chunks_fingerprint
    path = out_dir / INDEX_MANIFEST_FILENAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def _chunk_to_chroma_record(chunk: Chunk) -> tuple[str, str, dict[str, Any]]:
    """Map a chunk to Chroma id, document text, and metadata."""
    metadata = chunk.metadata()
    metadata["text"] = chunk.text
    return chunk.chunk_id, chunk.text, metadata


def _get_chroma_client(index_dir: Path | None = None) -> chromadb.PersistentClient:
    path = index_dir or INDEX_DIR
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def _get_or_create_collection(
    client: chromadb.PersistentClient,
    *,
    embedding_model: str,
) -> Collection:
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": embedding_model,
        },
    )


def load_collection(index_dir: Path | None = None) -> Collection:
    """Open the persisted Chroma collection or raise if the index is missing."""
    root = index_dir or INDEX_DIR
    manifest = read_index_manifest(root)
    if manifest is None:
        raise FileNotFoundError(
            f"Vector index not found under {root}. Run scripts/build_index.py first."
        )
    client = _get_chroma_client(root)
    try:
        return client.get_collection(CHROMA_COLLECTION_NAME)
    except (ValueError, chromadb.errors.NotFoundError) as exc:
        raise FileNotFoundError(
            f"Chroma collection {CHROMA_COLLECTION_NAME!r} not found under {root}."
        ) from exc


def get_index_stats(index_dir: Path | None = None) -> dict[str, Any]:
    """Return chunk counts, model name, and vector dimensions from the live index."""
    root = index_dir or INDEX_DIR
    manifest = read_index_manifest(root)
    try:
        collection = load_collection(root)
    except FileNotFoundError:
        return {
            "total_chunks": 0,
            "chunks_per_scheme": {},
            "embedding_model": manifest.get("embedding_model") if manifest else None,
            "embedded_at": manifest.get("embedded_at") if manifest else None,
            "collection_name": CHROMA_COLLECTION_NAME,
            "embedding_dimensions": None,
        }

    stored = collection.get(include=["metadatas", "embeddings"])
    per_scheme: dict[str, int] = {}
    for meta in stored["metadatas"]:
        per_scheme[meta["scheme_id"]] = per_scheme.get(meta["scheme_id"], 0) + 1

    dimensions = None
    embeddings = stored.get("embeddings")
    if embeddings is not None and len(embeddings) > 0:
        dimensions = len(embeddings[0])

    return {
        "total_chunks": collection.count(),
        "chunks_per_scheme": dict(sorted(per_scheme.items())),
        "embedding_model": (collection.metadata or {}).get("embedding_model")
        or (manifest.get("embedding_model") if manifest else None),
        "embedded_at": manifest.get("embedded_at") if manifest else None,
        "collection_name": CHROMA_COLLECTION_NAME,
        "embedding_dimensions": dimensions,
    }


def index_chunks(
    chunks: list[Chunk],
    *,
    model: SentenceTransformer | None = None,
    model_name: str | None = None,
    index_dir: Path | None = None,
    scheme_id: str | None = None,
    content_fingerprint: str | None = None,
) -> IndexResult:
    """Embed chunks and write them to the Chroma collection.

    When ``scheme_id`` is set, only that scheme's vectors are replaced; other
    schemes already in the index are preserved. A full rebuild (no ``scheme_id``)
    recreates the entire collection.
    """
    if not chunks:
        raise ValueError("Cannot index an empty chunk list")

    resolved_model_name = model_name or resolve_bge_model_name()
    embedder = model or load_embedding_model(resolved_model_name)
    client = _get_chroma_client(index_dir)
    embedded_at = datetime.now(UTC).replace(microsecond=0).isoformat()

    if scheme_id is None:
        try:
            client.delete_collection(CHROMA_COLLECTION_NAME)
            logger.info("Deleted existing collection %s for full rebuild", CHROMA_COLLECTION_NAME)
        except (ValueError, chromadb.errors.NotFoundError):
            pass
        collection = _get_or_create_collection(client, embedding_model=resolved_model_name)
        target_chunks = chunks
    else:
        collection = _get_or_create_collection(client, embedding_model=resolved_model_name)
        existing_model = (collection.metadata or {}).get("embedding_model")
        if existing_model and existing_model != resolved_model_name:
            raise ValueError(
                f"Index was built with {existing_model!r} but {resolved_model_name!r} is configured. "
                "Run a full rebuild without --scheme or set BGE_MODEL_NAME to match."
            )
        collection.delete(where={"scheme_id": scheme_id})
        logger.info("Removed existing vectors for scheme %s", scheme_id)
        target_chunks = [c for c in chunks if c.scheme_id == scheme_id]
        if not target_chunks:
            raise ValueError(f"No chunks for scheme {scheme_id!r} in provided chunk list")

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    embed_inputs: list[str] = []

    for chunk in target_chunks:
        chunk_id, document, metadata = _chunk_to_chroma_record(chunk)
        ids.append(chunk_id)
        documents.append(document)
        metadatas.append(metadata)
        embed_inputs.append(chunk.embed_text)

    embeddings = embed_document_texts(embedder, embed_inputs)
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    stored = collection.get(include=["metadatas"])
    scheme_ids = sorted({meta["scheme_id"] for meta in stored["metadatas"]})
    indexed_count = collection.count()

    write_index_manifest(
        chunk_count=indexed_count,
        embedding_model=resolved_model_name,
        embedded_at=embedded_at,
        scheme_ids=scheme_ids,
        index_dir=index_dir,
        chunks_fingerprint=content_fingerprint,
    )

    logger.info(
        "Indexed %s chunk(s) with %s into %s",
        len(target_chunks),
        resolved_model_name,
        CHROMA_COLLECTION_NAME,
    )

    return IndexResult(
        chunk_count=indexed_count,
        embedding_model=resolved_model_name,
        embedded_at=embedded_at,
        index_dir=index_dir or INDEX_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
        scheme_ids=scheme_ids,
    )


def build_index(
    chunks: list[Chunk] | None = None,
    *,
    chunks_dir: Path | None = None,
    scheme_id: str | None = None,
    model_name: str | None = None,
    index_dir: Path | None = None,
    content_fingerprint: str | None = None,
) -> IndexResult:
    """Load chunks (from memory or disk), embed, and persist the vector index."""
    if chunks is None:
        chunks = load_saved_chunks(chunks_dir=chunks_dir, scheme_id=scheme_id)
    if content_fingerprint is None and scheme_id is None:
        content_fingerprint = chunks_fingerprint(chunks)
    return index_chunks(
        chunks,
        model_name=model_name,
        index_dir=index_dir,
        scheme_id=scheme_id,
        content_fingerprint=content_fingerprint,
    )
