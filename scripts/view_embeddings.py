#!/usr/bin/env python3
"""Inspect chunk embeddings stored in the local Chroma index."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import CHUNKS_DATA_DIR, INDEX_DIR, SCHEME_IDS  # noqa: E402
from src.ingestion.chunker import CHUNK_MANIFEST_FILENAME  # noqa: E402
from src.ingestion.indexer import (  # noqa: E402
    embed_query_text,
    get_index_stats,
    load_collection,
    load_embedding_model,
    read_index_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View embeddings for indexed corpus chunks.",
    )
    parser.add_argument(
        "--scheme",
        metavar="SCHEME_ID",
        help="Filter by scheme (e.g. large_cap, gold_fof).",
    )
    parser.add_argument(
        "--chunk-id",
        metavar="ID",
        help="Show one chunk by id (e.g. large_cap_groww_0001).",
    )
    parser.add_argument(
        "--query",
        metavar="TEXT",
        help="Embed a query and rank chunks by cosine similarity.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=8,
        metavar="N",
        help="Number of embedding dimensions to preview (default: 8).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print the full embedding vector.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=INDEX_DIR,
        help=f"Path to Chroma index (default: {INDEX_DIR}).",
    )
    return parser.parse_args()


def _as_floats(vector: list[float] | Any) -> list[float]:
    return [float(v) for v in vector]


def _l2_norm(vector: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vector))


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _load_embed_text_map(chunks_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in sorted(chunks_dir.glob("*.json")):
        if path.name == CHUNK_MANIFEST_FILENAME:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("chunks", []):
            mapping[item["chunk_id"]] = item.get("embed_text", "")
    return mapping


def _fetch_records(
    *,
    index_dir: Path,
    scheme_id: str | None,
    chunk_id: str | None,
) -> dict[str, Any]:
    collection = load_collection(index_dir)
    if chunk_id is not None:
        stored = collection.get(
            ids=[chunk_id],
            include=["embeddings", "documents", "metadatas"],
        )
    elif scheme_id is not None:
        stored = collection.get(
            where={"scheme_id": scheme_id},
            include=["embeddings", "documents", "metadatas"],
        )
    else:
        stored = collection.get(include=["embeddings", "documents", "metadatas"])

    ids = stored["ids"]
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    return {
        "ids": [ids[i] for i in order],
        "embeddings": [stored["embeddings"][i] for i in order],
        "documents": [stored["documents"][i] for i in order],
        "metadatas": [stored["metadatas"][i] for i in order],
    }


def _build_record(
    *,
    chunk_id: str,
    embedding: list[float],
    document: str,
    metadata: dict[str, Any],
    embed_text: str,
    preview_dims: int,
    show_full: bool,
) -> dict[str, Any]:
    preview_count = len(embedding) if show_full else min(preview_dims, len(embedding))
    return {
        "chunk_id": chunk_id,
        "scheme_id": metadata.get("scheme_id"),
        "section_title": metadata.get("section_title"),
        "text": document,
        "embed_text": embed_text,
        "dimensions": len(embedding),
        "l2_norm": round(_l2_norm(embedding), 6),
        "embedding_preview": [round(v, 6) for v in embedding[:preview_count]],
        "embedding": [round(v, 6) for v in embedding] if show_full else None,
    }


def _rank_by_query(
    records: list[dict[str, Any]],
    query: str,
    *,
    index_dir: Path,
) -> list[dict[str, Any]]:
    model = load_embedding_model()
    query_vector = embed_query_text(model, query)
    ranked = []
    for record in records:
        vector = record["_vector"]
        ranked.append(
            {
                **{k: v for k, v in record.items() if k != "_vector"},
                "similarity": round(_cosine(query_vector, vector), 6),
            }
        )
    ranked.sort(key=lambda item: item["similarity"], reverse=True)
    return ranked


def main() -> int:
    args = parse_args()

    if args.scheme and args.scheme not in SCHEME_IDS:
        print(f"Unknown scheme: {args.scheme!r}. Valid: {sorted(SCHEME_IDS)}", file=sys.stderr)
        return 1

    try:
        manifest = read_index_manifest(args.index_dir)
        stats = get_index_stats(args.index_dir)
        stored = _fetch_records(
            index_dir=args.index_dir,
            scheme_id=args.scheme,
            chunk_id=args.chunk_id,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    if not stored["ids"]:
        print("No matching chunks in the index.", file=sys.stderr)
        return 1

    embed_text_map = _load_embed_text_map(CHUNKS_DATA_DIR)
    records: list[dict[str, Any]] = []
    for chunk_id, embedding, document, metadata in zip(
        stored["ids"],
        stored["embeddings"],
        stored["documents"],
        stored["metadatas"],
        strict=True,
    ):
        item = _build_record(
            chunk_id=chunk_id,
            embedding=_as_floats(embedding),
            document=document,
            metadata=metadata,
            embed_text=embed_text_map.get(chunk_id, ""),
            preview_dims=max(0, args.preview),
            show_full=args.full,
        )
        item["_vector"] = _as_floats(embedding)
        records.append(item)

    summary = {
        "embedding_model": stats.get("embedding_model") or manifest.get("embedding_model"),
        "embedded_at": stats.get("embedded_at") or manifest.get("embedded_at"),
        "total_chunks_in_index": stats.get("total_chunks"),
        "chunks_shown": len(records),
        "embedding_dimensions": stats.get("embedding_dimensions"),
        "chunks_per_scheme": stats.get("chunks_per_scheme"),
    }

    if args.query:
        records = _rank_by_query(records, args.query, index_dir=args.index_dir)
        summary["query"] = args.query

    for record in records:
        record.pop("_vector", None)
        if not args.full:
            record.pop("embedding", None)

    payload = {"summary": summary, "chunks": records}

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print(f"Model: {summary['embedding_model']}")
    print(f"Embedded at: {summary['embedded_at']}")
    print(f"Dimensions: {summary['embedding_dimensions']}")
    print(f"Showing {summary['chunks_shown']} of {summary['total_chunks_in_index']} chunk(s)")
    if args.query:
        print(f"Query: {args.query!r} (ranked by cosine similarity)")
    print()

    for index, record in enumerate(records, start=1):
        header = record["chunk_id"]
        if args.query:
            header = f"{index:2d}. sim={record['similarity']:.4f}  {header}"
        print(header)
        print(f"    scheme: {record['scheme_id']}  section: {record['section_title']}")
        print(f"    norm: {record['l2_norm']}")
        if record["embed_text"]:
            text = record["embed_text"]
            if len(text) > 120:
                text = text[:117] + "..."
            print(f"    embed_text: {text}")
        print(f"    text: {record['text'][:100]}{'...' if len(record['text']) > 100 else ''}")
        if args.full:
            print(f"    vector: {record.get('embedding')}")
        else:
            print(f"    preview: {record['embedding_preview']}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
