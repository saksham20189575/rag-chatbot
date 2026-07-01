"""End-to-end corpus build pipeline: fetch → parse → chunk → embed → index."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.constants import (
    ALLOWLISTED_GROWW_URLS,
    CHUNKS_DATA_DIR,
    INDEX_DIR,
    PARSED_DATA_DIR,
    RAW_DATA_DIR,
    SCHEME_IDS,
)
from src.ingestion.chunker import CHUNK_MANIFEST_FILENAME, Chunk, chunk_all, save_chunks
from src.ingestion.fetcher import FetchResult, fetch_all, load_schemes
from src.ingestion.indexer import (
    INDEX_MANIFEST_FILENAME,
    IndexResult,
    build_index,
    chunks_fingerprint,
    get_index_stats,
    load_saved_chunks,
    read_index_manifest,
)
from src.ingestion.parser import PARSED_MANIFEST_FILENAME, parse_all, save_parsed_pages

logger = logging.getLogger(__name__)

EXPECTED_CHUNKS_PER_SCHEME = 7
EXPECTED_TOTAL_CHUNKS = 35


@dataclass
class BuildOptions:
    force_refresh: bool = False
    scheme_id: str | None = None
    parse_only: bool = False
    index_only: bool = False


@dataclass
class BuildResult:
    fetch_results: list[FetchResult] = field(default_factory=list)
    parsed_scheme_count: int = 0
    chunks: list[Chunk] = field(default_factory=list)
    chunks_per_scheme: dict[str, int] = field(default_factory=dict)
    index_result: IndexResult | None = None
    index_skipped: bool = False
    validation_errors: list[str] = field(default_factory=list)


def validate_scheme_id(scheme_id: str | None) -> None:
    if scheme_id is None:
        return
    if scheme_id not in SCHEME_IDS:
        raise ValueError(f"Unknown scheme_id: {scheme_id!r}. Valid: {sorted(SCHEME_IDS)}")


def validate_build_options(options: BuildOptions) -> None:
    if options.parse_only and options.index_only:
        raise ValueError("Use either --parse-only or --index-only, not both.")
    validate_scheme_id(options.scheme_id)


def count_chunks_per_scheme(chunks: list[Chunk]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.scheme_id] = counts.get(chunk.scheme_id, 0) + 1
    return dict(sorted(counts.items()))


def should_skip_indexing(
    chunks: list[Chunk],
    *,
    options: BuildOptions,
    index_dir: Path | None = None,
) -> bool:
    """Return True when corpus is unchanged and re-embed is unnecessary."""
    if options.force_refresh or options.scheme_id is not None:
        return False

    manifest = read_index_manifest(index_dir)
    if manifest is None:
        return False

    fingerprint = chunks_fingerprint(chunks)
    return manifest.get("chunks_fingerprint") == fingerprint


def validate_corpus(
    chunks: list[Chunk],
    *,
    scheme_id: str | None = None,
    index_result: IndexResult | None = None,
) -> list[str]:
    """Check Phase 1 exit criteria; return a list of validation error messages."""
    errors: list[str] = []

    if not chunks:
        errors.append("No chunks produced.")
        return errors

    per_scheme = count_chunks_per_scheme(chunks)

    if scheme_id is None:
        if len(per_scheme) != len(SCHEME_IDS):
            errors.append(
                f"Expected {len(SCHEME_IDS)} schemes in chunks, found {len(per_scheme)}."
            )
        if len(chunks) != EXPECTED_TOTAL_CHUNKS:
            errors.append(
                f"Expected {EXPECTED_TOTAL_CHUNKS} total chunks, found {len(chunks)}."
            )
        for sid in SCHEME_IDS:
            count = per_scheme.get(sid, 0)
            if count != EXPECTED_CHUNKS_PER_SCHEME:
                errors.append(
                    f"Expected {EXPECTED_CHUNKS_PER_SCHEME} chunks for {sid}, found {count}."
                )
    else:
        count = per_scheme.get(scheme_id, 0)
        if count != EXPECTED_CHUNKS_PER_SCHEME:
            errors.append(
                f"Expected {EXPECTED_CHUNKS_PER_SCHEME} chunks for {scheme_id}, found {count}."
            )

    for chunk in chunks:
        if chunk.source_url not in ALLOWLISTED_GROWW_URLS:
            errors.append(f"Non-allowlisted URL in chunk {chunk.chunk_id}: {chunk.source_url}")
        if ".pdf" in chunk.source_url.lower():
            errors.append(f"PDF URL in chunk {chunk.chunk_id}: {chunk.source_url}")
        if "groww.in" not in chunk.source_url:
            errors.append(f"Non-Groww URL in chunk {chunk.chunk_id}: {chunk.source_url}")
        if chunk.section_title == "Fund management" and "Also manages these schemes" in chunk.text:
            errors.append(f"Fund management noise in chunk {chunk.chunk_id}.")

    large_cap_facts = [
        c for c in chunks if c.scheme_id == "large_cap" and c.section_title == "Key fund facts"
    ]
    if scheme_id is None and large_cap_facts:
        text = large_cap_facts[0].text
        if "1.04%" not in text:
            errors.append("Large Cap key fund facts missing expense ratio ~1.04%.")
        if "₹100" not in text:
            errors.append("Large Cap key fund facts missing min SIP ₹100.")

    if index_result is not None and index_result.chunk_count == 0:
        errors.append("Index contains zero chunks.")

    return errors


def run_build(options: BuildOptions | None = None) -> BuildResult:
    """Orchestrate fetch → parse → chunk → embed → index."""
    opts = options or BuildOptions()
    validate_build_options(opts)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    schemes = load_schemes()
    result = BuildResult()

    if opts.index_only:
        index_result = build_index(scheme_id=opts.scheme_id)
        result.index_result = index_result
        result.chunks = (
            load_saved_chunks()
            if opts.scheme_id is None
            else load_saved_chunks(scheme_id=opts.scheme_id)
        )
        result.chunks_per_scheme = count_chunks_per_scheme(result.chunks)
        result.validation_errors = validate_corpus(
            result.chunks,
            scheme_id=opts.scheme_id,
            index_result=index_result,
        )
        return result

    if not opts.parse_only:
        result.fetch_results = fetch_all(
            force_refresh=opts.force_refresh,
            scheme_id=opts.scheme_id,
        )

    parsed_pages = parse_all(scheme_id=opts.scheme_id)
    save_parsed_pages(parsed_pages)
    result.parsed_scheme_count = len(parsed_pages)

    chunks = chunk_all(
        scheme_id=opts.scheme_id,
        parsed_pages=parsed_pages,
        schemes=schemes,
    )
    save_chunks(chunks)
    result.chunks = chunks
    result.chunks_per_scheme = count_chunks_per_scheme(chunks)

    if opts.parse_only:
        result.validation_errors = validate_corpus(chunks, scheme_id=opts.scheme_id)
        return result

    if should_skip_indexing(chunks, options=opts):
        logger.info("Chunks unchanged — skipping re-embed (IX-05)")
        result.index_skipped = True
        stats = get_index_stats()
        manifest = read_index_manifest()
        result.index_result = IndexResult(
            chunk_count=stats["total_chunks"],
            embedding_model=stats["embedding_model"] or "",
            embedded_at=manifest.get("embedded_at", "") if manifest else "",
            index_dir=INDEX_DIR,
            collection_name=stats["collection_name"],
            scheme_ids=sorted(stats["chunks_per_scheme"]),
        )
    else:
        all_chunks = load_saved_chunks()
        index_result = build_index(
            chunks=chunks,
            scheme_id=opts.scheme_id,
            content_fingerprint=chunks_fingerprint(all_chunks),
        )
        result.index_result = index_result

    result.validation_errors = validate_corpus(
        chunks,
        scheme_id=opts.scheme_id,
        index_result=result.index_result,
    )
    return result


def format_build_summary(result: BuildResult) -> str:
    """Human-readable build summary for CLI output."""
    lines: list[str] = []

    if result.fetch_results:
        fetched = sum(1 for r in result.fetch_results if not r.skipped)
        skipped = sum(1 for r in result.fetch_results if r.skipped)
        lines.append(f"Fetch: {fetched} updated, {skipped} unchanged")
        for fetch in result.fetch_results:
            status = "skipped" if fetch.skipped else fetch.fetch_method
            lines.append(f"  {fetch.scheme_id}: {status} ({fetch.content_hash[:12]}…)")

    if result.parsed_scheme_count:
        lines.append(f"Parse: {result.parsed_scheme_count} scheme(s) → {PARSED_DATA_DIR}/")

    if result.chunks:
        lines.append(
            f"Chunk: {len(result.chunks)} chunk(s) across {len(result.chunks_per_scheme)} scheme(s)"
        )
        for scheme_id, count in result.chunks_per_scheme.items():
            lines.append(f"  {scheme_id}: {count} chunks")
        lines.append(f"  Review: {CHUNKS_DATA_DIR}/ ({CHUNK_MANIFEST_FILENAME})")

    if result.index_result:
        if result.index_skipped:
            lines.append("Index: skipped (unchanged corpus, IX-05)")
        else:
            lines.append("Index: embedded and persisted")
        lines.append(f"  Total vectors: {result.index_result.chunk_count}")
        lines.append(f"  Schemes indexed: {', '.join(result.index_result.scheme_ids)}")
        lines.append(f"  Model: {result.index_result.embedding_model}")
        lines.append(f"  Embedded at: {result.index_result.embedded_at}")
        lines.append(f"  Index dir: {result.index_result.index_dir}")
        lines.append(f"  Manifest: {INDEX_MANIFEST_FILENAME}")

        stats = get_index_stats(result.index_result.index_dir)
        if stats.get("embedding_dimensions"):
            lines.append(f"  Vector dimensions: {stats['embedding_dimensions']}")

    if result.validation_errors:
        lines.append("Validation:")
        for error in result.validation_errors:
            lines.append(f"  FAIL: {error}")
    else:
        lines.append("Validation: all Phase 1 exit checks passed")

    return "\n".join(lines)
