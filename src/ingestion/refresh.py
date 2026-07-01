"""Hash-aware corpus refresh for scheduled ingestion (Phase 6)."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from src.constants import (
    CHUNKS_DATA_DIR,
    DATA_DIR,
    INDEX_DIR,
    PARSED_DATA_DIR,
    RAW_DATA_DIR,
    SCHEME_IDS,
)
from src.ingestion.chunker import Chunk, chunk_all, save_chunks
from src.ingestion.fetcher import FetchResult, fetch_all, load_schemes
from src.ingestion.indexer import (
    IndexResult,
    build_index,
    chunks_fingerprint,
    get_index_stats,
    index_chunks,
    load_saved_chunks,
    read_index_manifest,
)
from src.ingestion.parser import parse_all, save_parsed_pages
from src.ingestion.pipeline import (
    BuildOptions,
    should_skip_indexing,
    validate_corpus,
    validate_scheme_id,
)

logger = logging.getLogger(__name__)

REFRESH_MANIFEST_FILENAME = "refresh_manifest.json"
REFRESH_MANIFEST_PATH = DATA_DIR / REFRESH_MANIFEST_FILENAME
IST = ZoneInfo("Asia/Kolkata")


@dataclass
class RefreshOptions:
    force_refresh: bool = False
    scheme_id: str | None = None
    dry_run: bool = False
    trigger: str = "manual"
    workflow_run_id: str | None = None


@dataclass
class RefreshResult:
    fetch_results: list[FetchResult] = field(default_factory=list)
    schemes_checked: int = 0
    schemes_updated: int = 0
    schemes_skipped: int = 0
    schemes_reindexed: int = 0
    chunks: list[Chunk] = field(default_factory=list)
    chunk_count: int = 0
    index_result: IndexResult | None = None
    index_skipped: bool = False
    errors: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    fatal: bool = False


def now_ist() -> str:
    return datetime.now(IST).replace(microsecond=0).isoformat()


def read_refresh_manifest(path: Path | None = None) -> dict[str, Any] | None:
    manifest_path = path or REFRESH_MANIFEST_PATH
    if not manifest_path.is_file():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def write_refresh_manifest(
    result: RefreshResult,
    *,
    options: RefreshOptions,
    status: str,
) -> Path:
    """Persist run metadata; preserve last_success_at on failure."""
    REFRESH_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    previous = read_refresh_manifest() or {}

    payload: dict[str, Any] = {
        "last_run_at": now_ist(),
        "status": status,
        "trigger": options.trigger,
        "schemes_checked": result.schemes_checked,
        "schemes_updated": result.schemes_updated,
        "schemes_skipped": result.schemes_skipped,
        "schemes_reindexed": result.schemes_reindexed,
        "chunk_count": result.chunk_count,
        "duration_seconds": round(result.duration_seconds, 1),
        "errors": result.errors,
        "dry_run": options.dry_run,
    }
    if options.workflow_run_id:
        payload["workflow_run_id"] = options.workflow_run_id

    if status == "success":
        payload["last_success_at"] = now_ist()
    else:
        payload["last_success_at"] = previous.get("last_success_at")

    REFRESH_MANIFEST_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return REFRESH_MANIFEST_PATH


def _chunks_available() -> bool:
    return CHUNKS_DATA_DIR.is_dir() and any(CHUNKS_DATA_DIR.glob("*.json"))


def _updated_scheme_ids(fetch_results: list[FetchResult], *, force_refresh: bool) -> list[str]:
    if force_refresh:
        return [r.scheme_id for r in fetch_results]
    return [r.scheme_id for r in fetch_results if not r.skipped]


def _process_scheme(
    scheme_id: str,
    *,
    errors: list[str],
) -> list[Chunk] | None:
    try:
        pages = parse_all(scheme_id=scheme_id)
        save_parsed_pages(pages)
        return chunk_all(scheme_id=scheme_id, parsed_pages=pages)
    except Exception as exc:
        message = f"{scheme_id}: parse/chunk failed — {exc}"
        logger.exception(message)
        errors.append(message)
        return None


def run_refresh(options: RefreshOptions | None = None) -> RefreshResult:
    """Fetch, then parse/chunk/index only schemes whose raw HTML changed."""
    opts = options or RefreshOptions()
    validate_scheme_id(opts.scheme_id)

    started = time.monotonic()
    result = RefreshResult()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    schemes = load_schemes()
    if opts.scheme_id is not None:
        schemes = [s for s in schemes if s.scheme_id == opts.scheme_id]
        if not schemes:
            result.fatal = True
            result.errors.append(f"Unknown scheme_id: {opts.scheme_id!r}")
            result.duration_seconds = time.monotonic() - started
            return result

    try:
        result.fetch_results = fetch_all(
            force_refresh=opts.force_refresh,
            scheme_id=opts.scheme_id,
        )
    except Exception as exc:
        result.fatal = True
        result.errors.append(f"Fetch failed: {exc}")
        result.duration_seconds = time.monotonic() - started
        return result

    result.schemes_checked = len(result.fetch_results)
    updated_ids = _updated_scheme_ids(result.fetch_results, force_refresh=opts.force_refresh)
    result.schemes_updated = len(updated_ids)
    result.schemes_skipped = result.schemes_checked - result.schemes_updated

    if opts.dry_run:
        logger.info(
            "Dry run — %s scheme(s) would update, %s unchanged",
            result.schemes_updated,
            result.schemes_skipped,
        )
        if _chunks_available():
            try:
                result.chunks = load_saved_chunks()
                result.chunk_count = len(result.chunks)
            except (FileNotFoundError, ValueError):
                pass
        result.duration_seconds = time.monotonic() - started
        return result

    if updated_ids:
        new_chunks_by_scheme: dict[str, list[Chunk]] = {}
        for scheme_id in updated_ids:
            scheme_chunks = _process_scheme(scheme_id, errors=result.errors)
            if scheme_chunks is not None:
                new_chunks_by_scheme[scheme_id] = scheme_chunks

        if new_chunks_by_scheme:
            try:
                if _chunks_available():
                    all_chunks = load_saved_chunks()
                    by_scheme = {sid: [c for c in all_chunks if c.scheme_id == sid] for sid in SCHEME_IDS}
                    for sid, chunks in new_chunks_by_scheme.items():
                        by_scheme[sid] = chunks
                    merged = [c for sid in sorted(by_scheme) for c in by_scheme.get(sid, [])]
                else:
                    merged = [c for chunks in new_chunks_by_scheme.values() for c in chunks]
                save_chunks(merged)
                result.chunks = merged
            except Exception as exc:
                result.errors.append(f"Failed to save chunks: {exc}")
        elif result.errors:
            result.duration_seconds = time.monotonic() - started
            return result
    else:
        logger.info("All schemes unchanged at fetch — skipping parse/chunk")

    try:
        if result.chunks:
            pass
        elif _chunks_available():
            result.chunks = load_saved_chunks()
        else:
            result.fatal = True
            result.errors.append("No saved chunks found. Run scripts/build_index.py first.")
            result.duration_seconds = time.monotonic() - started
            return result
    except (FileNotFoundError, ValueError) as exc:
        result.fatal = True
        result.errors.append(str(exc))
        result.duration_seconds = time.monotonic() - started
        return result

    result.chunk_count = len(result.chunks)

    build_opts = BuildOptions(
        force_refresh=opts.force_refresh,
        scheme_id=opts.scheme_id,
    )
    if updated_ids and not should_skip_indexing(result.chunks, options=build_opts):
        try:
            if opts.scheme_id is not None:
                scheme_chunks = [c for c in result.chunks if c.scheme_id == opts.scheme_id]
                result.index_result = index_chunks(
                    scheme_chunks,
                    scheme_id=opts.scheme_id,
                    content_fingerprint=chunks_fingerprint(result.chunks),
                )
                result.schemes_reindexed = 1
            elif len(updated_ids) < len(SCHEME_IDS) and read_index_manifest() is not None:
                for scheme_id in sorted(updated_ids):
                    scheme_chunks = [c for c in result.chunks if c.scheme_id == scheme_id]
                    result.index_result = index_chunks(
                        scheme_chunks,
                        scheme_id=scheme_id,
                        content_fingerprint=chunks_fingerprint(result.chunks),
                    )
                result.schemes_reindexed = len(updated_ids)
            else:
                result.index_result = build_index(
                    chunks=result.chunks,
                    content_fingerprint=chunks_fingerprint(result.chunks),
                )
                result.schemes_reindexed = len(updated_ids)
        except Exception as exc:
            result.errors.append(f"Indexing failed: {exc}")
            logger.exception("Indexing failed during refresh")
    else:
        result.index_skipped = True
        if read_index_manifest() is not None:
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
            result.chunk_count = stats["total_chunks"]

    result.validation_errors = validate_corpus(
        result.chunks,
        scheme_id=opts.scheme_id,
        index_result=result.index_result,
    )
    result.duration_seconds = time.monotonic() - started
    return result


def refresh_exit_code(result: RefreshResult) -> int:
    """Map refresh outcome to process exit code."""
    if result.fatal:
        return 2
    if result.errors or result.validation_errors:
        return 1
    return 0


def refresh_status(result: RefreshResult) -> str:
    if result.fatal:
        return "fatal"
    if result.errors or result.validation_errors:
        return "partial_failure"
    return "success"


def format_refresh_summary(result: RefreshResult, *, options: RefreshOptions) -> str:
    lines = [
        f"Refresh: {result.schemes_checked} checked, "
        f"{result.schemes_updated} updated, {result.schemes_skipped} skipped",
    ]
    if options.dry_run:
        lines.append("Mode: dry run (fetch + hash compare only)")
    if result.schemes_reindexed:
        lines.append(f"Re-indexed: {result.schemes_reindexed} scheme(s)")
    elif result.index_skipped:
        lines.append("Index: skipped (corpus unchanged)")
    if result.chunk_count:
        lines.append(f"Chunk count: {result.chunk_count}")
    lines.append(f"Duration: {result.duration_seconds:.1f}s")
    if result.errors:
        lines.append("Errors:")
        for error in result.errors:
            lines.append(f"  - {error}")
    if result.validation_errors:
        lines.append("Validation:")
        for error in result.validation_errors:
            lines.append(f"  FAIL: {error}")
    return "\n".join(lines)


def write_github_step_summary(result: RefreshResult, *, options: RefreshOptions) -> None:
    """Append refresh stats to GITHUB_STEP_SUMMARY when running in Actions."""
    summary_env = os.environ.get("GITHUB_STEP_SUMMARY", "").strip()
    if not summary_env:
        return
    summary_path = Path(summary_env)
    lines = [
        "## Corpus refresh",
        "",
        f"- **Checked:** {result.schemes_checked}",
        f"- **Updated:** {result.schemes_updated}",
        f"- **Skipped (unchanged):** {result.schemes_skipped}",
        f"- **Re-indexed:** {result.schemes_reindexed}",
        f"- **Chunk count:** {result.chunk_count}",
        f"- **Duration:** {result.duration_seconds:.1f}s",
        f"- **Dry run:** {options.dry_run}",
    ]
    if result.errors:
        lines.extend(["", "### Errors", ""] + [f"- {e}" for e in result.errors])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
