#!/usr/bin/env python3
"""Build the Groww corpus vector index (fetch → parse → chunk → embed → index)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import CHUNKS_DATA_DIR, INDEX_DIR, PARSED_DATA_DIR, RAW_DATA_DIR  # noqa: E402
from src.ingestion.fetcher import load_schemes  # noqa: E402
from src.ingestion.pipeline import BuildOptions, format_build_summary, run_build  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the HDFC MF Groww corpus index.")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-fetch all Groww pages even if content hash is unchanged; always re-embed.",
    )
    parser.add_argument(
        "--scheme",
        metavar="SCHEME_ID",
        help="Rebuild a single scheme (e.g. large_cap, mid_cap).",
    )
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Skip fetch and index; parse saved raw HTML and write review files through data/chunks/.",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Skip fetch/parse/chunk; embed saved data/chunks/ into the vector index.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    try:
        schemes = load_schemes()
        options = BuildOptions(
            force_refresh=args.force_refresh,
            scheme_id=args.scheme,
            parse_only=args.parse_only,
            index_only=args.index_only,
        )
        result = run_build(options)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Corpus: {len(schemes)} scheme(s) configured.")
    print(f"Raw data dir: {RAW_DATA_DIR}")
    print(f"Parsed data dir: {PARSED_DATA_DIR}")
    print(f"Chunks data dir: {CHUNKS_DATA_DIR}")
    print(f"Index dir: {INDEX_DIR}")
    print()
    print(format_build_summary(result))

    if result.validation_errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
