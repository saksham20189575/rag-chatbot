#!/usr/bin/env python3
"""Refresh the Groww corpus on a schedule (fetch → parse → chunk → embed → index)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.refresh import (  # noqa: E402
    RefreshOptions,
    format_refresh_summary,
    refresh_exit_code,
    refresh_status,
    run_refresh,
    write_github_step_summary,
    write_refresh_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh the HDFC MF Groww corpus (hash-aware incremental update).",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-fetch all Groww pages and re-process even when content hash is unchanged.",
    )
    parser.add_argument(
        "--scheme",
        metavar="SCHEME_ID",
        help="Refresh a single scheme (e.g. large_cap, mid_cap).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and compare content hashes only; do not parse, chunk, or re-index.",
    )
    parser.add_argument(
        "--trigger",
        default="manual",
        choices=["manual", "github_actions"],
        help="Provenance label written to refresh_manifest.json.",
    )
    parser.add_argument(
        "--workflow-run-id",
        metavar="ID",
        default=None,
        help="GitHub Actions workflow run ID for manifest provenance.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    workflow_run_id = args.workflow_run_id or os.environ.get("GITHUB_RUN_ID")

    options = RefreshOptions(
        force_refresh=args.force_refresh,
        scheme_id=args.scheme,
        dry_run=args.dry_run,
        trigger=args.trigger,
        workflow_run_id=workflow_run_id,
    )

    try:
        result = run_refresh(options)
    except Exception as exc:
        logging.exception("Fatal refresh error")
        print(exc, file=sys.stderr)
        return 2

    status = refresh_status(result)
    write_refresh_manifest(result, options=options, status=status)
    write_github_step_summary(result, options=options)

    print(format_refresh_summary(result, options=options))
    print()
    print(f"Manifest: data/refresh_manifest.json (status={status})")

    return refresh_exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
