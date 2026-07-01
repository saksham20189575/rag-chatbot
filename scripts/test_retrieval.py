#!/usr/bin/env python3
"""Interactive CLI to exercise scheme-aware retrieval."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.retriever import Retriever  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test retrieval against the local Chroma index.")
    parser.add_argument("query", nargs="?", help="Query text (omit for REPL mode).")
    parser.add_argument(
        "--scheme",
        metavar="SCHEME_ID",
        help="Override detected scheme_id (e.g. mid_cap).",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Path to data/index (default: project data/index).",
    )
    return parser.parse_args()


def _print_result(result) -> None:
    print(f"Query:       {result.query}")
    print(f"Expanded:    {result.expanded_query}")
    print(f"Scheme:      {result.scheme_id or '(none detected)'}")
    if not result.chunks:
        print("Results:     (empty — below threshold or no scheme)")
        return
    print(f"Results:     {len(result.chunks)} chunk(s)")
    for idx, chunk in enumerate(result.chunks, start=1):
        preview = chunk.text[:120] + ("…" if len(chunk.text) > 120 else "")
        print(f"\n  [{idx}] score={chunk.score:.4f}  {chunk.section_title}")
        print(f"      {chunk.chunk_id}  ({chunk.scheme_id})")
        print(f"      {preview}")
    print("\nContext block:")
    print("-" * 60)
    print(result.context or "(empty)")
    print("-" * 60)


def main() -> int:
    args = parse_args()
    retriever = Retriever(index_dir=args.index_dir)

    if args.query:
        started = time.perf_counter()
        result = retriever.retrieve(args.query, scheme_id=args.scheme)
        elapsed_ms = (time.perf_counter() - started) * 1000
        _print_result(result)
        print(f"\nLatency: {elapsed_ms:.1f} ms")
        return 0

    print("Mutual Fund FAQ retrieval REPL (empty line to quit)")
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            break
        started = time.perf_counter()
        result = retriever.retrieve(query, scheme_id=args.scheme)
        elapsed_ms = (time.perf_counter() - started) * 1000
        _print_result(result)
        print(f"Latency: {elapsed_ms:.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
