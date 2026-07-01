#!/usr/bin/env python3
"""Run the Phase 3 pipeline for a single query (requires GROQ_API_KEY for factual queries)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from src.rag.pipeline import PipelineStats, answer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the full RAG answer pipeline.")
    parser.add_argument("query", help="User question.")
    return parser.parse_args()


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    stats = PipelineStats()
    response = answer(args.query, stats=stats)
    print(json.dumps(response.to_dict(), indent=2, ensure_ascii=False))
    print(f"\nGroq calls: {stats.groq_calls}  Retrieval calls: {stats.retrieval_calls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
