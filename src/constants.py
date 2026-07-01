"""Shared constants: allowlisted Groww URLs, scheme IDs, and query aliases."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PARSED_DATA_DIR = DATA_DIR / "parsed"
CHUNKS_DATA_DIR = DATA_DIR / "chunks"
INDEX_DIR = DATA_DIR / "index"

DOCUMENT_TYPE = "groww_fund_page"

# Chunking strategy (Architecture §5.1). Sections on Groww fund pages are small
# and fact-dense, so chunks are section-atomic; MAX_CHUNK_TOKENS only triggers a
# split for unusually large sections.
MAX_CHUNK_TOKENS = 480
CHUNK_OVERLAP_TOKENS = 60
CORPUS_CONFIG_PATH = CONFIG_DIR / "corpus.yaml"

CHROMA_COLLECTION_NAME = "hdfc_mf_groww_corpus"
ALLOWED_DOMAIN = "groww.in"

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_BGE_MODEL = "BAAI/bge-small-en-v1.5"

# Groq generation defaults (Phase 3 — llama-3.3-70b-versatile free tier).
DEFAULT_GROQ_MAX_TOKENS = 150
DEFAULT_GROQ_REQUEST_TIMEOUT = 25
DEFAULT_GROQ_RPM_LIMIT = 25
DEFAULT_GROQ_TPM_LIMIT = 10_000
DEFAULT_GROQ_RPD_LIMIT = 900
DEFAULT_GROQ_TPD_LIMIT = 90_000
DEFAULT_GROQ_TEMPERATURE = 0.1

# BGE v1.5 asymmetric retrieval: queries use this prefix; documents/chunks do not.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DISCLAIMER = "Facts-only. No investment advice."

SCHEME_IDS = frozenset({"large_cap", "mid_cap", "small_cap", "gold_fof", "silver_fof"})

# Exact Groww URLs permitted in citations (Architecture §6.3).
ALLOWLISTED_GROWW_URLS: frozenset[str] = frozenset(
    {
        "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
        "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
        "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
        "https://groww.in/mutual-funds/hdfc-gold-etf-fund-of-fund-direct-plan-growth",
        "https://groww.in/mutual-funds/hdfc-silver-etf-fof-direct-growth",
    }
)

SCHEME_URLS: dict[str, str] = {
    "large_cap": "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
    "mid_cap": "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
    "small_cap": "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
    "gold_fof": "https://groww.in/mutual-funds/hdfc-gold-etf-fund-of-fund-direct-plan-growth",
    "silver_fof": "https://groww.in/mutual-funds/hdfc-silver-etf-fof-direct-growth",
}

SCHEME_NAMES: dict[str, str] = {
    "large_cap": "HDFC Large Cap Fund Direct Growth",
    "mid_cap": "HDFC Mid Cap Fund Direct Growth",
    "small_cap": "HDFC Small Cap Fund Direct Growth",
    "gold_fof": "HDFC Gold ETF Fund of Fund Direct Plan Growth",
    "silver_fof": "HDFC Silver ETF FoF Direct Growth",
}

DEFAULT_SCHEME_ID = "large_cap"
DEFAULT_CITATION_URL = SCHEME_URLS[DEFAULT_SCHEME_ID]

# Query phrase → scheme_id (used by classifier and retriever).
SCHEME_ALIASES: dict[str, str] = {
    "large cap": "large_cap",
    "largecap": "large_cap",
    "large-cap": "large_cap",
    "mid cap": "mid_cap",
    "midcap": "mid_cap",
    "mid-cap": "mid_cap",
    "small cap": "small_cap",
    "smallcap": "small_cap",
    "small-cap": "small_cap",
    "gold etf fof": "gold_fof",
    "gold etf fund of fund": "gold_fof",
    "gold fof": "gold_fof",
    "gold fund": "gold_fof",
    "gold etf": "gold_fof",
    "silver etf fof": "silver_fof",
    "silver fof": "silver_fof",
    "silver etf": "silver_fof",
    "silver fund": "silver_fof",
}

# Abbreviation expansions for query enhancement (Phase 2+).
QUERY_ABBREVIATIONS: dict[str, str] = {
    "ter": "expense ratio",
    "fof": "fund of fund",
}
