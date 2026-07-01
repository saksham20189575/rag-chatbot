"""Chunk parsed Groww fund pages into retrieval units with §5.1 metadata.

Strategy (tuned to the parsed data, not a generic 400–800 token packer):

- **Section-atomic chunks.** Groww fund pages yield small, fact-dense sections
  (expense ratio, exit load, benchmark, etc.), each well under 100 tokens.
  Merging them to hit a 400–800 token target would collapse every fact of a
  scheme into one blob, so distinct queries ("expense ratio?", "exit load?")
  would all retrieve the same chunk. Keeping one chunk per section maximises
  retrieval precision for a facts-only FAQ.
- **Noise stripping.** The "Fund management" section embeds long "Also manages
  these schemes ..." lists that name *other* HDFC schemes — a cross-scheme
  retrieval leakage risk (edge-case RT-05). Those lists are removed; manager
  name, tenure, education, and experience are kept.
- **Scheme-context embedding text.** Short facts ("Exit load of 1% ...") are
  ambiguous across schemes in embedding space, so each chunk's `embed_text`
  is prefixed with the scheme name and section title. Hard scheme filtering
  still uses `scheme_id` metadata at retrieval time.
- **Overflow split.** `MAX_CHUNK_TOKENS` only triggers an overlap-aware split
  for unusually large sections; in practice sections fit in a single chunk.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.constants import (
    ALLOWED_DOMAIN,
    CHUNK_OVERLAP_TOKENS,
    CHUNKS_DATA_DIR,
    DOCUMENT_TYPE,
    MAX_CHUNK_TOKENS,
)
from src.ingestion.fetcher import SchemeConfig, load_schemes
from src.ingestion.parser import ParsedPage, TextBlock, parse_all

logger = logging.getLogger(__name__)

CHUNK_MANIFEST_FILENAME = "chunk_manifest.json"

_MONTHS = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"

# Start of a fund-manager entry, e.g. "CS Chirag Setalvad Jan 2013 - Present".
# \b prevents matching "FC" inside "HDFC"; the name is bounded to 1–4 words so
# the pattern can't greedily swallow an "Also manages" scheme list as a name.
_MANAGER_HEADER = re.compile(
    rf"\b[A-Z]{{2}}\s+[A-Z][A-Za-z.]+(?:\s+[A-Z][A-Za-z.]+){{0,3}}\s+(?:{_MONTHS})\s+\d{{4}}\s*-\s*Present"
)
_ALSO_MANAGES = "Also manages these schemes"

_MONTH_TO_NUM = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
    "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}


@dataclass
class Chunk:
    chunk_id: str
    scheme_id: str
    scheme_name: str
    document_type: str
    source_url: str
    source_domain: str
    section_title: str
    last_updated: str
    text: str
    embed_text: str
    token_estimate: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def metadata(self) -> dict[str, Any]:
        """Metadata payload for the vector store (excludes embed_text)."""
        return {
            "chunk_id": self.chunk_id,
            "scheme_id": self.scheme_id,
            "scheme_name": self.scheme_name,
            "document_type": self.document_type,
            "source_url": self.source_url,
            "source_domain": self.source_domain,
            "section_title": self.section_title,
            "last_updated": self.last_updated,
        }


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def estimate_tokens(text: str) -> int:
    """Approximate subword token count (BGE/BERT-style ≈ words / 0.75)."""
    words = len(text.split())
    if words == 0:
        return 0
    return max(1, round(words / 0.75))


def normalize_last_updated(nav_date: str | None, fetched_at: str | None) -> str:
    """Convert Groww's '25-Jun-2026' NAV date to ISO 'YYYY-MM-DD'.

    Falls back to the fetch date when NAV date is missing or unparseable.
    """
    if nav_date:
        match = re.match(r"(\d{1,2})-([A-Za-z]{3})-(\d{4})", nav_date.strip())
        if match:
            day, mon, year = match.groups()
            month = _MONTH_TO_NUM.get(mon.lower())
            if month:
                return f"{year}-{month}-{int(day):02d}"
    if fetched_at:
        return fetched_at[:10]
    return datetime.now(UTC).strftime("%Y-%m-%d")


def clean_fund_management(text: str) -> str:
    """Strip 'Also manages these schemes ...' lists and 'View details' chrome.

    Keeps each manager's name, tenure, education, and experience while removing
    the other-scheme lists that cause cross-scheme retrieval leakage.
    """
    headers = list(_MANAGER_HEADER.finditer(text))
    if not headers:
        cleaned = text.split(_ALSO_MANAGES)[0]
        return _normalize_whitespace(cleaned.replace("View details", " "))

    parts: list[str] = []
    for index, match in enumerate(headers):
        start = match.start()
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        entry = text[start:end]
        entry = entry.split(_ALSO_MANAGES)[0]
        entry = entry.replace("View details", " ")
        entry = _normalize_whitespace(entry)
        if entry:
            parts.append(entry)
    return " ".join(parts)


def _clean_section_text(section_title: str, text: str) -> str:
    if section_title.casefold() == "fund management":
        return clean_fund_management(text)
    return _normalize_whitespace(text)


def _split_with_overlap(
    text: str,
    *,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Word-window split for oversized sections, preserving token overlap."""
    words = text.split()
    if estimate_tokens(text) <= max_tokens:
        return [text]

    max_words = max(1, round(max_tokens * 0.75))
    overlap_words = min(max(0, round(overlap_tokens * 0.75)), max_words - 1)
    step = max(1, max_words - overlap_words)

    pieces: list[str] = []
    start = 0
    while start < len(words):
        window = words[start : start + max_words]
        pieces.append(" ".join(window))
        if start + max_words >= len(words):
            break
        start += step
    return pieces


def _build_embed_text(scheme_name: str, section_title: str, text: str) -> str:
    return f"{scheme_name} | {section_title}: {text}"


def chunk_page(
    page: ParsedPage,
    *,
    scheme: SchemeConfig,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Convert one parsed page into section-atomic chunks with metadata."""
    last_updated = normalize_last_updated(page.nav_date, page.fetched_at)
    chunks: list[Chunk] = []
    counter = 0

    for block in page.text_blocks:
        cleaned = _clean_section_text(block.section_title, block.text)
        if len(cleaned) < 3:
            continue

        pieces = _split_with_overlap(
            cleaned, max_tokens=max_tokens, overlap_tokens=overlap_tokens
        )
        for piece in pieces:
            chunk_id = f"{scheme.scheme_id}_groww_{counter:04d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    scheme_id=scheme.scheme_id,
                    scheme_name=scheme.scheme_name,
                    document_type=DOCUMENT_TYPE,
                    source_url=scheme.url,
                    source_domain=ALLOWED_DOMAIN,
                    section_title=block.section_title,
                    last_updated=last_updated,
                    text=piece,
                    embed_text=_build_embed_text(scheme.scheme_name, block.section_title, piece),
                    token_estimate=estimate_tokens(piece),
                )
            )
            counter += 1

    if not chunks:
        raise ValueError(f"No chunks produced for scheme {scheme.scheme_id!r}")
    return chunks


def chunk_all(
    *,
    scheme_id: str | None = None,
    raw_dir: Path | None = None,
    parsed_pages: list[ParsedPage] | None = None,
    schemes: list[SchemeConfig] | None = None,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunk all (or one) schemes' parsed pages."""
    configured = schemes or load_schemes()
    by_id = {s.scheme_id: s for s in configured}

    if scheme_id is not None and scheme_id not in by_id:
        raise ValueError(f"Unknown scheme_id: {scheme_id!r}. Valid: {sorted(by_id)}")

    if parsed_pages is None:
        parsed_pages = parse_all(scheme_id=scheme_id, raw_dir=raw_dir, schemes=configured)

    all_chunks: list[Chunk] = []
    for page in parsed_pages:
        scheme = by_id.get(page.scheme_id)
        if scheme is None:
            logger.warning("No scheme config for parsed page %s — skipping", page.scheme_id)
            continue
        all_chunks.extend(
            chunk_page(page, scheme=scheme, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        )
    return all_chunks


def save_chunks(
    chunks: list[Chunk],
    *,
    chunks_dir: Path | None = None,
) -> list[Path]:
    """Write per-scheme chunk review files plus a summary manifest."""
    out_dir = chunks_dir or CHUNKS_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    by_scheme: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_scheme.setdefault(chunk.scheme_id, []).append(chunk)

    paths: list[Path] = []
    for scheme_id, scheme_chunks in by_scheme.items():
        path = out_dir / f"{scheme_id}.json"
        payload = {
            "scheme_id": scheme_id,
            "scheme_name": scheme_chunks[0].scheme_name,
            "source_url": scheme_chunks[0].source_url,
            "last_updated": scheme_chunks[0].last_updated,
            "chunk_count": len(scheme_chunks),
            "chunks": [chunk.to_dict() for chunk in scheme_chunks],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        paths.append(path)

    manifest = {
        "chunked_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "total_chunks": len(chunks),
        "scheme_count": len(by_scheme),
        "schemes": [
            {
                "scheme_id": scheme_id,
                "chunk_count": len(scheme_chunks),
                "sections": sorted({c.section_title for c in scheme_chunks}),
                "token_min": min(c.token_estimate for c in scheme_chunks),
                "token_max": max(c.token_estimate for c in scheme_chunks),
            }
            for scheme_id, scheme_chunks in by_scheme.items()
        ],
    }
    manifest_path = out_dir / CHUNK_MANIFEST_FILENAME
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    logger.info("Saved %s chunk file(s) to %s", len(paths), out_dir)
    return paths
