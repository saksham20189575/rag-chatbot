"""Query normalization: abbreviation expansion and scheme detection."""

from __future__ import annotations

import re

from src.constants import QUERY_ABBREVIATIONS, SCHEME_ALIASES, SCHEME_IDS


def expand_abbreviations(query: str) -> str:
    """Expand known abbreviations so BGE matches section vocabulary (RT-14)."""
    expanded = query
    for abbrev, replacement in QUERY_ABBREVIATIONS.items():
        pattern = re.compile(rf"\b{re.escape(abbrev)}\b", re.IGNORECASE)
        expanded = pattern.sub(replacement, expanded)
    return expanded


def detect_scheme_id(query: str) -> str | None:
    """Return scheme_id when the query mentions a known alias (longest match wins)."""
    lowered = query.lower()
    matches: list[tuple[int, str]] = []
    for alias, scheme_id in SCHEME_ALIASES.items():
        if alias in lowered and scheme_id in SCHEME_IDS:
            matches.append((len(alias), scheme_id))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0], reverse=True)
    return matches[0][1]


def normalize_query(query: str) -> tuple[str, str | None]:
    """Expand abbreviations and detect scheme in one pass."""
    stripped = " ".join(query.split())
    expanded = expand_abbreviations(stripped)
    scheme_id = detect_scheme_id(expanded)
    return expanded, scheme_id
