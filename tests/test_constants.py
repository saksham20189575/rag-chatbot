"""Smoke tests for shared constants (Phase 0)."""

from src.constants import (
    ALLOWLISTED_GROWW_URLS,
    SCHEME_ALIASES,
    SCHEME_IDS,
    SCHEME_URLS,
)


def test_allowlisted_urls_match_schemes():
    assert len(ALLOWLISTED_GROWW_URLS) == 5
    assert set(SCHEME_URLS.values()) == set(ALLOWLISTED_GROWW_URLS)


def test_scheme_ids():
    assert SCHEME_IDS == {"large_cap", "mid_cap", "small_cap", "gold_fof", "silver_fof"}


def test_scheme_aliases_map_to_valid_ids():
    for scheme_id in SCHEME_ALIASES.values():
        assert scheme_id in SCHEME_IDS
