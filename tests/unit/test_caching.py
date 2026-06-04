"""Search result caching contract tests (RED).

Tests define the LRU cache contract: cache hits return identical results,
cache respects max size, and eviction works correctly. All tests must fail
until Step 5 implements the caching layer.
"""
from __future__ import annotations

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def search_cache():
    """Import and construct a SearchCache (will fail until Step 5)."""
    from sap_cxii_tech_ex_01.cache import SearchCache
    return SearchCache(max_size=3)


# ── Cache hit returns identical results ───────────────────────────────────────

def test_cache_hit_returns_same_result(search_cache) -> None:
    """Repeated identical queries must return the same result."""
    result1 = ["prod_001", "prod_002", "prod_003"]
    search_cache.put("prod_000", 3, result1)
    cached = search_cache.get("prod_000", 3)
    assert cached == result1


def test_cache_miss_returns_none(search_cache) -> None:
    """Uncached queries must return None."""
    assert search_cache.get("prod_999", 3) is None


def test_different_k_is_separate_entry(search_cache) -> None:
    """Same product_id with different num_similar are distinct cache entries."""
    result_3 = ["a", "b", "c"]
    result_5 = ["a", "b", "c", "d", "e"]
    search_cache.put("prod_000", 3, result_3)
    search_cache.put("prod_000", 5, result_5)
    assert search_cache.get("prod_000", 3) == result_3
    assert search_cache.get("prod_000", 5) == result_5


# ── LRU eviction ──────────────────────────────────────────────────────────────

def test_cache_respects_max_size(search_cache) -> None:
    """Adding beyond max_size evicts the least-recently-used entry."""
    search_cache.put("a", 1, ["x"])
    search_cache.put("b", 1, ["y"])
    search_cache.put("c", 1, ["z"])
    # Cache is full (max_size=3). Adding one more should evict "a".
    search_cache.put("d", 1, ["w"])
    assert search_cache.get("a", 1) is None  # evicted
    assert search_cache.get("d", 1) == ["w"]  # present


def test_access_refreshes_lru_order(search_cache) -> None:
    """Accessing an entry makes it most-recently-used (not evicted next)."""
    search_cache.put("a", 1, ["x"])
    search_cache.put("b", 1, ["y"])
    search_cache.put("c", 1, ["z"])
    # Access "a" to refresh it
    search_cache.get("a", 1)
    # Now "b" is LRU. Adding "d" should evict "b", not "a".
    search_cache.put("d", 1, ["w"])
    assert search_cache.get("b", 1) is None  # evicted
    assert search_cache.get("a", 1) == ["x"]  # still present


# ── Cache size reporting ──────────────────────────────────────────────────────

def test_len_reports_current_size(search_cache) -> None:
    assert len(search_cache) == 0
    search_cache.put("a", 1, ["x"])
    assert len(search_cache) == 1
    search_cache.put("b", 1, ["y"])
    assert len(search_cache) == 2


# ── Clear ─────────────────────────────────────────────────────────────────────

def test_clear_empties_cache(search_cache) -> None:
    search_cache.put("a", 1, ["x"])
    search_cache.put("b", 1, ["y"])
    search_cache.clear()
    assert len(search_cache) == 0
    assert search_cache.get("a", 1) is None
