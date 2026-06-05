"""Core search function unit-level contract tests.

Constructs a ``SearchState`` directly with small random data
so ``find_similar_products(product_id, num_similar)`` runs
without loading real ML models or the full dataset.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from sap_cxii_tech_ex_01.search import SearchState, find_similar_products, set_search_state


# ── Helpers & fixtures ────────────────────────────────────────────────────────

def _make_search_state(n: int = 8, dim: int = 16, seed: int = 0) -> SearchState:
    """Build a SearchState with *n* products and random normalised vectors."""
    rng = np.random.default_rng(seed)
    ids = [f"prod_{i:03d}" for i in range(n)]
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    normalized = vecs / norms
    prices = rng.uniform(100, 2000, n).astype(np.float64)
    return SearchState(ids=ids, normalized=normalized, prices=prices)


@pytest.fixture(autouse=True)
def _install_state() -> None:
    set_search_state(_make_search_state())


# ── Public API signature compliance ──────────────────────────────────────────

def test_public_signature_matches_readme_spec():
    """find_similar_products(product_id, num_similar) — exactly two positional params."""
    sig = inspect.signature(find_similar_products)
    assert list(sig.parameters.keys()) == ["product_id", "num_similar"]


# ── Sanity ────────────────────────────────────────────────────────────────────

def test_known_product_not_in_results():
    """Query product must not appear in its own results."""
    results = find_similar_products("prod_000", 3)
    assert "prod_000" not in results
    assert len(results) == 3


# ── num_similar respected ─────────────────────────────────────────────────────

@pytest.mark.parametrize("k", [1, 2, 4])
def test_num_similar_respected(k):
    """Return exactly num_similar product IDs."""
    assert len(find_similar_products("prod_001", k)) == k


# ── Invalid product_id raises ValueError ─────────────────────────────────────

def test_invalid_product_id_raises_value_error():
    """Unknown product_id must raise ValueError."""
    with pytest.raises(ValueError, match="prod_nonexistent"):
        find_similar_products("prod_nonexistent", 3)


# ── Determinism ───────────────────────────────────────────────────────────────

def test_results_are_deterministic():
    """Same inputs must always produce the same output."""
    assert find_similar_products("prod_002", 3) == find_similar_products("prod_002", 3)


# ── Return type ───────────────────────────────────────────────────────────────

def test_returns_list_of_strings():
    """Return value must be a list[str]."""
    results = find_similar_products("prod_003", 2)
    assert isinstance(results, list)
    assert all(isinstance(r, str) for r in results)


# ── No duplicates ─────────────────────────────────────────────────────────────

def test_no_duplicate_ids_in_results():
    """All returned product IDs must be unique."""
    results = find_similar_products("prod_004", 5)
    assert len(results) == len(set(results))
