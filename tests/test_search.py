"""Core search function contract tests (GREEN/REFACTOR).

Covers unit-level contracts (Step 8 RED) and integration-level assertions
added in Step 10 REFACTOR to validate end-to-end pipeline behaviour.

Tests patch ``sap_cxii_tech_ex_01.search._load_dataset`` and
``sap_cxii_tech_ex_01.search.get_settings`` to inject controlled fixtures,
keeping the public ``find_similar_products(product_id, num_similar)`` signature
intact and avoiding a redundant internal wrapper function.
"""
from __future__ import annotations

import inspect
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import sap_cxii_tech_ex_01.search as _search_mod
from sap_cxii_tech_ex_01.search import find_similar_products


# ── Public API signature compliance ──────────────────────────────────────────

def test_public_signature_matches_readme_spec():
    """find_similar_products(product_id, num_similar) -> list[str].

    The README mandates exactly two positional parameters.
    """
    sig = inspect.signature(find_similar_products)
    params = list(sig.parameters.keys())
    assert params == ["product_id", "num_similar"]


# ── Helpers & fixtures ────────────────────────────────────────────────────────

def _make_tiny_df(n: int = 5) -> pd.DataFrame:
    """Return a minimal preprocessed-style DataFrame with *n* rows."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "uniq_id": [f"prod_{i:03d}" for i in range(n)],
            "product_name": [f"Product {i}" for i in range(n)],
            "brand": ["BrandA"] * n,
            "sales_price": rng.uniform(100, 2000, n),
            "rating": rng.uniform(1, 5, n),
            "delivery_type": ["Standard"] * n,
            "amazon_prime__y_or_n": ["N"] * n,
            "best_seller_tag__y_or_n": ["N"] * n,
            "image_urls__small": [None] * n,
        }
    )


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    return _make_tiny_df(n=8)


@pytest.fixture
def patched(settings, tiny_df):
    """Patch data-loading and settings so find_similar_products uses test fixtures."""
    with (
        patch.object(_search_mod, "get_settings", return_value=settings),
        patch.object(_search_mod, "_load_dataset", return_value=tiny_df),
    ):
        yield


@pytest.fixture
def patched_df(settings):
    """Factory fixture — returns a context manager that patches with a custom df."""
    from contextlib import contextmanager

    @contextmanager
    def _patch(df: pd.DataFrame):
        with (
            patch.object(_search_mod, "get_settings", return_value=settings),
            patch.object(_search_mod, "_load_dataset", return_value=df),
        ):
            yield

    return _patch


# ── Sanity: product returns itself as top result ──────────────────────────────

def test_known_product_returns_itself_as_most_similar(patched):
    """A query product should have cosine similarity 1.0 with itself.

    find_similar_products must NOT include the query product in results,
    but the nearest result should come from the same underlying feature space.
    """
    results = find_similar_products("prod_000", 3)
    assert "prod_000" not in results
    assert len(results) == 3


# ── num_similar respected ─────────────────────────────────────────────────────

@pytest.mark.parametrize("k", [1, 2, 4])
def test_num_similar_respected(patched, k):
    """Return exactly num_similar product IDs."""
    results = find_similar_products("prod_001", k)
    assert len(results) == k


# ── Invalid product_id raises ValueError ─────────────────────────────────────

def test_invalid_product_id_raises_value_error(patched):
    """Unknown product_id must raise ValueError — not KeyError or IndexError."""
    with pytest.raises(ValueError, match="prod_nonexistent"):
        find_similar_products("prod_nonexistent", 3)


# ── Determinism ───────────────────────────────────────────────────────────────

def test_results_are_deterministic(patched):
    """Two calls with identical inputs must return identical results."""
    r1 = find_similar_products("prod_002", 3)
    r2 = find_similar_products("prod_002", 3)
    assert r1 == r2


# ── Return type ───────────────────────────────────────────────────────────────

def test_returns_list_of_strings(patched):
    """Return value must be a list of str product IDs."""
    results = find_similar_products("prod_003", 2)
    assert isinstance(results, list)
    assert all(isinstance(r, str) for r in results)


# ── No duplicate IDs in results ───────────────────────────────────────────────

def test_no_duplicate_ids_in_results(patched):
    """All returned product IDs must be unique."""
    results = find_similar_products("prod_004", 5)
    assert len(results) == len(set(results))


# ── Integration: all returned IDs exist in the input DataFrame ────────────────

def test_results_are_valid_product_ids(patched, tiny_df):
    """Every returned ID must be present in the input DataFrame."""
    valid_ids = set(tiny_df["uniq_id"].tolist())
    results = find_similar_products("prod_001", 4)
    for pid in results:
        assert pid in valid_ids, f"{pid!r} not in input DataFrame"


# ── Integration: num_similar == len(df) - 1 (max possible) ───────────────────

def test_num_similar_max_returns_all_others(patched, tiny_df):
    """Requesting N-1 results returns every product except the query."""
    n = len(tiny_df)
    results = find_similar_products("prod_000", n - 1)
    assert len(results) == n - 1
    assert "prod_000" not in results
    assert set(results) == set(tiny_df["uniq_id"].tolist()) - {"prod_000"}


# ── Integration: text similarity drives ranking ───────────────────────────────

def test_text_identical_products_rank_highest(settings, patched_df):
    """Two products with the same product_name should be most similar to each other."""
    df = pd.DataFrame(
        {
            "uniq_id": ["prod_000", "prod_001", "prod_002", "prod_003", "prod_004"],
            "product_name": [
                "Blue Denim Jacket",
                "Blue Denim Jacket",   # identical — should rank first
                "Red Floral Dress",
                "White Sneakers",
                "Black Leather Belt",
            ],
            "brand": ["BrandA"] * 5,
            "sales_price": [1000.0] * 5,
            "rating": [4.0] * 5,
            "delivery_type": ["Standard"] * 5,
            "amazon_prime__y_or_n": ["N"] * 5,
            "best_seller_tag__y_or_n": ["N"] * 5,
            "image_urls__small": [None] * 5,
        }
    )
    with patched_df(df):
        results = find_similar_products("prod_000", 1)
    assert results[0] == "prod_001", (
        f"Expected 'prod_001' (identical text) to rank first, got {results[0]!r}"
    )
