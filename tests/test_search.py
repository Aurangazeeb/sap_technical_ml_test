"""Core search function contract tests — RED phase.

sap_cxii_tech_ex_01.search does not yet exist; every test here is
expected to FAIL (ImportError) until Step 9/10 (GREEN).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from sap_cxii_tech_ex_01.search import find_similar_products  # noqa: F401 — RED


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


# ── Sanity: product returns itself as top result ──────────────────────────────

def test_known_product_returns_itself_as_most_similar(settings, tiny_df):
    """A query product should have cosine similarity 1.0 with itself.

    find_similar_products must NOT include the query product in results,
    but the nearest result should come from the same underlying feature space.
    """
    results = find_similar_products(
        product_id="prod_000",
        df=tiny_df,
        settings=settings,
        num_similar=3,
    )
    # Query itself should not appear in results
    assert "prod_000" not in results
    assert len(results) == 3


# ── num_similar respected ─────────────────────────────────────────────────────

@pytest.mark.parametrize("k", [1, 2, 4])
def test_num_similar_respected(settings, tiny_df, k):
    """Return exactly num_similar product IDs."""
    results = find_similar_products(
        product_id="prod_001",
        df=tiny_df,
        settings=settings,
        num_similar=k,
    )
    assert len(results) == k


# ── Invalid product_id raises ValueError ─────────────────────────────────────

def test_invalid_product_id_raises_value_error(settings, tiny_df):
    """Unknown product_id must raise ValueError — not KeyError or IndexError."""
    with pytest.raises(ValueError, match="prod_nonexistent"):
        find_similar_products(
            product_id="prod_nonexistent",
            df=tiny_df,
            settings=settings,
            num_similar=3,
        )


# ── Determinism ───────────────────────────────────────────────────────────────

def test_results_are_deterministic(settings, tiny_df):
    """Two calls with identical inputs must return identical results."""
    r1 = find_similar_products("prod_002", df=tiny_df, settings=settings, num_similar=3)
    r2 = find_similar_products("prod_002", df=tiny_df, settings=settings, num_similar=3)
    assert r1 == r2


# ── Return type ───────────────────────────────────────────────────────────────

def test_returns_list_of_strings(settings, tiny_df):
    """Return value must be a list of str product IDs."""
    results = find_similar_products("prod_003", df=tiny_df, settings=settings, num_similar=2)
    assert isinstance(results, list)
    assert all(isinstance(r, str) for r in results)


# ── No duplicate IDs in results ───────────────────────────────────────────────

def test_no_duplicate_ids_in_results(settings, tiny_df):
    """All returned product IDs must be unique."""
    results = find_similar_products("prod_004", df=tiny_df, settings=settings, num_similar=5)
    assert len(results) == len(set(results))
