"""Search integration tests.

Validates end-to-end pipeline behaviour via ``find_similar_products``.
Constructs ``SearchState`` via ``build_search_state`` with small test DataFrames
so no real dataset or .env is required during CI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sap_cxii_tech_ex_01.search import SearchState, build_search_state, find_similar_products


# ── Helpers & fixtures ────────────────────────────────────────────────────────

def _make_tiny_df(n: int = 5) -> pd.DataFrame:
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
def state(settings, tiny_df) -> SearchState:
    return build_search_state(tiny_df, settings)


# ── Integration: all returned IDs exist in the input DataFrame ───────────────

def test_results_are_valid_product_ids(state, tiny_df):
    valid_ids = set(tiny_df["uniq_id"].tolist())
    for pid in find_similar_products("prod_001", 4, state):
        assert pid in valid_ids, f"{pid!r} not in input DataFrame"


# ── Integration: num_similar == len(df) - 1 (max possible) ───────────────────

def test_num_similar_max_returns_all_others(state, tiny_df):
    n = len(tiny_df)
    results = find_similar_products("prod_000", n - 1, state)
    assert len(results) == n - 1
    assert "prod_000" not in results
    assert set(results) == set(tiny_df["uniq_id"].tolist()) - {"prod_000"}


# ── Integration: text similarity drives ranking ───────────────────────────────

def test_text_identical_products_rank_highest(settings):
    """Identical product names should rank first when structured features are equal."""
    df = pd.DataFrame(
        {
            "uniq_id": ["prod_000", "prod_001", "prod_002", "prod_003", "prod_004"],
            "product_name": [
                "Blue Denim Jacket",
                "Blue Denim Jacket",  # identical — should rank first
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
    state = build_search_state(df, settings)
    results = find_similar_products("prod_000", 1, state)
    assert results[0] == "prod_001", (
        f"Expected 'prod_001' (identical text) to rank first, got {results[0]!r}"
    )
