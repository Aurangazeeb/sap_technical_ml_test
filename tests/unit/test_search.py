"""Core search function unit-level contract tests.

Patches ``sap_cxii_tech_ex_01.search.load_dataset`` and ``get_settings``
so ``find_similar_products(product_id, num_similar)`` runs against
controlled test fixtures without touching the real dataset or .env.
"""
from __future__ import annotations

import inspect
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import sap_cxii_tech_ex_01.search as _search_mod
from sap_cxii_tech_ex_01.search import find_similar_products


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
    """Patch data-loading so find_similar_products uses test fixtures."""
    with (
        patch.object(_search_mod, "get_settings", return_value=settings),
        patch.object(_search_mod, "load_dataset", return_value=tiny_df),
    ):
        yield


# ── Public API signature compliance ──────────────────────────────────────────

def test_public_signature_matches_readme_spec():
    """find_similar_products(product_id, num_similar) — exactly two positional params."""
    sig = inspect.signature(find_similar_products)
    assert list(sig.parameters.keys()) == ["product_id", "num_similar"]


# ── Sanity ────────────────────────────────────────────────────────────────────

def test_known_product_not_in_results(patched):
    """Query product must not appear in its own results."""
    results = find_similar_products("prod_000", 3)
    assert "prod_000" not in results
    assert len(results) == 3


# ── num_similar respected ─────────────────────────────────────────────────────

@pytest.mark.parametrize("k", [1, 2, 4])
def test_num_similar_respected(patched, k):
    """Return exactly num_similar product IDs."""
    assert len(find_similar_products("prod_001", k)) == k


# ── Invalid product_id raises ValueError ─────────────────────────────────────

def test_invalid_product_id_raises_value_error(patched):
    """Unknown product_id must raise ValueError."""
    with pytest.raises(ValueError, match="prod_nonexistent"):
        find_similar_products("prod_nonexistent", 3)


# ── Determinism ───────────────────────────────────────────────────────────────

def test_results_are_deterministic(patched):
    """Same inputs must always produce the same output."""
    assert find_similar_products("prod_002", 3) == find_similar_products("prod_002", 3)


# ── Return type ───────────────────────────────────────────────────────────────

def test_returns_list_of_strings(patched):
    """Return value must be a list[str]."""
    results = find_similar_products("prod_003", 2)
    assert isinstance(results, list)
    assert all(isinstance(r, str) for r in results)


# ── No duplicates ─────────────────────────────────────────────────────────────

def test_no_duplicate_ids_in_results(patched):
    """All returned product IDs must be unique."""
    results = find_similar_products("prod_004", 5)
    assert len(results) == len(set(results))
