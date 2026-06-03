"""Similarity engine contract tests — RED phase.

sap_cxii_tech_ex_01.similarity does not yet exist; every test here is
expected to FAIL (ImportError / AttributeError) until Step 9 (GREEN).
"""
from __future__ import annotations

import numpy as np
import pytest

from sap_cxii_tech_ex_01.similarity import SimilarityEngine  # noqa: F401 — RED


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _rng_matrix(rows: int, cols: int, seed: int = 42) -> np.ndarray:
    """Return a (rows, cols) float32 matrix with reproducible random values."""
    rng = np.random.default_rng(seed)
    return rng.random((rows, cols), dtype=np.float32)


@pytest.fixture
def engine(settings):
    return SimilarityEngine(settings)


@pytest.fixture
def text_feats(settings):
    return _rng_matrix(5, settings.text_embedding_dim)


@pytest.fixture
def image_feats(settings):
    return _rng_matrix(5, settings.image_embedding_dim)


@pytest.fixture
def struct_feats():
    return _rng_matrix(5, 6).astype(np.float64)


# ── Output shape ──────────────────────────────────────────────────────────────

def test_combined_shape_with_image(engine, text_feats, image_feats, struct_feats, settings):
    """Weighted concat embedding must have shape (N, text+image+struct dims)."""
    combined = engine.combine(text_feats, image_feats, struct_feats)
    expected_cols = (
        settings.text_embedding_dim
        + settings.image_embedding_dim
        + struct_feats.shape[1]
    )
    assert combined.shape == (5, expected_cols)


def test_combined_dtype_float32(engine, text_feats, image_feats, struct_feats):
    """Combined embedding must be float32."""
    combined = engine.combine(text_feats, image_feats, struct_feats)
    assert combined.dtype == np.float32


# ── Weight redistribution ─────────────────────────────────────────────────────

def test_combine_without_image_uses_fallback_weights(engine, text_feats, struct_feats, settings):
    """When image_feats is None, text/structured weights must be the fallback values."""
    combined_with = engine.combine(text_feats, np.zeros_like(
        np.zeros((text_feats.shape[0], settings.image_embedding_dim), dtype=np.float32)
    ), struct_feats)
    combined_without = engine.combine(text_feats, None, struct_feats)
    # Without image: output excludes the image slot entirely
    expected_cols_no_image = settings.text_embedding_dim + struct_feats.shape[1]
    assert combined_without.shape == (text_feats.shape[0], expected_cols_no_image)


# ── Cosine similarity ─────────────────────────────────────────────────────────

def test_cosine_similarity_range(engine, text_feats, image_feats, struct_feats):
    """Pairwise cosine similarity scores must lie in [-1, 1]."""
    combined = engine.combine(text_feats, image_feats, struct_feats)
    scores = engine.cosine_similarity(combined)
    assert scores.shape == (5, 5)
    assert np.all(scores >= -1.0 - 1e-6)
    assert np.all(scores <= 1.0 + 1e-6)


def test_cosine_similarity_diagonal_is_one(engine, text_feats, image_feats, struct_feats):
    """Each vector must have cosine similarity 1.0 with itself."""
    combined = engine.combine(text_feats, image_feats, struct_feats)
    scores = engine.cosine_similarity(combined)
    np.testing.assert_allclose(np.diag(scores), 1.0, atol=1e-5)


def test_cosine_similarity_is_symmetric(engine, text_feats, image_feats, struct_feats):
    """Cosine similarity matrix must be symmetric."""
    combined = engine.combine(text_feats, image_feats, struct_feats)
    scores = engine.cosine_similarity(combined)
    np.testing.assert_allclose(scores, scores.T, atol=1e-5)


# ── Top-k ranking ──────────────────────────────────────────────────────────────

def test_top_k_excludes_query_itself(engine, text_feats, image_feats, struct_feats):
    """top_k(query_idx=0, k=2) must not include index 0 in returned indices."""
    combined = engine.combine(text_feats, image_feats, struct_feats)
    scores = engine.cosine_similarity(combined)
    indices = engine.top_k(scores, query_idx=0, k=2)
    assert 0 not in indices
    assert len(indices) == 2


def test_top_k_respects_k(engine, text_feats, image_feats, struct_feats):
    """top_k must return exactly k indices."""
    combined = engine.combine(text_feats, image_feats, struct_feats)
    scores = engine.cosine_similarity(combined)
    for k in (1, 2, 3, 4):
        indices = engine.top_k(scores, query_idx=0, k=k)
        assert len(indices) == k


def test_top_k_results_are_deterministic(engine, text_feats, image_feats, struct_feats):
    """Calling top_k twice with same inputs must return the same result."""
    combined = engine.combine(text_feats, image_feats, struct_feats)
    scores = engine.cosine_similarity(combined)
    assert engine.top_k(scores, query_idx=0, k=3) == engine.top_k(scores, query_idx=0, k=3)


# ── Tie-breaking ──────────────────────────────────────────────────────────────

def test_tie_breaking_by_price_ascending(engine, settings):
    """When similarity scores tie, cheaper items must rank first."""
    # 3 items with identical embeddings (all cosine sim = 1.0 for any pair)
    identical = np.ones((4, 4), dtype=np.float32)
    scores = engine.cosine_similarity(identical)
    prices = np.array([500.0, 200.0, 800.0, 100.0])
    # query_idx=0; expected order by ascending price (excluding idx 0): idx 3 (100), idx 1 (200), idx 2 (800)
    indices = engine.top_k(scores, query_idx=0, k=3, prices=prices)
    assert list(indices) == [3, 1, 2]
