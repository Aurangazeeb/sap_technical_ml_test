"""Dimensionality reduction contract tests (RED).

Tests define the PCA reduction contract: variance preservation, target
dimensions, and recall impact. All tests must fail until Step 4 implements
the ``DimensionalityReducer`` class.
"""
from __future__ import annotations

import numpy as np
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def high_dim_text_vectors() -> np.ndarray:
    """Simulates (500, 384) text embeddings."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((500, 384)).astype(np.float32)


@pytest.fixture
def high_dim_image_vectors() -> np.ndarray:
    """Simulates (500, 1280) image embeddings."""
    rng = np.random.default_rng(99)
    return rng.standard_normal((500, 1280)).astype(np.float32)


@pytest.fixture
def reducer():
    """Import DimensionalityReducer (will fail until Step 4)."""
    from sap_cxii_tech_ex_01.dim_reduction import DimensionalityReducer
    return DimensionalityReducer()


# ── Variance preservation ─────────────────────────────────────────────────────

def test_pca_preserves_95_percent_variance(reducer, high_dim_text_vectors) -> None:
    """PCA-reduced embeddings must preserve ≥ 95% of original variance."""
    reducer.fit(high_dim_text_vectors, target_variance=0.95)
    reduced = reducer.transform(high_dim_text_vectors)

    original_var = np.var(high_dim_text_vectors, axis=0).sum()
    reduced_var = np.var(reduced, axis=0).sum()
    ratio = reduced_var / original_var
    assert ratio >= 0.95, f"variance ratio {ratio:.3f} < 0.95"


# ── Target dimensions ────────────────────────────────────────────────────────

def test_text_384_reduces_to_128(reducer, high_dim_text_vectors) -> None:
    """384-dim text embeddings should reduce to ~128 dims at 95% variance."""
    reducer.fit(high_dim_text_vectors, target_dim=128)
    reduced = reducer.transform(high_dim_text_vectors)
    assert reduced.shape == (500, 128)


def test_image_1280_reduces_to_256(reducer, high_dim_image_vectors) -> None:
    """1280-dim image embeddings should reduce to ~256 dims at 95% variance."""
    reducer.fit(high_dim_image_vectors, target_dim=256)
    reduced = reducer.transform(high_dim_image_vectors)
    assert reduced.shape == (500, 256)


# ── Output dtype ──────────────────────────────────────────────────────────────

def test_reduced_output_is_float32(reducer, high_dim_text_vectors) -> None:
    reducer.fit(high_dim_text_vectors, target_dim=128)
    reduced = reducer.transform(high_dim_text_vectors)
    assert reduced.dtype == np.float32


# ── fit_transform convenience ─────────────────────────────────────────────────

def test_fit_transform_equivalent_to_fit_then_transform(
    reducer, high_dim_text_vectors
) -> None:
    reduced_ft = reducer.fit_transform(high_dim_text_vectors, target_dim=128)

    from sap_cxii_tech_ex_01.dim_reduction import DimensionalityReducer
    r2 = DimensionalityReducer()
    r2.fit(high_dim_text_vectors, target_dim=128)
    reduced_sep = r2.transform(high_dim_text_vectors)

    np.testing.assert_allclose(reduced_ft, reduced_sep, atol=1e-5)


# ── Recall@k on reduced embeddings ───────────────────────────────────────────

def _brute_force_top_k(vectors: np.ndarray, query_idx: int, k: int) -> set[int]:
    query = vectors[query_idx]
    dists = np.linalg.norm(vectors - query, axis=1)
    dists[query_idx] = np.inf
    return set(np.argsort(dists)[:k].tolist())


def test_reduced_recall_above_093(reducer, high_dim_text_vectors) -> None:
    """Similarity search on PCA-reduced embeddings must maintain recall@10 ≥ 0.93
    vs full-dimensional brute force."""
    k = 10
    reducer.fit(high_dim_text_vectors, target_dim=128)
    reduced = reducer.transform(high_dim_text_vectors)

    hits = 0
    total = 0
    for i in range(min(50, len(high_dim_text_vectors))):
        bf_full = _brute_force_top_k(high_dim_text_vectors, i, k)
        bf_reduced = _brute_force_top_k(reduced, i, k)
        hits += len(bf_full & bf_reduced)
        total += k
    recall = hits / total if total > 0 else 0.0
    assert recall >= 0.93, f"reduced recall@{k} = {recall:.3f} < 0.93"
