"""ScaNN index contract tests (RED).

Tests define the contract for ``ScaNNIndex`` behind the ``VectorIndex`` Protocol.
All tests must fail until Step 10 implements ``ScaNNIndex``.

Contract mirrors ``test_vector_index.py`` with ScaNN-specific additions:
  - Graceful fallback to FAISSHNSWIndex when ``scann`` is unavailable
  - recall@k ≥ 0.95 vs brute force L2 (same bar as HNSW)
  - ``make_vector_index("scann")`` factory function
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def random_vectors() -> np.ndarray:
    """(1000, 64) float32 matrix with low-rank cluster structure.

    Partition-based ANN (ScaNN) achieves meaningful recall only when vectors
    have cluster structure — random isotropic Gaussian is adversarial for
    k-means partitioning because all points are equidistant in cosine space.
    """
    rng = np.random.default_rng(42)
    latent = rng.standard_normal((1000, 16)).astype(np.float32)
    proj = rng.standard_normal((16, 64)).astype(np.float32)
    vecs = latent @ proj
    vecs += 0.05 * rng.standard_normal(vecs.shape).astype(np.float32)
    return vecs


@pytest.fixture
def pca_reduced_vectors() -> np.ndarray:
    """(500, 128) float32 — simulates PCA-reduced combined embeddings."""
    rng = np.random.default_rng(7)
    latent = rng.standard_normal((500, 50)).astype(np.float32)
    proj = rng.standard_normal((50, 128)).astype(np.float32)
    vecs = latent @ proj
    vecs += 0.05 * rng.standard_normal(vecs.shape).astype(np.float32)
    return vecs


@pytest.fixture
def scann_index():
    """Import and construct a ScaNNIndex (will fail until Step 10)."""
    from sap_cxii_tech_ex_01.backends.scann_index import ScaNNIndex
    # Search 50% of leaves for high recall in tests
    return ScaNNIndex(num_leaves_to_search_ratio=0.5, num_reorder=100)


@pytest.fixture
def built_scann_index(scann_index, random_vectors):
    """ScaNNIndex built on random_vectors."""
    scann_index.build(random_vectors)
    return scann_index


# ── Protocol: method signatures ──────────────────────────────────────────────

def test_scann_has_build_method(scann_index) -> None:
    assert hasattr(scann_index, "build")
    sig = inspect.signature(scann_index.build)
    assert "vectors" in sig.parameters


def test_scann_has_query_method(scann_index) -> None:
    assert hasattr(scann_index, "query")
    sig = inspect.signature(scann_index.query)
    assert "vector" in sig.parameters
    assert "k" in sig.parameters


def test_scann_has_save_method(scann_index) -> None:
    assert hasattr(scann_index, "save")
    sig = inspect.signature(scann_index.save)
    assert "path" in sig.parameters


def test_scann_has_load_method(scann_index) -> None:
    assert hasattr(scann_index, "load")
    sig = inspect.signature(scann_index.load)
    assert "path" in sig.parameters


# ── Query output shape & types ───────────────────────────────────────────────

def test_scann_query_returns_two_arrays(built_scann_index, random_vectors) -> None:
    indices, distances = built_scann_index.query(random_vectors[0], k=5)
    assert indices.shape == (5,)
    assert distances.shape == (5,)


def test_scann_query_indices_in_valid_range(built_scann_index, random_vectors) -> None:
    n = len(random_vectors)
    indices, _ = built_scann_index.query(random_vectors[0], k=10)
    assert all(0 <= int(i) < n for i in indices)


def test_scann_query_distances_non_negative(built_scann_index, random_vectors) -> None:
    _, distances = built_scann_index.query(random_vectors[0], k=10)
    assert np.all(distances >= 0), "All distances must be ≥ 0"


# ── Recall@k ─────────────────────────────────────────────────────────────────

def _brute_force_top_k(vectors: np.ndarray, query_idx: int, k: int) -> set[int]:
    """Brute-force cosine top-k (matching ScaNN's dot-product metric)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=False)
    norms = np.where(norms == 0.0, 1.0, norms)
    normed = vectors / norms[:, None]
    scores = normed @ normed[query_idx]
    scores[query_idx] = -np.inf
    return set(np.argsort(scores)[::-1][:k].tolist())


def test_scann_recall_at_k_above_threshold(built_scann_index, random_vectors) -> None:
    """ScaNN recall@10 must be ≥ 0.95 vs brute-force L2."""
    k = 10
    hits = 0
    total = 0
    for i in range(min(50, len(random_vectors))):
        bf = _brute_force_top_k(random_vectors, i, k)
        ann_indices, _ = built_scann_index.query(random_vectors[i], k=k + 1)
        ann_set = set(int(j) for j in ann_indices if j != i)
        ann_set = set(list(ann_set)[:k])
        hits += len(ann_set & bf)
        total += k
    recall = hits / total if total > 0 else 0.0
    assert recall >= 0.95, f"ScaNN recall@{k} = {recall:.3f} < 0.95"


# ── Works with PCA-reduced embeddings ────────────────────────────────────────

def test_scann_works_with_pca_reduced_vectors(pca_reduced_vectors) -> None:
    """ScaNN must accept 128-dim PCA-reduced vectors without error."""
    from sap_cxii_tech_ex_01.backends.scann_index import ScaNNIndex
    idx = ScaNNIndex()
    idx.build(pca_reduced_vectors)
    indices, distances = idx.query(pca_reduced_vectors[0], k=10)
    assert len(indices) == 10


# ── Serialization roundtrip ──────────────────────────────────────────────────

def test_scann_save_load_roundtrip(built_scann_index, random_vectors, tmp_path: Path) -> None:
    """save() then load() must produce identical query results."""
    from sap_cxii_tech_ex_01.backends.scann_index import ScaNNIndex
    query = random_vectors[0]
    k = 5
    indices_before, dists_before = built_scann_index.query(query, k=k)

    save_path = tmp_path / "scann_index"
    built_scann_index.save(save_path)

    loaded = ScaNNIndex()
    loaded.load(save_path)
    indices_after, dists_after = loaded.query(query, k=k)

    np.testing.assert_array_equal(indices_before, indices_after)


# ── Factory function & graceful fallback ─────────────────────────────────────

def test_make_vector_index_returns_scann_when_available() -> None:
    """make_vector_index('scann') must return a ScaNNIndex when scann is installed."""
    from sap_cxii_tech_ex_01.backends.scann_index import make_vector_index
    idx = make_vector_index("scann")
    # Must conform to the VectorIndex Protocol
    assert hasattr(idx, "build")
    assert hasattr(idx, "query")
    assert hasattr(idx, "save")
    assert hasattr(idx, "load")


def test_make_vector_index_falls_back_to_faiss_when_scann_unavailable() -> None:
    """make_vector_index('scann') must return FAISSHNSWIndex if scann cannot be imported."""
    from sap_cxii_tech_ex_01.vector_index import FAISSHNSWIndex

    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__  # type: ignore[attr-defined]

    def _mock_import(name, *args, **kwargs):
        if name == "scann":
            raise ImportError("scann not available (mocked)")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_mock_import):
        # Re-import to trigger the patched import path
        if "sap_cxii_tech_ex_01.backends.scann_index" in sys.modules:
            del sys.modules["sap_cxii_tech_ex_01.backends.scann_index"]
        from sap_cxii_tech_ex_01.backends.scann_index import make_vector_index
        idx = make_vector_index("scann")

    assert isinstance(idx, FAISSHNSWIndex), (
        f"Expected FAISSHNSWIndex fallback, got {type(idx).__name__}"
    )
