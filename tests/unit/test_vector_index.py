"""VectorIndex Protocol contract tests (RED).

Tests define the interface for any ANN backend: build(), query(), save(), load().
They use a ``FAISSHNSWIndex`` that does not exist yet — all tests must fail.

The test suite is backend-agnostic: parametrise over implementations to reuse
the same contract tests for ScaNN, TurboQuant, etc. in later steps.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def random_vectors() -> np.ndarray:
    """(200, 64) float32 matrix — small but enough for recall tests."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((200, 64)).astype(np.float32)


@pytest.fixture
def faiss_index():
    """Import and construct a FAISSHNSWIndex (will fail until Step 3)."""
    from sap_cxii_tech_ex_01.vector_index import FAISSHNSWIndex
    return FAISSHNSWIndex()


@pytest.fixture
def built_index(faiss_index, random_vectors):
    """A FAISSHNSWIndex that has been built with random_vectors."""
    faiss_index.build(random_vectors)
    return faiss_index


# ── Protocol: method existence & signatures ───────────────────────────────────

def test_has_build_method(faiss_index) -> None:
    assert hasattr(faiss_index, "build")
    sig = inspect.signature(faiss_index.build)
    assert "vectors" in sig.parameters


def test_has_query_method(faiss_index) -> None:
    assert hasattr(faiss_index, "query")
    sig = inspect.signature(faiss_index.query)
    assert "vector" in sig.parameters
    assert "k" in sig.parameters


def test_has_save_method(faiss_index) -> None:
    assert hasattr(faiss_index, "save")
    sig = inspect.signature(faiss_index.save)
    assert "path" in sig.parameters


def test_has_load_method(faiss_index) -> None:
    assert hasattr(faiss_index, "load")
    sig = inspect.signature(faiss_index.load)
    assert "path" in sig.parameters


# ── Query returns correct shapes ──────────────────────────────────────────────

def test_query_returns_indices_and_distances(built_index, random_vectors) -> None:
    indices, distances = built_index.query(random_vectors[0], k=5)
    assert isinstance(indices, np.ndarray)
    assert isinstance(distances, np.ndarray)
    assert indices.shape == (5,)
    assert distances.shape == (5,)


def test_query_indices_are_valid(built_index, random_vectors) -> None:
    indices, _ = built_index.query(random_vectors[0], k=10)
    assert all(0 <= idx < len(random_vectors) for idx in indices)


def test_query_returns_sorted_distances(built_index, random_vectors) -> None:
    _, distances = built_index.query(random_vectors[0], k=10)
    # Distances should be non-decreasing (nearest first)
    for i in range(len(distances) - 1):
        assert distances[i] <= distances[i + 1]


# ── Recall@k ≥ 0.95 vs brute force ───────────────────────────────────────────

def _brute_force_top_k(vectors: np.ndarray, query_idx: int, k: int) -> np.ndarray:
    """Ground-truth top-k by exhaustive L2 distance."""
    query = vectors[query_idx]
    diffs = vectors - query
    dists = np.linalg.norm(diffs, axis=1)
    dists[query_idx] = np.inf  # exclude self
    return np.argsort(dists)[:k]


def test_recall_at_k_above_threshold(built_index, random_vectors) -> None:
    """HNSW recall@10 must be ≥ 0.95 vs brute force on this small dataset."""
    k = 10
    hits = 0
    total = 0
    for i in range(min(50, len(random_vectors))):
        bf_indices = set(_brute_force_top_k(random_vectors, i, k).tolist())
        # Query k+1 because FAISS returns the query vector itself (dist=0)
        ann_indices, _ = built_index.query(random_vectors[i], k=k + 1)
        ann_set = set(ann_indices.tolist()) - {i}  # exclude self
        # Take only top-k after removing self
        ann_set = set(list(ann_set)[:k])
        bf_indices.discard(i)
        hits += len(ann_set & bf_indices)
        total += len(bf_indices)
    recall = hits / total if total > 0 else 0.0
    assert recall >= 0.95, f"recall@{k} = {recall:.3f} < 0.95"


# ── Serialization roundtrip ───────────────────────────────────────────────────

def test_save_load_roundtrip(built_index, random_vectors, tmp_path: Path) -> None:
    """save() then load() must produce identical query results."""
    query = random_vectors[0]
    indices_before, dists_before = built_index.query(query, k=5)

    index_path = tmp_path / "test_index.bin"
    built_index.save(index_path)
    assert index_path.exists()

    from sap_cxii_tech_ex_01.vector_index import FAISSHNSWIndex
    loaded = FAISSHNSWIndex()
    loaded.load(index_path)

    indices_after, dists_after = loaded.query(query, k=5)
    np.testing.assert_array_equal(indices_before, indices_after)
    np.testing.assert_allclose(dists_before, dists_after, atol=1e-6)


# ── Latency bound ─────────────────────────────────────────────────────────────

def test_build_completes_in_reasonable_time(faiss_index, random_vectors) -> None:
    """Build on 200 vectors should complete in < 5 seconds."""
    import time
    t0 = time.perf_counter()
    faiss_index.build(random_vectors)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"build took {elapsed:.2f}s"


def test_query_completes_in_reasonable_time(built_index, random_vectors) -> None:
    """Single query should complete in < 100ms."""
    import time
    t0 = time.perf_counter()
    built_index.query(random_vectors[0], k=10)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.1, f"query took {elapsed * 1000:.2f}ms"
