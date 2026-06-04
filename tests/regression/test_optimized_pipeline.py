"""End-to-end regression tests for the optimized pipeline.

Verifies that PCA + HNSW + cache produce results within recall tolerance
of brute-force cosine similarity and that query latency stays bounded.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from sap_cxii_tech_ex_01.cache import SearchCache
from sap_cxii_tech_ex_01.dim_reduction import DimensionalityReducer
from sap_cxii_tech_ex_01.vector_index import FAISSHNSWIndex


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_vectors() -> np.ndarray:
    """Low-rank (1000, 390) matrix simulating combined text+structured features."""
    rng = np.random.default_rng(42)
    latent = rng.standard_normal((1000, 80)).astype(np.float32)
    proj = rng.standard_normal((80, 390)).astype(np.float32)
    vectors = latent @ proj
    vectors += 0.1 * rng.standard_normal(vectors.shape).astype(np.float32)
    return vectors


@pytest.fixture
def reduced_vectors(synthetic_vectors: np.ndarray) -> np.ndarray:
    reducer = DimensionalityReducer()
    return reducer.fit_transform(synthetic_vectors, target_dim=128)


@pytest.fixture
def built_index(reduced_vectors: np.ndarray) -> FAISSHNSWIndex:
    idx = FAISSHNSWIndex()
    idx.build(reduced_vectors)
    return idx


def _brute_force_cosine_top_k(
    matrix: np.ndarray, query_idx: int, k: int
) -> set[int]:
    """Return brute-force cosine-similarity top-k (excluding self)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=False)
    norms = np.where(norms == 0.0, 1.0, norms)
    normed = matrix / norms[:, None]
    scores = normed @ normed[query_idx]
    scores[query_idx] = -np.inf
    return set(np.argsort(scores)[::-1][:k].tolist())


def _brute_force_l2_top_k(
    matrix: np.ndarray, query_idx: int, k: int
) -> set[int]:
    """Return brute-force L2 top-k (excluding self)."""
    diffs = matrix - matrix[query_idx]
    dists = np.linalg.norm(diffs, axis=1)
    dists[query_idx] = np.inf
    return set(np.argsort(dists)[:k].tolist())


# ── Recall regression: optimized pipeline vs brute-force ──────────────────────

def test_pca_hnsw_recall_vs_brute_force_l2(
    built_index: FAISSHNSWIndex,
    reduced_vectors: np.ndarray,
) -> None:
    """HNSW on PCA-reduced vectors must achieve recall@10 ≥ 0.90 vs L2 brute force."""
    k = 10
    n_queries = 50
    hits = 0
    total = 0
    for i in range(n_queries):
        bf = _brute_force_l2_top_k(reduced_vectors, i, k)
        ann_indices, _ = built_index.query(reduced_vectors[i], k=k + 1)
        ann_set = set(int(j) for j in ann_indices if j != i)
        ann_set = set(list(ann_set)[:k])
        hits += len(ann_set & bf)
        total += k
    recall = hits / total
    assert recall >= 0.90, f"recall@{k} = {recall:.3f} < 0.90"


def test_pca_preserves_neighborhood_structure(
    synthetic_vectors: np.ndarray,
    reduced_vectors: np.ndarray,
) -> None:
    """PCA reduction must preserve ≥ 85% of cosine top-10 neighbors from full-dim."""
    k = 10
    n_queries = 50
    hits = 0
    total = 0
    for i in range(n_queries):
        full_top = _brute_force_cosine_top_k(synthetic_vectors, i, k)
        reduced_top = _brute_force_cosine_top_k(reduced_vectors, i, k)
        hits += len(full_top & reduced_top)
        total += k
    recall = hits / total
    assert recall >= 0.85, f"PCA neighborhood recall = {recall:.3f} < 0.85"


# ── Latency regression ───────────────────────────────────────────────────────

def test_hnsw_query_latency_under_10ms(
    built_index: FAISSHNSWIndex,
    reduced_vectors: np.ndarray,
) -> None:
    """Single HNSW query on 1k vectors must complete in < 10ms (p95)."""
    k = 10
    latencies: list[float] = []
    for i in range(100):
        t0 = time.perf_counter()
        built_index.query(reduced_vectors[i], k=k)
        latencies.append(time.perf_counter() - t0)
    p95 = float(np.percentile(latencies, 95)) * 1000
    assert p95 < 10.0, f"p95 query latency = {p95:.2f}ms > 10ms"


# ── Cache integration regression ─────────────────────────────────────────────

def test_cache_hit_faster_than_index_query(
    built_index: FAISSHNSWIndex,
    reduced_vectors: np.ndarray,
) -> None:
    """Cache hit must be at least 10x faster than an index query."""
    k = 10
    cache = SearchCache(max_size=128)

    # Prime cache
    ann_indices, _ = built_index.query(reduced_vectors[0], k=k)
    result = [str(j) for j in ann_indices]
    cache.put("query_0", k, result)

    # Time cache hit
    t0 = time.perf_counter()
    for _ in range(1000):
        cache.get("query_0", k)
    cache_time = (time.perf_counter() - t0) / 1000

    # Time index query
    t0 = time.perf_counter()
    for _ in range(100):
        built_index.query(reduced_vectors[0], k=k)
    index_time = (time.perf_counter() - t0) / 100

    assert cache_time < index_time / 10, (
        f"cache={cache_time*1e6:.1f}µs, index={index_time*1e6:.1f}µs — "
        f"cache should be >10x faster"
    )


# ── Full pipeline roundtrip ──────────────────────────────────────────────────

def test_full_pipeline_deterministic(synthetic_vectors: np.ndarray) -> None:
    """Two runs of PCA→HNSW→query must produce identical results."""
    k = 10

    results = []
    for _ in range(2):
        reducer = DimensionalityReducer()
        reduced = reducer.fit_transform(synthetic_vectors, target_dim=128)
        index = FAISSHNSWIndex()
        index.build(reduced)
        indices, _ = index.query(reduced[0], k=k)
        results.append(indices.tolist())

    assert results[0] == results[1], "Pipeline indices are not deterministic across runs"
