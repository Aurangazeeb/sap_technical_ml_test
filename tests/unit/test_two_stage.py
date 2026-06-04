"""Two-stage retrieval contract tests (RED).

Tests define the contract for ``TwoStageRetriever``, which wraps any
``VectorIndex`` as a first-stage ANN (over-fetch) and applies exact cosine
re-ranking as a second stage.

All tests must **fail** (RED) — no ``TwoStageRetriever`` exists until Step 14.

Contract
--------
- ``TwoStageRetriever(first_stage, vectors, alpha=3)``
  - ``first_stage``: any ``VectorIndex`` (build/query/save/load)
  - ``vectors``: (N, D) float32 full-precision embeddings for re-ranking
  - ``alpha``: int ≥ 1, over-fetch multiplier (default=3)
- ``retriever.query(vector, k)`` → ``(indices, distances)`` — same protocol as
  ``VectorIndex.query``
- At α=1 the two-stage produces identical results to calling
  ``first_stage.query`` directly (degenerate case)
- Recall@k of two-stage ≥ recall@k of single-pass ANN on same first-stage index
- Latency ≤ 2× single-pass latency
"""
from __future__ import annotations

import inspect
import time

import numpy as np
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def structured_vectors() -> np.ndarray:
    """(500, 64) float32 with low-rank structure — fast and realistic."""
    rng = np.random.default_rng(42)
    latent = rng.standard_normal((500, 16)).astype(np.float32)
    proj = rng.standard_normal((16, 64)).astype(np.float32)
    vecs = latent @ proj
    vecs += 0.05 * rng.standard_normal(vecs.shape).astype(np.float32)
    return vecs


@pytest.fixture
def faiss_index(structured_vectors: np.ndarray):
    """Built FAISSHNSWIndex over structured_vectors."""
    from sap_cxii_tech_ex_01.vector_index import FAISSHNSWIndex
    idx = FAISSHNSWIndex()
    idx.build(structured_vectors)
    return idx


@pytest.fixture
def two_stage(faiss_index, structured_vectors: np.ndarray):
    """TwoStageRetriever wrapping faiss_index with alpha=3."""
    from sap_cxii_tech_ex_01.retrieval import TwoStageRetriever
    return TwoStageRetriever(faiss_index, structured_vectors, alpha=3)


# ── Constructor & interface ───────────────────────────────────────────────────

def test_two_stage_is_importable() -> None:
    """TwoStageRetriever must be importable from retrieval module."""
    from sap_cxii_tech_ex_01.retrieval import TwoStageRetriever  # noqa: F401


def test_two_stage_constructor_signature() -> None:
    """Constructor must accept (first_stage, vectors, alpha=3)."""
    from sap_cxii_tech_ex_01.retrieval import TwoStageRetriever
    sig = inspect.signature(TwoStageRetriever.__init__)
    params = list(sig.parameters)
    assert "first_stage" in params
    assert "vectors" in params
    assert "alpha" in params
    assert sig.parameters["alpha"].default == 3


def test_two_stage_has_query_method() -> None:
    """TwoStageRetriever must expose a query(vector, k) method."""
    from sap_cxii_tech_ex_01.retrieval import TwoStageRetriever
    sig = inspect.signature(TwoStageRetriever.query)
    assert "vector" in sig.parameters
    assert "k" in sig.parameters


# ── Output shape & validity ───────────────────────────────────────────────────

def test_query_returns_two_arrays(two_stage, structured_vectors: np.ndarray) -> None:
    """query() must return (indices, distances) tuple."""
    result = two_stage.query(structured_vectors[0], k=5)
    assert isinstance(result, tuple) and len(result) == 2
    indices, distances = result
    assert indices.shape == (5,)
    assert distances.shape == (5,)


def test_query_indices_in_valid_range(
    two_stage, structured_vectors: np.ndarray
) -> None:
    """All returned indices must be valid row indices."""
    n = len(structured_vectors)
    indices, _ = two_stage.query(structured_vectors[0], k=10)
    assert all(0 <= int(i) < n for i in indices)


def test_query_distances_non_negative(
    two_stage, structured_vectors: np.ndarray
) -> None:
    """Distances (1 – cosine) must be ≥ 0."""
    _, distances = two_stage.query(structured_vectors[0], k=5)
    assert np.all(distances >= -1e-6), f"negative distances: {distances}"


# ── Alpha=1 degenerate case ───────────────────────────────────────────────────

def test_alpha_1_matches_single_pass(
    faiss_index, structured_vectors: np.ndarray
) -> None:
    """At alpha=1, two-stage results must match single-pass first-stage."""
    from sap_cxii_tech_ex_01.retrieval import TwoStageRetriever

    k = 5
    ts = TwoStageRetriever(faiss_index, structured_vectors, alpha=1)
    ts_indices, _ = ts.query(structured_vectors[5], k=k)
    # Single-pass: first_stage returns k candidates; two-stage re-ranks same set
    # The final top-k must be a subset of (or equal to) first_stage's top-k
    single_indices, _ = faiss_index.query(structured_vectors[5], k=k)
    # Sets must match since re-ranking a set of k with exact scores picks same k
    assert set(int(i) for i in ts_indices) == set(int(i) for i in single_indices), (
        f"alpha=1 two-stage {sorted(ts_indices)} != single-pass {sorted(single_indices)}"
    )


# ── Recall: two-stage ≥ single-pass ──────────────────────────────────────────

def _cosine_brute_force_top_k(
    vectors: np.ndarray, query_idx: int, k: int
) -> set[int]:
    """Brute-force cosine top-k (excluding self)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    normed = vectors / norms
    scores = normed @ normed[query_idx]
    scores[query_idx] = -np.inf
    return set(np.argsort(scores)[::-1][:k].tolist())


def test_two_stage_recall_not_worse_than_single_pass(
    faiss_index, structured_vectors: np.ndarray
) -> None:
    """Recall@k of two-stage must be ≥ recall@k of single-pass."""
    from sap_cxii_tech_ex_01.retrieval import TwoStageRetriever

    ts = TwoStageRetriever(faiss_index, structured_vectors, alpha=3)
    k = 10
    ts_hits = 0
    sp_hits = 0
    n_queries = 30
    for i in range(n_queries):
        bf = _cosine_brute_force_top_k(structured_vectors, i, k)
        ts_idx, _ = ts.query(structured_vectors[i], k=k + 1)
        sp_idx, _ = faiss_index.query(structured_vectors[i], k=k + 1)
        ts_set = set(int(j) for j in ts_idx if j != i)
        sp_set = set(int(j) for j in sp_idx if j != i)
        ts_hits += len(set(list(ts_set)[:k]) & bf)
        sp_hits += len(set(list(sp_set)[:k]) & bf)
    ts_recall = ts_hits / (n_queries * k)
    sp_recall = sp_hits / (n_queries * k)
    assert ts_recall >= sp_recall, (
        f"two-stage recall {ts_recall:.3f} < single-pass {sp_recall:.3f}"
    )


# ── Latency: two-stage ≤ 2× single-pass ──────────────────────────────────────

def test_two_stage_latency_bounded(
    two_stage, faiss_index, structured_vectors: np.ndarray
) -> None:
    """Two-stage latency must be ≤ 2× single-pass p50 latency."""
    k = 10
    n_warmup = 10
    n_measure = 100

    # warmup
    for i in range(n_warmup):
        faiss_index.query(structured_vectors[i % len(structured_vectors)], k)
        two_stage.query(structured_vectors[i % len(structured_vectors)], k)

    sp_times: list[float] = []
    ts_times: list[float] = []
    for i in range(n_measure):
        v = structured_vectors[i % len(structured_vectors)]
        t0 = time.perf_counter()
        faiss_index.query(v, k)
        sp_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        two_stage.query(v, k)
        ts_times.append(time.perf_counter() - t0)

    sp_p50 = float(np.percentile(sp_times, 50))
    ts_p50 = float(np.percentile(ts_times, 50))
    assert ts_p50 <= sp_p50 * 2.0, (
        f"two-stage p50 {ts_p50 * 1000:.3f}ms > 2× single-pass {sp_p50 * 1000:.3f}ms"
    )


# ── Determinism ───────────────────────────────────────────────────────────────

def test_two_stage_is_deterministic(
    two_stage, structured_vectors: np.ndarray
) -> None:
    """Repeated queries must return identical indices."""
    indices1, _ = two_stage.query(structured_vectors[0], k=5)
    indices2, _ = two_stage.query(structured_vectors[0], k=5)
    np.testing.assert_array_equal(indices1, indices2)
