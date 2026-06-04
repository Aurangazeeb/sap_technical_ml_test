"""Multi-backend regression tests.

Verifies that all VectorIndex backends (FAISS HNSW, ScaNN, TurboQuant) and
the TwoStageRetriever maintain recall tolerance vs brute-force cosine and
that switching backends is a zero-code-change operation (config only).

All Part-1 and Part-2 tests must continue passing alongside these.
"""
from __future__ import annotations

import numpy as np
import pytest

from sap_cxii_tech_ex_01.vector_index import FAISSHNSWIndex
from sap_cxii_tech_ex_01.backends.scann_index import ScaNNIndex
from sap_cxii_tech_ex_01.backends.turboquant_index import TurboQuantIndex
from sap_cxii_tech_ex_01.retrieval import TwoStageRetriever


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def structured_vectors() -> np.ndarray:
    """(1000, 128) float32 with low-rank structure — realistic embeddings."""
    rng = np.random.default_rng(99)
    latent = rng.standard_normal((1000, 32)).astype(np.float32)
    proj = rng.standard_normal((32, 128)).astype(np.float32)
    vecs = latent @ proj
    vecs += 0.05 * rng.standard_normal(vecs.shape).astype(np.float32)
    return vecs


@pytest.fixture
def normed_vectors(structured_vectors: np.ndarray) -> np.ndarray:
    """L2-normalized vectors for backends that use dot-product internally."""
    norms = np.linalg.norm(structured_vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (structured_vectors / norms).astype(np.float32)


def _cosine_brute_force_top_k(
    vectors: np.ndarray, query_idx: int, k: int
) -> set[int]:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    normed = vectors / norms
    scores = normed @ normed[query_idx]
    scores[query_idx] = -np.inf
    return set(np.argsort(scores)[::-1][:k].tolist())


def _measure_recall(
    index, vectors: np.ndarray, k: int = 10, n_queries: int = 50
) -> float:
    hits = 0
    total = 0
    for i in range(n_queries):
        bf = _cosine_brute_force_top_k(vectors, i, k)
        ann_indices, _ = index.query(vectors[i], k=k + 1)
        ann_set = set(int(j) for j in ann_indices if j != i)
        ann_set = set(list(ann_set)[:k])
        hits += len(ann_set & bf)
        total += k
    return hits / total if total > 0 else 0.0


# ── FAISS HNSW pipeline regression ───────────────────────────────────────────

def test_faiss_hnsw_recall_tolerance(normed_vectors: np.ndarray) -> None:
    """FAISS HNSW recall@10 must be ≥ 0.95 vs brute-force cosine."""
    idx = FAISSHNSWIndex()
    idx.build(normed_vectors)
    recall = _measure_recall(idx, normed_vectors)
    assert recall >= 0.95, f"FAISS HNSW recall@10 = {recall:.3f} < 0.95"


# ── ScaNN pipeline regression ─────────────────────────────────────────────────

def test_scann_recall_tolerance(normed_vectors: np.ndarray) -> None:
    """ScaNN recall@10 must be ≥ 0.85 vs brute-force cosine."""
    idx = ScaNNIndex(num_leaves_to_search_ratio=0.5)
    idx.build(normed_vectors)
    recall = _measure_recall(idx, normed_vectors)
    assert recall >= 0.85, f"ScaNN recall@10 = {recall:.3f} < 0.85"


# ── TurboQuant 4-bit pipeline regression ──────────────────────────────────────

def test_turboquant_4bit_recall_tolerance(structured_vectors: np.ndarray) -> None:
    """TurboQuant 4-bit recall@10 must be ≥ 0.93 vs brute-force cosine."""
    idx = TurboQuantIndex(bit_width=4)
    idx.build(structured_vectors)
    recall = _measure_recall(idx, structured_vectors)
    assert recall >= 0.93, f"TurboQuant 4-bit recall@10 = {recall:.3f} < 0.93"


# ── TurboQuant 2-bit pipeline regression ──────────────────────────────────────

def test_turboquant_2bit_recall_tolerance(structured_vectors: np.ndarray) -> None:
    """TurboQuant 2-bit recall@10 must be ≥ 0.88 vs brute-force cosine."""
    idx = TurboQuantIndex(bit_width=2)
    idx.build(structured_vectors)
    recall = _measure_recall(idx, structured_vectors)
    assert recall >= 0.88, f"TurboQuant 2-bit recall@10 = {recall:.3f} < 0.88"


# ── Two-Stage pipeline regression ─────────────────────────────────────────────

def test_two_stage_recall_tolerance(normed_vectors: np.ndarray) -> None:
    """Two-stage (HNSW + cosine rerank) recall@10 must be ≥ 0.98."""
    idx = FAISSHNSWIndex()
    idx.build(normed_vectors)
    ts = TwoStageRetriever(idx, normed_vectors, alpha=3)
    recall = _measure_recall(ts, normed_vectors)
    assert recall >= 0.98, f"Two-Stage recall@10 = {recall:.3f} < 0.98"


# ── Two-Stage never worse than single-pass ────────────────────────────────────

def test_two_stage_recall_geq_single_pass(normed_vectors: np.ndarray) -> None:
    """Two-stage recall must be ≥ single-pass HNSW recall (always improves)."""
    idx = FAISSHNSWIndex()
    idx.build(normed_vectors)
    sp_recall = _measure_recall(idx, normed_vectors)

    ts = TwoStageRetriever(idx, normed_vectors, alpha=3)
    ts_recall = _measure_recall(ts, normed_vectors)
    assert ts_recall >= sp_recall, (
        f"Two-stage {ts_recall:.3f} < single-pass {sp_recall:.3f}"
    )


# ── Backend swap via protocol ─────────────────────────────────────────────────

@pytest.mark.parametrize("backend_cls,kwargs", [
    (FAISSHNSWIndex, {}),
    (TurboQuantIndex, {"bit_width": 4}),
])
def test_backend_swap_produces_results(
    structured_vectors: np.ndarray,
    backend_cls: type,
    kwargs: dict,
) -> None:
    """Any VectorIndex backend must build and return valid query results."""
    idx = backend_cls(**kwargs)
    idx.build(structured_vectors)
    indices, distances = idx.query(structured_vectors[0], k=5)
    assert indices.shape == (5,)
    assert distances.shape == (5,)
    assert all(0 <= int(i) < len(structured_vectors) for i in indices)
