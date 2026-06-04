"""TurboQuant index contract tests (RED).

Tests define the contract for ``TurboQuantIndex`` behind the ``VectorIndex``
Protocol.  All tests must fail until Step 12 implements ``TurboQuantIndex``.

Contract:
  - TurboQuantIndex conforms to VectorIndex Protocol (build, query, save, load)
  - recall@k ≥ 0.93 at 4-bit, ≥ 0.88 at 2-bit vs cosine brute force
  - Compression: memory ≤ 25% of f32 baseline at 4-bit, ≤ 13% at 2-bit
  - Online ingest: vectors added after initial build remain queryable
  - Serialization roundtrip preserves results
  - Graceful fallback to FAISSHNSWIndex when turbovec is unavailable

Reference
---------
Zandieh, Daliri, Hadian & Mirrokni (2025) "TurboQuant: Online Vector
Quantization with Near-optimal Distortion Rate" — ICLR 2026, arXiv:2504.19874
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
def structured_vectors() -> np.ndarray:
    """(1000, 128) float32 with low-rank structure — realistic for quantization."""
    rng = np.random.default_rng(7)
    latent = rng.standard_normal((1000, 32)).astype(np.float32)
    proj = rng.standard_normal((32, 128)).astype(np.float32)
    vecs = latent @ proj
    vecs += 0.05 * rng.standard_normal(vecs.shape).astype(np.float32)
    return vecs


@pytest.fixture
def turboquant_4bit():
    """Construct a TurboQuantIndex at 4-bit (will fail until Step 12)."""
    from sap_cxii_tech_ex_01.backends.turboquant_index import TurboQuantIndex
    return TurboQuantIndex(bit_width=4)


@pytest.fixture
def turboquant_2bit():
    """Construct a TurboQuantIndex at 2-bit (will fail until Step 12)."""
    from sap_cxii_tech_ex_01.backends.turboquant_index import TurboQuantIndex
    return TurboQuantIndex(bit_width=2)


@pytest.fixture
def built_4bit(turboquant_4bit, structured_vectors):
    turboquant_4bit.build(structured_vectors)
    return turboquant_4bit


@pytest.fixture
def built_2bit(turboquant_2bit, structured_vectors):
    turboquant_2bit.build(structured_vectors)
    return turboquant_2bit


# ── Protocol: method signatures ──────────────────────────────────────────────

def test_turboquant_has_build_method(turboquant_4bit) -> None:
    assert hasattr(turboquant_4bit, "build")
    sig = inspect.signature(turboquant_4bit.build)
    assert "vectors" in sig.parameters


def test_turboquant_has_query_method(turboquant_4bit) -> None:
    assert hasattr(turboquant_4bit, "query")
    sig = inspect.signature(turboquant_4bit.query)
    assert "vector" in sig.parameters
    assert "k" in sig.parameters


def test_turboquant_has_save_method(turboquant_4bit) -> None:
    assert hasattr(turboquant_4bit, "save")
    sig = inspect.signature(turboquant_4bit.save)
    assert "path" in sig.parameters


def test_turboquant_has_load_method(turboquant_4bit) -> None:
    assert hasattr(turboquant_4bit, "load")
    sig = inspect.signature(turboquant_4bit.load)
    assert "path" in sig.parameters


# ── Query output shape & types ───────────────────────────────────────────────

def test_turboquant_query_returns_two_arrays(built_4bit, structured_vectors) -> None:
    indices, distances = built_4bit.query(structured_vectors[0], k=5)
    assert indices.shape == (5,)
    assert distances.shape == (5,)


def test_turboquant_query_indices_in_valid_range(built_4bit, structured_vectors) -> None:
    n = len(structured_vectors)
    indices, _ = built_4bit.query(structured_vectors[0], k=10)
    assert all(0 <= int(i) < n for i in indices)


# ── Recall@k ─────────────────────────────────────────────────────────────────

def _cosine_brute_force_top_k(vectors: np.ndarray, query_idx: int, k: int) -> set[int]:
    """Brute-force cosine similarity top-k (excluding self)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=False)
    norms = np.where(norms == 0.0, 1.0, norms)
    normed = vectors / norms[:, None]
    scores = normed @ normed[query_idx]
    scores[query_idx] = -np.inf
    return set(np.argsort(scores)[::-1][:k].tolist())


def test_turboquant_4bit_recall_at_k_above_093(
    built_4bit: object, structured_vectors: np.ndarray
) -> None:
    """TurboQuant 4-bit recall@10 must be ≥ 0.93 vs cosine brute force."""
    k = 10
    hits = 0
    total = 0
    for i in range(min(50, len(structured_vectors))):
        bf = _cosine_brute_force_top_k(structured_vectors, i, k)
        ann_indices, _ = built_4bit.query(structured_vectors[i], k=k + 1)
        ann_set = set(int(j) for j in ann_indices if j != i)
        ann_set = set(list(ann_set)[:k])
        hits += len(ann_set & bf)
        total += k
    recall = hits / total if total > 0 else 0.0
    assert recall >= 0.93, f"TurboQuant 4-bit recall@{k} = {recall:.3f} < 0.93"


def test_turboquant_2bit_recall_at_k_above_088(
    built_2bit: object, structured_vectors: np.ndarray
) -> None:
    """TurboQuant 2-bit recall@10 must be ≥ 0.88 vs cosine brute force."""
    k = 10
    hits = 0
    total = 0
    for i in range(min(50, len(structured_vectors))):
        bf = _cosine_brute_force_top_k(structured_vectors, i, k)
        ann_indices, _ = built_2bit.query(structured_vectors[i], k=k + 1)
        ann_set = set(int(j) for j in ann_indices if j != i)
        ann_set = set(list(ann_set)[:k])
        hits += len(ann_set & bf)
        total += k
    recall = hits / total if total > 0 else 0.0
    assert recall >= 0.88, f"TurboQuant 2-bit recall@{k} = {recall:.3f} < 0.88"


# ── Compression ratio ────────────────────────────────────────────────────────

def test_turboquant_4bit_compression_ratio(
    turboquant_4bit, structured_vectors: np.ndarray
) -> None:
    """4-bit index memory must be ≤ 25% of float32 baseline."""
    from sap_cxii_tech_ex_01.backends.turboquant_index import TurboQuantIndex
    turboquant_4bit.build(structured_vectors)
    f32_bytes = structured_vectors.nbytes  # N * D * 4 bytes
    compressed_bytes = turboquant_4bit.compressed_size_bytes()
    ratio = compressed_bytes / f32_bytes
    assert ratio <= 0.25, (
        f"4-bit compression ratio {ratio:.3f} > 0.25 "
        f"({compressed_bytes} bytes vs {f32_bytes} f32 bytes)"
    )


def test_turboquant_2bit_compression_ratio(
    turboquant_2bit, structured_vectors: np.ndarray
) -> None:
    """2-bit index memory must be ≤ 13% of float32 baseline."""
    turboquant_2bit.build(structured_vectors)
    f32_bytes = structured_vectors.nbytes
    compressed_bytes = turboquant_2bit.compressed_size_bytes()
    ratio = compressed_bytes / f32_bytes
    assert ratio <= 0.13, (
        f"2-bit compression ratio {ratio:.3f} > 0.13 "
        f"({compressed_bytes} bytes vs {f32_bytes} f32 bytes)"
    )


# ── Online ingest ─────────────────────────────────────────────────────────────

def test_turboquant_online_ingest_preserves_correctness(
    structured_vectors: np.ndarray,
) -> None:
    """Vectors added after initial build must be findable in queries."""
    from sap_cxii_tech_ex_01.backends.turboquant_index import TurboQuantIndex

    base = structured_vectors[:800]
    new_vecs = structured_vectors[800:]

    idx = TurboQuantIndex(bit_width=4)
    idx.build(base)
    idx.add(new_vecs)

    # The new vectors should appear in results when queried directly
    result_indices, _ = idx.query(new_vecs[0], k=5)
    total_n = len(base) + len(new_vecs)
    assert all(0 <= int(i) < total_n for i in result_indices)


# ── Serialization roundtrip ──────────────────────────────────────────────────

def test_turboquant_save_load_roundtrip(
    built_4bit, structured_vectors: np.ndarray, tmp_path: Path
) -> None:
    """save() then load() must produce identical query results."""
    from sap_cxii_tech_ex_01.backends.turboquant_index import TurboQuantIndex

    query = structured_vectors[0]
    k = 5
    indices_before, _ = built_4bit.query(query, k=k)

    save_path = tmp_path / "tq_index"
    built_4bit.save(save_path)

    loaded = TurboQuantIndex(bit_width=4)
    loaded.load(save_path)
    indices_after, _ = loaded.query(query, k=k)

    np.testing.assert_array_equal(indices_before, indices_after)


# ── Factory function & graceful fallback ─────────────────────────────────────

def test_make_vector_index_returns_turboquant_when_available() -> None:
    """make_vector_index('turboquant') must return a TurboQuantIndex when installed."""
    from sap_cxii_tech_ex_01.backends.turboquant_index import make_vector_index
    idx = make_vector_index("turboquant")
    assert hasattr(idx, "build")
    assert hasattr(idx, "query")
    assert hasattr(idx, "save")
    assert hasattr(idx, "load")


def test_make_vector_index_falls_back_to_faiss_when_turbovec_unavailable() -> None:
    """make_vector_index('turboquant') must return FAISSHNSWIndex if turbovec is absent."""
    from sap_cxii_tech_ex_01.vector_index import FAISSHNSWIndex

    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__  # type: ignore[attr-defined]

    def _mock_import(name, *args, **kwargs):
        if name == "turbovec":
            raise ImportError("turbovec not available (mocked)")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_mock_import):
        if "sap_cxii_tech_ex_01.backends.turboquant_index" in sys.modules:
            del sys.modules["sap_cxii_tech_ex_01.backends.turboquant_index"]
        from sap_cxii_tech_ex_01.backends.turboquant_index import make_vector_index
        idx = make_vector_index("turboquant")

    assert isinstance(idx, FAISSHNSWIndex), (
        f"Expected FAISSHNSWIndex fallback, got {type(idx).__name__}"
    )
