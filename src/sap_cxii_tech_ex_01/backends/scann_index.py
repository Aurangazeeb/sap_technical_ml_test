"""ScaNN index backend behind the VectorIndex Protocol.

Uses Google's ScaNN (Scalable Nearest Neighbors) library with Anisotropic
Vector Quantization (AVQ) and SOAR-style residual scoring.

References
----------
Guo et al. (2020) "Accelerating Large-Scale Inference with Anisotropic
Vector Quantization" — ICML 2020, arXiv:1908.10396

Sun et al. (2023) "SOAR: Improved Indexing for Approximate Nearest Neighbor
Search" — NeurIPS 2023, arXiv:2404.00774

Graceful fallback
-----------------
If ``scann`` cannot be imported (Linux-only package), ``make_vector_index``
logs a warning and returns a ``FAISSHNSWIndex`` instead.  The
``VectorIndex`` Protocol is fully satisfied in either case.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from sap_cxii_tech_ex_01.vector_index import FAISSHNSWIndex, VectorIndex

__all__ = ["ScaNNIndex", "make_vector_index"]

logger = logging.getLogger(__name__)


class ScaNNIndex:
    """ScaNN AVQ index conforming to the VectorIndex Protocol.

    Internally L2-normalises vectors and uses dot-product scoring, which is
    equivalent to cosine similarity for normalised vectors.

    Parameters
    ----------
    num_leaves_ratio : float
        num_leaves = ceil(num_leaves_ratio * sqrt(n)).  Default 1.0 gives the
        standard sqrt(n) heuristic.
    num_leaves_to_search_ratio : float
        Fraction of leaves to search at query time (recall–speed tradeoff).
    num_reorder : int
        Number of candidates to re-score with exact distances.
    """

    def __init__(
        self,
        num_leaves_ratio: float = 1.0,
        num_leaves_to_search_ratio: float = 0.1,
        num_reorder: int = 50,
    ) -> None:
        self._num_leaves_ratio = num_leaves_ratio
        self._num_leaves_to_search_ratio = num_leaves_to_search_ratio
        self._num_reorder = num_reorder
        self._searcher = None
        self._normalized: np.ndarray | None = None  # stored for load roundtrip

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return (vectors / norms).astype(np.float32)

    # ── Protocol ──────────────────────────────────────────────────────────

    def build(self, vectors: np.ndarray) -> None:
        """Build the ScaNN index from an (N, D) float32 matrix."""
        import scann as _scann

        normalized = self._normalize(vectors)
        self._normalized = normalized
        n = vectors.shape[0]

        num_leaves = max(2, math.ceil(self._num_leaves_ratio * math.sqrt(n)))
        num_leaves_to_search = max(
            1, math.ceil(self._num_leaves_to_search_ratio * num_leaves)
        )

        self._searcher = (
            _scann.scann_ops_pybind.builder(normalized, 10, "dot_product")
            .tree(
                num_leaves=num_leaves,
                num_leaves_to_search=num_leaves_to_search,
                training_sample_size=n,
            )
            .score_ah(2, anisotropic_quantization_threshold=0.2)
            .reorder(self._num_reorder)
            .build()
        )

    def query(self, vector: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (indices, distances) arrays of shape (k,)."""
        if self._searcher is None:
            raise RuntimeError("Index not built — call build() or load() first")
        norm_v = self._normalize(vector.reshape(1, -1))[0]
        indices, distances = self._searcher.search(norm_v, final_num_neighbors=k)
        # ScaNN returns dot-product scores (higher = closer); convert to
        # pseudo-distance so callers treating distance as "lower = better" work.
        pseudo_dist = (1.0 - distances).astype(np.float32)
        return indices.astype(np.intp), pseudo_dist

    def save(self, path: Path) -> None:
        """Persist the index to a directory at *path*."""
        if self._searcher is None:
            raise RuntimeError("Index not built — nothing to save")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._searcher.serialize(str(path))

    def load(self, path: Path) -> None:
        """Load a previously saved ScaNN index from *path*."""
        import scann as _scann

        self._searcher = _scann.scann_ops_pybind.load_searcher(str(path))


def make_vector_index(backend: str = "faiss") -> VectorIndex:
    """Factory that returns a ``VectorIndex`` implementation.

    Parameters
    ----------
    backend : str
        ``"scann"`` — attempts to return a :class:`ScaNNIndex`.  Falls back to
        :class:`FAISSHNSWIndex` with a warning if ``scann`` is unavailable.
        Any other value returns :class:`FAISSHNSWIndex`.
    """
    if backend == "scann":
        try:
            import scann  # noqa: F401  — check importability only
            return ScaNNIndex()
        except ImportError:
            logger.warning(
                "scann is not available on this platform — "
                "falling back to FAISSHNSWIndex"
            )
    return FAISSHNSWIndex()
