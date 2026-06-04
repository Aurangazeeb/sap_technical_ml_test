"""VectorIndex Protocol and FAISS HNSW implementation.

Defines a minimal ``VectorIndex`` Protocol (build, query, save, load) so
the ANN backend is swappable without touching calling code.

Reference
---------
Malkov & Yashunin (2018) "Efficient and robust approximate nearest neighbor
using Hierarchical Navigable Small World graphs" — arXiv:1603.09320
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import faiss
import numpy as np

__all__ = ["VectorIndex", "FAISSHNSWIndex"]


@runtime_checkable
class VectorIndex(Protocol):
    """Minimal interface for approximate nearest-neighbour backends."""

    def build(self, vectors: np.ndarray) -> None:
        """Build the index from an (N, D) float32 matrix."""
        ...

    def query(self, vector: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (indices, distances) arrays of shape (k,)."""
        ...

    def save(self, path: Path) -> None:
        """Persist the index to *path*."""
        ...

    def load(self, path: Path) -> None:
        """Load the index from *path*, replacing any current state."""
        ...


class FAISSHNSWIndex:
    """FAISS HNSW (Hierarchical Navigable Small World) index.

    Parameters
    ----------
    M : int
        Number of neighbours per layer (default 32 — good recall/speed
        tradeoff for datasets up to ~1M).
    ef_construction : int
        Search depth during build (higher → better recall, slower build).
    ef_search : int
        Search depth during query (higher → better recall, slower query).
    """

    def __init__(
        self,
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 128,
    ) -> None:
        self._M = M
        self._ef_construction = ef_construction
        self._ef_search = ef_search
        self._index: faiss.IndexHNSWFlat | None = None

    def build(self, vectors: np.ndarray) -> None:
        """Build the HNSW graph from an (N, D) float32 matrix."""
        d = vectors.shape[1]
        index = faiss.IndexHNSWFlat(d, self._M)
        index.hnsw.efConstruction = self._ef_construction
        index.hnsw.efSearch = self._ef_search
        index.add(vectors)
        self._index = index

    def query(self, vector: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the k nearest neighbours as (indices, distances)."""
        if self._index is None:
            raise RuntimeError("Index not built — call build() or load() first")
        query_vec = vector.reshape(1, -1)
        distances, indices = self._index.search(query_vec, k)
        return indices[0].astype(np.intp), distances[0].astype(np.float32)

    def save(self, path: Path) -> None:
        """Write the index to disk."""
        if self._index is None:
            raise RuntimeError("Index not built — nothing to save")
        faiss.write_index(self._index, str(path))

    def load(self, path: Path) -> None:
        """Load a previously saved index from disk."""
        self._index = faiss.read_index(str(path))
        self._index.hnsw.efSearch = self._ef_search
