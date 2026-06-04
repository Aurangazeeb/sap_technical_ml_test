"""Dimensionality reduction via PCA.

Wraps scikit-learn's ``PCA`` to provide a slim interface for reducing
embedding dimensionality before ANN indexing.  Supports two fitting
modes: target variance ratio or explicit target dimension.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

__all__ = ["DimensionalityReducer"]


class DimensionalityReducer:
    """PCA-based dimensionality reducer for embedding vectors.

    Usage
    -----
    >>> reducer = DimensionalityReducer()
    >>> reducer.fit(vectors, target_dim=128)
    >>> reduced = reducer.transform(vectors)
    """

    def __init__(self) -> None:
        self._pca: PCA | None = None

    def fit(
        self,
        vectors: np.ndarray,
        *,
        target_variance: float | None = None,
        target_dim: int | None = None,
    ) -> None:
        """Fit PCA on *vectors*.

        Exactly one of *target_variance* or *target_dim* must be given.

        Parameters
        ----------
        vectors : np.ndarray
            (N, D) float32 matrix of embeddings.
        target_variance : float, optional
            Fraction of variance to preserve (e.g. 0.95).
        target_dim : int, optional
            Explicit number of output dimensions.
        """
        if target_variance is not None and target_dim is not None:
            raise ValueError("Specify target_variance or target_dim, not both")
        if target_variance is None and target_dim is None:
            raise ValueError("Specify target_variance or target_dim")

        if target_variance is not None:
            self._pca = PCA(n_components=target_variance, svd_solver="full")
        else:
            self._pca = PCA(n_components=target_dim)

        self._pca.fit(vectors)

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        """Project *vectors* into the reduced space."""
        if self._pca is None:
            raise RuntimeError("Call fit() before transform()")
        return self._pca.transform(vectors).astype(np.float32)

    def fit_transform(
        self,
        vectors: np.ndarray,
        *,
        target_variance: float | None = None,
        target_dim: int | None = None,
    ) -> np.ndarray:
        """Convenience: fit and transform in one call."""
        self.fit(vectors, target_variance=target_variance, target_dim=target_dim)
        return self.transform(vectors)
