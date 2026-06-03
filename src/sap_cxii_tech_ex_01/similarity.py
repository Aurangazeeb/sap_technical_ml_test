"""Multimodal similarity engine using weighted concatenation + cosine similarity.

Combines text, image, and structured embeddings into a single vector via
weighted scaling and concatenation, then computes pairwise cosine similarity.
"""
from __future__ import annotations

import numpy as np

from sap_cxii_tech_ex_01.config import Settings

__all__ = ["SimilarityEngine"]


class SimilarityEngine:
    """Combine multimodal features and compute cosine similarity.

    Parameters
    ----------
    settings:
        Application settings; uses weight_text, weight_image, weight_structured
        and the fallback weights.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def combine(
        self,
        text_feats: np.ndarray,
        image_feats: np.ndarray | None,
        struct_feats: np.ndarray,
    ) -> np.ndarray:
        """Produce a weighted concatenation of modality embeddings.

        Parameters
        ----------
        text_feats:
            (N, text_dim) float32 array.
        image_feats:
            (N, image_dim) float32 array, or None when images are unavailable.
            When None, fallback weights are used and image columns are excluded.
        struct_feats:
            (N, D) float64 array.

        Returns
        -------
        np.ndarray
            (N, combined_dim) float32 array.
        """
        s = self._settings
        struct_f32 = struct_feats.astype(np.float32)

        if image_feats is None:
            parts = [
                text_feats * np.float32(s.fallback_weight_text),
                struct_f32 * np.float32(s.fallback_weight_structured),
            ]
        else:
            parts = [
                text_feats * np.float32(s.weight_text),
                image_feats * np.float32(s.weight_image),
                struct_f32 * np.float32(s.weight_structured),
            ]

        return np.hstack(parts).astype(np.float32)

    def cosine_similarity(self, matrix: np.ndarray) -> np.ndarray:
        """Return the (N, N) pairwise cosine similarity matrix.

        Zero-norm rows (zero vectors) are handled gracefully — similarity to
        any other vector is 0.0.
        """
        mat = matrix.astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        normalized = mat / norms
        sim: np.ndarray = normalized @ normalized.T
        # Clamp to [-1, 1] to correct for floating-point drift
        return np.clip(sim, -1.0, 1.0).astype(np.float32)

    def top_k(
        self,
        scores: np.ndarray,
        query_idx: int,
        k: int,
        prices: np.ndarray | None = None,
    ) -> list[int]:
        """Return the indices of the *k* most similar items to *query_idx*.

        The query item itself is excluded from results.
        Tie-breaking: when *prices* is provided, ties are broken by ascending
        price; otherwise by ascending index (stable, deterministic).
        """
        n = scores.shape[0]
        candidates = [i for i in range(n) if i != query_idx]

        if prices is not None:
            candidates.sort(key=lambda i: (-float(scores[query_idx, i]), float(prices[i])))
        else:
            candidates.sort(key=lambda i: (-float(scores[query_idx, i]), i))

        return candidates[:k]
