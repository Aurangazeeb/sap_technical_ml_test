"""Two-stage retrieval: over-fetch ANN candidates, then exact cosine re-rank.

Architecture
------------
Stage 1 — first-stage ``VectorIndex`` retrieves ``k × alpha`` candidates using
          its internal quantized/approximate representation.
Stage 2 — exact cosine similarity on stored full-precision normalized vectors
          re-ranks the candidates and returns the true top-k.

This pattern decouples search speed (aggressive quantization in stage 1) from
accuracy (full-feature re-scoring in stage 2) and is standard practice in
production retrieval systems (Google, Pinterest, Spotify).

Usage
-----
>>> from sap_cxii_tech_ex_01.retrieval import TwoStageRetriever
>>> from sap_cxii_tech_ex_01.vector_index import FAISSHNSWIndex
>>> idx = FAISSHNSWIndex()
>>> idx.build(vectors)
>>> retriever = TwoStageRetriever(idx, vectors, alpha=3)
>>> indices, distances = retriever.query(query_vector, k=10)
"""
from __future__ import annotations

import numpy as np


class TwoStageRetriever:
    """Two-stage retrieval: ANN over-fetch + exact cosine re-rank.

    Parameters
    ----------
    first_stage:
        Any ``VectorIndex``-compatible object (``build``, ``query``,
        ``save``, ``load``).  Used as the coarse first-stage ANN.
    vectors:
        (N, D) float32 full-precision embeddings.  L2-normalized internally
        for exact cosine re-ranking.
    alpha:
        Over-fetch multiplier.  Stage 1 retrieves ``k × alpha`` candidates;
        stage 2 re-ranks them and returns the top-k.  Default=3.
        At alpha=1 the re-rank set equals the first-stage result set, so
        the output is identical to a single-pass query.
    """

    def __init__(
        self,
        first_stage: object,
        vectors: np.ndarray,
        alpha: int = 3,
    ) -> None:
        if alpha < 1:
            raise ValueError(f"alpha must be ≥ 1, got {alpha}")
        self._first_stage = first_stage
        self._alpha = alpha
        # Pre-compute L2-normalized vectors for fast exact cosine re-rank.
        vecs = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        self._normed: np.ndarray = (vecs / norms).astype(np.float32)

    def query(self, vector: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return *(indices, distances)* for the *k* nearest neighbours.

        Parameters
        ----------
        vector:
            (D,) float32 query embedding.
        k:
            Number of results to return.

        Returns
        -------
        indices:
            (k,) int array of row indices into the original ``vectors``.
        distances:
            (k,) float32 pseudo-distances: ``1 – cosine_similarity``.
        """
        fetch = max(k * self._alpha, k)
        # Stage 1: coarse ANN over-fetch
        cand_indices, _ = self._first_stage.query(vector, fetch)
        # Stage 2: exact cosine re-rank
        q = np.asarray(vector, dtype=np.float32)
        q_norm = q / max(float(np.linalg.norm(q)), 1e-10)
        exact_scores = self._normed[cand_indices] @ q_norm  # (fetch,)
        order = np.argsort(exact_scores)[::-1][:k]
        top_indices = cand_indices[order]
        distances = (1.0 - exact_scores[order]).astype(np.float32)
        return top_indices.astype(np.intp), distances
