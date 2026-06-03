"""Text feature extraction using sentence-transformers.

Produces semantic embeddings from product text (product_name + description).
Empty or None inputs return a zero vector without invoking the model.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from sap_cxii_tech_ex_01.config import Settings

__all__ = ["TextExtractor"]


class TextExtractor:
    """Encode a list of text strings into dense embedding vectors.

    Parameters
    ----------
    settings:
        Application settings; uses ``text_model`` and ``text_embedding_dim``.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = SentenceTransformer(settings.text_model)

    def extract(self, texts: list[str | None]) -> np.ndarray:
        """Return an (N, text_embedding_dim) float32 array.

        Each row corresponds to the input text at the same position.
        Empty strings and None values produce a zero vector for that row.
        """
        dim = self._settings.text_embedding_dim
        result = np.zeros((len(texts), dim), dtype=np.float32)

        valid_indices: list[int] = []
        valid_texts: list[str] = []
        for i, text in enumerate(texts):
            if text is not None and text.strip():
                valid_indices.append(i)
                valid_texts.append(text)

        if valid_texts:
            embeddings: np.ndarray = self._model.encode(
                valid_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            for row_idx, emb in zip(valid_indices, embeddings):
                result[row_idx] = emb.astype(np.float32)

        return result
