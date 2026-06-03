"""Image feature extraction using EfficientNet-B0 (torchvision).

Produces 1280-dimensional embeddings from product image URLs.
None or unreachable URLs return a zero vector. Results are cached in-memory
by URL to avoid redundant downloads and forward passes.
"""
from __future__ import annotations

import io
import urllib.request

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from sap_cxii_tech_ex_01.config import Settings

__all__ = ["ImageExtractor"]

_PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_FETCH_TIMEOUT_SECS = 10


def _load_model(model_name: str) -> nn.Module:
    """Load EfficientNet-B0 without the classifier head (penultimate layer)."""
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    # Remove classifier; keep avgpool + features → 1280-dim output
    model.classifier = nn.Identity()
    model.eval()
    return model


class ImageExtractor:
    """Extract visual embeddings from image URLs using EfficientNet-B0.

    Parameters
    ----------
    settings:
        Application settings; uses ``image_model`` and ``image_embedding_dim``.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: nn.Module = _load_model(settings.image_model)
        self._cache: dict[str, np.ndarray] = {}

    def _fetch_embedding(self, url: str) -> np.ndarray:
        """Download *url*, run through the model, return a (dim,) float32 array."""
        with urllib.request.urlopen(url, timeout=_FETCH_TIMEOUT_SECS) as resp:
            raw = resp.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        tensor = _PREPROCESS(image).unsqueeze(0)  # (1, 3, 224, 224)
        with torch.no_grad():
            embedding: torch.Tensor = self._model(tensor)
        return embedding.squeeze(0).numpy().astype(np.float32)

    def extract(self, image_urls: list[str | None]) -> np.ndarray:
        """Return an (N, image_embedding_dim) float32 array.

        Each row corresponds to the URL at the same position. None URLs and
        any fetch/decode errors produce a zero vector for that row.
        """
        dim = self._settings.image_embedding_dim
        result = np.zeros((len(image_urls), dim), dtype=np.float32)

        for i, url in enumerate(image_urls):
            if url is None:
                continue
            if url in self._cache:
                result[i] = self._cache[url]
                continue
            try:
                emb = self._fetch_embedding(url)
                self._cache[url] = emb
                result[i] = emb
            except Exception:
                # Network error, decode error, etc. → zero vector (already set)
                pass

        return result
