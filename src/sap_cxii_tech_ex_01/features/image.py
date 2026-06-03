"""Image feature extraction — stub (implemented in Step 6)."""
from __future__ import annotations

import numpy as np

from sap_cxii_tech_ex_01.config import Settings

__all__ = ["ImageExtractor"]


def _load_model(model_name: str):
    """Load the vision model (patched in tests)."""
    raise NotImplementedError("Implemented in Step 6")


class ImageExtractor:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = _load_model(settings.image_model)

    def extract(self, image_urls: list[str | None]) -> np.ndarray:
        raise NotImplementedError("Implemented in Step 6")
