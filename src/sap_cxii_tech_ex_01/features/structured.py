"""Structured feature extraction — stub (implemented in Step 7)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sap_cxii_tech_ex_01.config import Settings

__all__ = ["StructuredExtractor"]


class StructuredExtractor:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def fit(self, df: pd.DataFrame) -> "StructuredExtractor":
        raise NotImplementedError("Implemented in Step 7")

    def extract(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("Implemented in Step 7")
