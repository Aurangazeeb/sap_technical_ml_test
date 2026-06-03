"""Structured feature extraction combining numeric and categorical columns.

Produces a float64 array from price, rating, brand, and delivery attributes.
Numeric columns are imputed with per-column median and scaled with
StandardScaler. Categorical columns are label-encoded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sap_cxii_tech_ex_01.config import Settings

__all__ = ["StructuredExtractor"]

# Columns to use (must be present in the DataFrame; absent ones are skipped)
_NUMERIC_FEATURES: frozenset[str] = frozenset({"sales_price", "rating"})
_CATEGORICAL_FEATURES: frozenset[str] = frozenset(
    {"brand", "delivery_type", "amazon_prime__y_or_n", "best_seller_tag__y_or_n"}
)


class StructuredExtractor:
    """Combine numeric + categorical product attributes into a feature vector.

    Parameters
    ----------
    settings:
        Application settings (currently used for future extension; reserved).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._numeric_cols: list[str] = []
        self._categorical_cols: list[str] = []
        self._medians: dict[str, float] = {}
        self._scaler: StandardScaler | None = None
        self._encoders: dict[str, LabelEncoder] = {}

    def fit(self, df: pd.DataFrame) -> "StructuredExtractor":
        """Learn imputation statistics, scaler, and encoders from *df*."""
        self._numeric_cols = sorted(
            col for col in _NUMERIC_FEATURES if col in df.columns
        )
        self._categorical_cols = sorted(
            col for col in _CATEGORICAL_FEATURES if col in df.columns
        )

        # Per-column median for numeric imputation
        for col in self._numeric_cols:
            median = df[col].median()
            self._medians[col] = float(median) if not pd.isna(median) else 0.0

        # Fit StandardScaler on imputed numeric data
        if self._numeric_cols:
            num_data = df[self._numeric_cols].copy()
            for col in self._numeric_cols:
                num_data[col] = num_data[col].fillna(self._medians[col])
            self._scaler = StandardScaler()
            self._scaler.fit(num_data.values.astype(np.float64))

        # Fit LabelEncoders; guarantee "unknown" is always a known class
        for col in self._categorical_cols:
            filled = df[col].fillna("unknown").astype(str)
            all_classes: list[str] = sorted(set(filled.tolist()) | {"unknown"})
            le = LabelEncoder()
            le.fit(all_classes)
            self._encoders[col] = le

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Return an (N, D) float64 array for the rows in *df*."""
        parts: list[np.ndarray] = []

        if self._numeric_cols and self._scaler is not None:
            num_data = df[self._numeric_cols].copy()
            for col in self._numeric_cols:
                num_data[col] = num_data[col].fillna(self._medians.get(col, 0.0))
            scaled = self._scaler.transform(num_data.values.astype(np.float64))
            parts.append(scaled)

        for col in self._categorical_cols:
            filled = df[col].fillna("unknown").astype(str)
            le = self._encoders[col]
            known: set[str] = set(le.classes_)
            mapped = filled.map(lambda x, k=known: x if x in k else "unknown")
            encoded = le.transform(mapped).astype(np.float64).reshape(-1, 1)
            parts.append(encoded)

        if not parts:
            return np.zeros((len(df), 0), dtype=np.float64)

        return np.hstack(parts)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit on *df* and return its transformed representation."""
        return self.fit(df).transform(df)
