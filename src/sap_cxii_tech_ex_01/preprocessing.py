"""Data preprocessing pipeline for the Amazon Fashion similarity-search dataset.

Implements an sklearn-compatible ``Preprocessor`` that:

* Replaces weight sentinel values (999 999 999) and zero prices with NaN.
* Drops columns whose missing rate exceeds ``settings.col_drop_threshold``.
* Drops rows whose missing rate exceeds ``settings.row_drop_threshold``.
* Imputes numeric features with per-column medians.
* Scales numeric features with :class:`~sklearn.preprocessing.StandardScaler`.
* Fills missing categorical features with ``"unknown"`` then encodes them with
  :class:`~sklearn.preprocessing.LabelEncoder`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sap_cxii_tech_ex_01.config import Settings

__all__ = ["Preprocessor"]

# ---------------------------------------------------------------------------
# Column role definitions
# ---------------------------------------------------------------------------

#: Numeric columns to impute (median) and scale (StandardScaler).
_NUMERIC_FEATURES: frozenset[str] = frozenset({"sales_price", "rating"})

#: Categorical columns to fill ("unknown") and encode (LabelEncoder).
_CATEGORICAL_FEATURES: frozenset[str] = frozenset(
    {"brand", "delivery_type", "amazon_prime__y_or_n", "best_seller_tag__y_or_n"}
)


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------


class Preprocessor:
    """sklearn-compatible preprocessing pipeline.

    Parameters
    ----------
    settings:
        Configuration instance controlling sentinel value, drop thresholds,
        and other tunable parameters.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cols_to_drop: list[str] = []
        self._numeric_cols: list[str] = []
        self._categorical_cols: list[str] = []
        self._medians: dict[str, float] = {}
        self._scaler: StandardScaler | None = None
        self._encoders: dict[str, LabelEncoder] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """Learn preprocessing parameters from *df*.

        Parameters
        ----------
        df:
            Raw input DataFrame.

        Returns
        -------
        Preprocessor
            ``self`` (for method chaining).
        """
        work = self._clean_raw(df.copy())

        # ── Column-level drop (learn which columns are too sparse) ────────────
        miss_rates: pd.Series = work.isnull().mean()
        self._cols_to_drop = (
            miss_rates[miss_rates > self._settings.col_drop_threshold].index.tolist()
        )
        work = work.drop(columns=self._cols_to_drop)

        # ── Row-level drop (remove rows that are mostly empty) ────────────────
        row_miss: pd.Series = work.isnull().mean(axis=1)
        work = work.loc[row_miss <= self._settings.row_drop_threshold].copy()

        # ── Identify active feature columns ───────────────────────────────────
        self._numeric_cols = sorted(
            col for col in _NUMERIC_FEATURES if col in work.columns
        )
        self._categorical_cols = sorted(
            col for col in _CATEGORICAL_FEATURES if col in work.columns
        )

        # ── Learn per-column medians for numeric imputation ───────────────────
        for col in self._numeric_cols:
            self._medians[col] = float(work[col].median())

        # ── Fit StandardScaler on imputed numeric data ────────────────────────
        if self._numeric_cols:
            num_data = work[self._numeric_cols].copy()
            for col in self._numeric_cols:
                num_data[col] = num_data[col].fillna(self._medians[col])
            self._scaler = StandardScaler()
            self._scaler.fit(num_data.values.astype(np.float64))

        # ── Fit LabelEncoders; guarantee "unknown" is always a known class ────
        for col in self._categorical_cols:
            filled = work[col].fillna("unknown").astype(str)
            all_classes: list[str] = sorted(set(filled.tolist()) | {"unknown"})
            le = LabelEncoder()
            le.fit(all_classes)
            self._encoders[col] = le

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted preprocessing parameters to *df*.

        Parameters
        ----------
        df:
            Raw input DataFrame (same schema as the data passed to :meth:`fit`).

        Returns
        -------
        pd.DataFrame
            Cleaned, imputed, scaled, and encoded DataFrame.
        """
        out = self._clean_raw(df.copy())

        # ── Drop the columns learned during fit ───────────────────────────────
        cols_present = [c for c in self._cols_to_drop if c in out.columns]
        out = out.drop(columns=cols_present)

        # ── Drop rows that are too sparse ─────────────────────────────────────
        row_miss: pd.Series = out.isnull().mean(axis=1)
        out = (
            out.loc[row_miss <= self._settings.row_drop_threshold]
            .copy()
            .reset_index(drop=True)
        )

        # ── Numeric imputation (median) then cast to float64 ──────────────────
        for col in self._numeric_cols:
            if col in out.columns:
                out[col] = out[col].fillna(self._medians[col]).astype(np.float64)

        # ── StandardScaler ────────────────────────────────────────────────────
        num_present = [c for c in self._numeric_cols if c in out.columns]
        if num_present and self._scaler is not None:
            scaled = self._scaler.transform(
                out[num_present].values.astype(np.float64)
            )
            out[num_present] = scaled.astype(np.float64)

        # ── Categorical fill → LabelEncode → int32 ────────────────────────────
        for col in self._categorical_cols:
            if col in out.columns:
                filled = out[col].fillna("unknown").astype(str)
                le = self._encoders[col]
                known: set[str] = set(le.classes_)
                filled = filled.apply(lambda x: x if x in known else "unknown")
                out[col] = le.transform(filled).astype("int32")

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit to *df* and transform it in one call.

        Equivalent to ``self.fit(df).transform(df)``.

        Parameters
        ----------
        df:
            Raw input DataFrame.

        Returns
        -------
        pd.DataFrame
            Cleaned, imputed, scaled, and encoded DataFrame.
        """
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clean_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace weight sentinel and zero prices with NaN in-place on *df*.

        Parameters
        ----------
        df:
            A copy of the raw input DataFrame.

        Returns
        -------
        pd.DataFrame
            The same DataFrame with sentinel and zero-price values replaced.
        """
        if "weight" in df.columns:
            df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
            df.loc[
                df["weight"] == self._settings.weight_sentinel, "weight"
            ] = np.nan

        for col in ("sales_price", "rating", "discount_percentage"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "sales_price" in df.columns:
            df.loc[df["sales_price"] == 0.0, "sales_price"] = np.nan

        return df
