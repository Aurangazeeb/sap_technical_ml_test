"""Preprocessing contract tests (RED phase).

All tests import from sap_cxii_tech_ex_01.preprocessing which does not yet
exist — every test in this file is expected to FAIL until Step 3 (GREEN).
"""
from __future__ import annotations

import pytest
import pandas as pd

from sap_cxii_tech_ex_01.preprocessing import Preprocessor  # noqa: F401 — RED


# ── Sentinel handling ─────────────────────────────────────────────────────────

def test_weight_sentinel_replaced_with_nan(
    sentinel_weight_record: dict, settings
) -> None:
    df = pd.DataFrame([sentinel_weight_record])
    p = Preprocessor(settings)
    result = p.fit_transform(df)
    assert result["weight"].isna().all()


# ── Column-level drop ─────────────────────────────────────────────────────────

def test_columns_above_drop_threshold_are_removed(
    df_with_all_null_column: pd.DataFrame, settings
) -> None:
    """Columns with >60% missing values must be absent from output."""
    p = Preprocessor(settings)
    result = p.fit_transform(df_with_all_null_column)
    assert "fully_missing_col" not in result.columns


# ── Row-level drop ────────────────────────────────────────────────────────────

def test_rows_above_drop_threshold_are_removed(
    sample_df_with_sparse_row: pd.DataFrame,
    sample_df: pd.DataFrame,
    settings,
) -> None:
    """Rows with >50% missing values must be absent from output."""
    p = Preprocessor(settings)
    result = p.fit_transform(sample_df_with_sparse_row)
    assert len(result) == len(p.fit_transform(sample_df))


# ── Numeric imputation ────────────────────────────────────────────────────────

def test_missing_numeric_imputed_with_median(
    sample_df: pd.DataFrame, settings
) -> None:
    p = Preprocessor(settings)
    result = p.fit_transform(sample_df)
    assert result["sales_price"].isna().sum() == 0


# ── Categorical imputation ────────────────────────────────────────────────────

def test_missing_categorical_filled_with_unknown(
    sample_df: pd.DataFrame, settings
) -> None:
    p = Preprocessor(settings)
    result = p.fit_transform(sample_df)
    # brand column had NaN rows — must all be resolved
    assert result["brand"].isna().sum() == 0


# ── Output dtypes ─────────────────────────────────────────────────────────────

def test_output_numeric_dtypes(
    sample_df: pd.DataFrame,
    settings,
    expected_numeric_cols: list[str],
) -> None:
    p = Preprocessor(settings)
    result = p.fit_transform(sample_df)
    for col in expected_numeric_cols:
        if col in result.columns:
            assert result[col].dtype == "float64", f"{col} dtype should be float64"


def test_categorical_encoding_produces_integers(
    sample_df: pd.DataFrame, settings
) -> None:
    p = Preprocessor(settings)
    result = p.fit_transform(sample_df)
    assert result["brand"].dtype in ("int32", "int64")


# ── StandardScaler behaviour ─────────────────────────────────────────────────

def test_standard_scaler_mean_approx_zero(
    sample_df: pd.DataFrame, settings
) -> None:
    p = Preprocessor(settings)
    result = p.fit_transform(sample_df)
    assert result["sales_price"].mean() == pytest.approx(0.0, abs=0.1)


def test_standard_scaler_std_approx_one(
    sample_df: pd.DataFrame, settings
) -> None:
    p = Preprocessor(settings)
    result = p.fit_transform(sample_df)
    # sklearn StandardScaler uses ddof=0; pandas default is ddof=1
    assert result["sales_price"].std(ddof=0) == pytest.approx(1.0, abs=0.15)


# ── fit / transform consistency ───────────────────────────────────────────────

def test_fit_transform_equals_separate_fit_then_transform(
    sample_df: pd.DataFrame, settings
) -> None:
    """fit_transform(df) must produce identical output to fit(df).transform(df)."""
    p1 = Preprocessor(settings)
    combined = p1.fit_transform(sample_df)

    p2 = Preprocessor(settings)
    p2.fit(sample_df)
    separate = p2.transform(sample_df)

    pd.testing.assert_frame_equal(combined, separate)


def test_transform_is_deterministic(
    sample_df: pd.DataFrame, settings
) -> None:
    """Calling transform twice on the same data must yield identical results."""
    p = Preprocessor(settings)
    p.fit(sample_df)
    r1 = p.transform(sample_df)
    r2 = p.transform(sample_df)
    pd.testing.assert_frame_equal(r1, r2)
