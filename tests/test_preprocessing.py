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
    """Sentinel 999_999_999 → NaN, making weight 100% missing → column dropped."""
    df = pd.DataFrame([sentinel_weight_record])
    p = Preprocessor(settings)
    result = p.fit_transform(df)
    # weight is 100% NaN after sentinel cleanup, so it is dropped by col threshold
    assert "weight" not in result.columns


# ── Zero-price handling ───────────────────────────────────────────────────────

def test_zero_sales_price_treated_as_missing(
    valid_record: dict, settings
) -> None:
    """Zero prices are coerced to NaN and then imputed with median."""
    records = [
        {**valid_record, "uniq_id": "p1", "sales_price": 800.0},
        {**valid_record, "uniq_id": "p2", "sales_price": 1200.0},
        {**valid_record, "uniq_id": "p3", "sales_price": 0.0},  # zero → NaN → imputed to median=1000
    ]
    df = pd.DataFrame(records)
    p = Preprocessor(settings)
    result = p.fit_transform(df)
    # No NaN after imputation
    assert result["sales_price"].isna().sum() == 0
    # p3 was imputed to the median (1000), which scales to 0.0 in StandardScaler space.
    # p1 (800) and p2 (1200) are equidistant from the median so they scale symmetrically.
    assert result["sales_price"].iloc[2] == pytest.approx(0.0, abs=0.01)
    assert result["sales_price"].iloc[0] == pytest.approx(-result["sales_price"].iloc[1], abs=0.01)


# ── Column-level drop ─────────────────────────────────────────────────────────

def test_columns_above_drop_threshold_are_removed(
    df_with_all_null_column: pd.DataFrame, settings
) -> None:
    """Columns with >60% missing values must be absent from output."""
    p = Preprocessor(settings)
    result = p.fit_transform(df_with_all_null_column)
    assert "fully_missing_col" not in result.columns


@pytest.mark.parametrize(
    "col",
    ["weight", "discount_percentage", "colour", "no__of_reviews"],
    ids=["weight_100pct", "discount_100pct", "colour_80pct", "reviews_88pct"],
)
def test_known_sparse_columns_are_dropped(
    sample_df: pd.DataFrame, settings, col: str
) -> None:
    """Columns identified in exploration as >60% missing must be dropped."""
    p = Preprocessor(settings)
    result = p.fit_transform(sample_df)
    assert col not in result.columns


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


@pytest.mark.parametrize(
    "col",
    ["delivery_type", "amazon_prime__y_or_n", "best_seller_tag__y_or_n"],
)
def test_binary_categoricals_survive_and_are_encoded(
    sample_df: pd.DataFrame, settings, col: str
) -> None:
    """Low-cardinality binary columns must be present and integer-encoded."""
    p = Preprocessor(settings)
    result = p.fit_transform(sample_df)
    assert col in result.columns
    assert result[col].dtype in ("int32", "int64")


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
