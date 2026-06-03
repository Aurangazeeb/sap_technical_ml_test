from __future__ import annotations

import pytest
import pandas as pd

from sap_cxii_tech_ex_01.config import Settings

# Minimal representative column set (subset of the 33-column dataset schema)
SCHEMA_COLUMNS = [
    "uniq_id",
    "product_name",
    "brand",
    "colour",
    "sales_price",
    "rating",
    "weight",
    "meta_keywords",
    "image_urls__small",
    "delivery_type",
    "amazon_prime__y_or_n",
    "best_seller_tag__y_or_n",
    "discount_percentage",
    "no__of_reviews",
]


# ── Settings fixture ─────────────────────────────────────────────────────────

@pytest.fixture
def settings() -> Settings:
    """Isolated Settings instance for tests — never reads from .env."""
    return Settings(
        data_path="data/test.ldjson",
        weight_sentinel=999_999_999,
        col_drop_threshold=0.60,
        row_drop_threshold=0.50,
    )


# ── Base product records ─────────────────────────────────────────────────────

@pytest.fixture
def valid_record() -> dict:
    """Fully populated product record with no missing feature values."""
    return {
        "uniq_id": "prod_001",
        "product_name": "Blue Denim Jacket",
        "brand": "Levis",
        "colour": "Blue",
        "sales_price": 1299.0,
        "rating": 4.2,
        "weight": None,            # weight is always None after sentinel cleanup
        "meta_keywords": "denim jacket blue casual wear",
        "image_urls__small": "https://example.com/img1.jpg|https://example.com/img2.jpg",
        "delivery_type": "Standard",
        "amazon_prime__y_or_n": "Y",
        "best_seller_tag__y_or_n": "N",
        "discount_percentage": None,  # always 100% NaN in dataset
        "no__of_reviews": None,       # ~88% NaN — dropped by col threshold
    }


@pytest.fixture
def missing_price_record(valid_record: dict) -> dict:
    """Record where sales_price is missing — numeric imputation required."""
    return {**valid_record, "uniq_id": "prod_002", "sales_price": None}


@pytest.fixture
def missing_brand_record(valid_record: dict) -> dict:
    """Record where brand is missing — categorical fill with 'unknown' required."""
    return {**valid_record, "uniq_id": "prod_003", "brand": None}


@pytest.fixture
def sentinel_weight_record(valid_record: dict) -> dict:
    """Record where weight holds the raw sentinel value 999999999."""
    return {**valid_record, "uniq_id": "prod_004", "weight": 999_999_999}


@pytest.fixture
def sparse_record() -> dict:
    """Row with >50% missing values — must be dropped by the preprocessor."""
    return {
        "uniq_id": "prod_sparse",
        "product_name": None,
        "brand": None,
        "colour": None,
        "sales_price": None,
        "rating": None,
        "weight": None,
        "meta_keywords": None,
        "image_urls__small": None,
        "delivery_type": None,
        "amazon_prime__y_or_n": None,
        "best_seller_tag__y_or_n": None,
        "discount_percentage": None,
        "no__of_reviews": None,
    }


# ── DataFrame fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def sample_df(
    valid_record: dict,
    missing_price_record: dict,
    missing_brand_record: dict,
    sentinel_weight_record: dict,
) -> pd.DataFrame:
    """Small DataFrame covering the main edge cases.

    Contains 10 rows — enough for StandardScaler to produce a meaningful
    mean/std check.  sales_price values span 1299–2000 to provide variance.
    """
    records = [
        valid_record,
        missing_price_record,
        missing_brand_record,
        sentinel_weight_record,
    ]
    # Extra valid records with varying prices for scaler assertions
    for i in range(5, 11):
        records.append({
            **valid_record,
            "uniq_id": f"prod_{i:03d}",
            "sales_price": float(1000 + i * 100),
        })
    return pd.DataFrame(records, columns=SCHEMA_COLUMNS)


@pytest.fixture
def sample_df_with_sparse_row(
    sample_df: pd.DataFrame,
    sparse_record: dict,
) -> pd.DataFrame:
    """sample_df plus one row that has >50% nulls (drop candidate)."""
    return pd.concat(
        [sample_df, pd.DataFrame([sparse_record], columns=SCHEMA_COLUMNS)],
        ignore_index=True,
    )


@pytest.fixture
def df_with_all_null_column(sample_df: pd.DataFrame) -> pd.DataFrame:
    """sample_df with an extra column that is 100% null (drop candidate)."""
    df = sample_df.copy()
    df["fully_missing_col"] = None
    return df


# ── Schema fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def expected_numeric_cols() -> list[str]:
    return ["sales_price", "rating"]


@pytest.fixture
def expected_categorical_cols() -> list[str]:
    return ["brand", "delivery_type", "amazon_prime__y_or_n", "best_seller_tag__y_or_n"]
