"""Core similarity search function.

Orchestrates feature extraction, combination, and cosine similarity ranking
to find the most similar products for a given product ID.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sap_cxii_tech_ex_01.config import Settings, get_settings
from sap_cxii_tech_ex_01.features.image import ImageExtractor
from sap_cxii_tech_ex_01.features.structured import StructuredExtractor
from sap_cxii_tech_ex_01.features.text import TextExtractor
from sap_cxii_tech_ex_01.similarity import SimilarityEngine

__all__ = ["find_similar_products", "load_dataset"]


def load_dataset(settings: Settings) -> pd.DataFrame:
    """Load the product dataset from the path configured in *settings*."""
    return pd.read_json(settings.data_path, lines=True)


def find_similar_products(product_id: str, num_similar: int) -> list[str]:
    """Return the *num_similar* most similar product IDs to *product_id*.

    Data and configuration are loaded automatically from ``Settings`` / ``.env``.
    In tests, patch ``sap_cxii_tech_ex_01.search.load_dataset`` and/or
    ``sap_cxii_tech_ex_01.search.get_settings`` to inject controlled fixtures.

    Parameters
    ----------
    product_id:
        The ``uniq_id`` of the query product.
    num_similar:
        Number of similar products to return (query product excluded).

    Returns
    -------
    list[str]
        Ordered list of ``uniq_id`` values, most similar first.

    Raises
    ------
    ValueError
        If *product_id* is not present in the dataset.
    """
    settings = get_settings()
    df = load_dataset(settings)

    ids: list[str] = df["uniq_id"].tolist()
    if product_id not in ids:
        raise ValueError(
            f"product_id '{product_id}' not found in the provided DataFrame"
        )
    query_idx = ids.index(product_id)

    # ── Text features ─────────────────────────────────────────────────────────
    text_col = "product_name" if "product_name" in df.columns else "uniq_id"
    texts: list[str | None] = df[text_col].tolist()
    text_feats = TextExtractor(settings).extract(texts)

    # ── Image features — skip model load when no URLs are present ─────────────
    image_col = "image_urls__small"
    has_images = image_col in df.columns and df[image_col].notna().any()
    image_feats: np.ndarray | None = (
        ImageExtractor(settings).extract(df[image_col].tolist())
        if has_images
        else None
    )

    # ── Structured features ───────────────────────────────────────────────────
    struct_feats = StructuredExtractor(settings).fit_transform(df)

    # ── Combine & rank ────────────────────────────────────────────────────────
    engine = SimilarityEngine(settings)
    combined = engine.combine(text_feats, image_feats, struct_feats)
    scores = engine.cosine_similarity(combined)

    prices: np.ndarray | None = (
        df["sales_price"].fillna(0.0).to_numpy(dtype=np.float64)
        if "sales_price" in df.columns
        else None
    )

    indices = engine.top_k(scores, query_idx=query_idx, k=num_similar, prices=prices)
    return [ids[i] for i in indices]
