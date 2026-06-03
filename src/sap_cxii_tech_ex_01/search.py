"""Core similarity search function.

Orchestrates feature extraction, combination, and cosine similarity ranking
to find the most similar products for a given product ID.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sap_cxii_tech_ex_01.config import Settings
from sap_cxii_tech_ex_01.features.image import ImageExtractor
from sap_cxii_tech_ex_01.features.structured import StructuredExtractor
from sap_cxii_tech_ex_01.features.text import TextExtractor
from sap_cxii_tech_ex_01.similarity import SimilarityEngine

__all__ = ["find_similar_products"]


def find_similar_products(
    product_id: str,
    df: pd.DataFrame,
    settings: Settings,
    num_similar: int = 10,
) -> list[str]:
    """Return the *num_similar* most similar product IDs to *product_id*.

    Parameters
    ----------
    product_id:
        The ``uniq_id`` of the query product.
    df:
        DataFrame containing product records (already loaded; need not be
        preprocessed — feature extractors handle missing values internally).
    settings:
        Application settings controlling model names and similarity weights.
    num_similar:
        Number of similar products to return (query product excluded).

    Returns
    -------
    list[str]
        Ordered list of ``uniq_id`` values, most similar first.

    Raises
    ------
    ValueError
        If *product_id* is not present in ``df['uniq_id']``.
    """
    ids: list[str] = df["uniq_id"].tolist()
    if product_id not in ids:
        raise ValueError(
            f"product_id '{product_id}' not found in the provided DataFrame"
        )
    query_idx = ids.index(product_id)

    # ── Text features ─────────────────────────────────────────────────────────
    text_col = "product_name" if "product_name" in df.columns else "uniq_id"
    texts: list[str | None] = df[text_col].tolist()
    text_extractor = TextExtractor(settings)
    text_feats = text_extractor.extract(texts)

    # ── Image features — skip model load when no URLs are present ─────────────
    image_col = "image_urls__small"
    has_images = image_col in df.columns and df[image_col].notna().any()
    if has_images:
        image_extractor = ImageExtractor(settings)
        image_feats: np.ndarray | None = image_extractor.extract(
            df[image_col].tolist()
        )
    else:
        image_feats = None

    # ── Structured features ───────────────────────────────────────────────────
    struct_extractor = StructuredExtractor(settings)
    struct_feats = struct_extractor.fit_transform(df)

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
