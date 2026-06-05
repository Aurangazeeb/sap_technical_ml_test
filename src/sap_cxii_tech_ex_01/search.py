"""Core similarity search function.

Orchestrates feature extraction, combination, and cosine similarity ranking
to find the most similar products for a given product ID.

Feature extraction is expensive (~90 s on 30 k products) so it is performed
**once** at startup via :func:`build_search_state`.  The per-request function
:func:`find_similar_products` does only a fast vector lookup + dot product.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sap_cxii_tech_ex_01.config import Settings, get_settings
from sap_cxii_tech_ex_01.features.image import ImageExtractor
from sap_cxii_tech_ex_01.features.structured import StructuredExtractor
from sap_cxii_tech_ex_01.features.text import TextExtractor
from sap_cxii_tech_ex_01.similarity import SimilarityEngine

__all__ = ["SearchState", "build_search_state", "find_similar_products", "load_dataset", "set_search_state"]

logger = logging.getLogger(__name__)

# Module-level singleton — set once at startup via set_search_state().
_search_state: SearchState | None = None


def set_search_state(state: SearchState) -> None:
    """Install the pre-computed search state for query-time use."""
    global _search_state  # noqa: PLW0603
    _search_state = state


@dataclass(frozen=True)
class SearchState:
    """Pre-computed artefacts needed for fast query-time similarity search."""

    ids: list[str]
    normalized: np.ndarray  # (N, D) L2-normalised combined features
    prices: np.ndarray | None  # (N,) or None


def load_dataset(settings: Settings) -> pd.DataFrame:
    """Load the product dataset from the path configured in *settings*."""
    return pd.read_json(settings.data_path, lines=True)


def build_search_state(df: pd.DataFrame, settings: Settings) -> SearchState:
    """Extract & combine all features once; return an immutable SearchState.

    This is meant to be called **once** at application startup.
    """
    ids: list[str] = df["uniq_id"].tolist()

    # ── Text features ─────────────────────────────────────────────────────────
    text_col = "product_name" if "product_name" in df.columns else "uniq_id"
    texts: list[str | None] = df[text_col].tolist()
    logger.info("Extracting text features for %d products…", len(texts))
    text_feats = TextExtractor(settings).extract(texts)
    logger.info("Text features: shape %s", text_feats.shape)

    # ── Image features — probe a small sample before committing ───────────────
    image_col = "image_urls__small"
    has_images = (
        not settings.skip_images
        and image_col in df.columns
        and df[image_col].notna().any()
    )
    image_feats: np.ndarray | None = None
    if settings.skip_images:
        logger.info("Image extraction skipped (SAP_SKIP_IMAGES=true).")
    elif has_images:
        urls = df[image_col].tolist()
        image_feats = _try_extract_images(urls, settings)

    # ── Structured features ───────────────────────────────────────────────────
    logger.info("Extracting structured features…")
    struct_feats = StructuredExtractor(settings).fit_transform(df)
    logger.info("Structured features: shape %s", struct_feats.shape)

    # ── Combine ───────────────────────────────────────────────────────────────
    engine = SimilarityEngine(settings)
    combined = engine.combine(text_feats, image_feats, struct_feats)
    logger.info("Combined features: shape %s", combined.shape)

    # Pre-normalise so query-time similarity is a single matrix-vector product.
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    normalized = (combined / norms).astype(np.float32)

    prices: np.ndarray | None = (
        df["sales_price"].fillna(0.0).to_numpy(dtype=np.float64)
        if "sales_price" in df.columns
        else None
    )

    logger.info("Search state ready — %d products indexed.", len(ids))
    return SearchState(ids=ids, normalized=normalized, prices=prices)


def _try_extract_images(
    urls: list[str | None],
    settings: Settings,
    probe_size: int = 5,
) -> np.ndarray | None:
    """Attempt image extraction; skip gracefully if URLs are unreachable.

    Probes *probe_size* non-null URLs first. If none succeed, the entire
    image modality is dropped and a warning is logged.
    """
    non_null = [u for u in urls if u is not None and str(u).strip()]
    if not non_null:
        return None

    extractor = ImageExtractor(settings)
    dim = settings.image_embedding_dim

    # Probe a small sample to see if URLs are reachable.
    probe = non_null[:probe_size]
    logger.info("Probing %d image URLs to check reachability…", len(probe))
    probe_result = extractor.extract(probe)
    success_count = int(np.any(probe_result != 0.0, axis=1).sum())

    if success_count == 0:
        logger.warning(
            "All %d probed image URLs returned zero vectors (unreachable / "
            "expired links). Skipping image features entirely — using "
            "text + structured fallback weights.",
            len(probe),
        )
        return None

    logger.info(
        "%d/%d probe URLs succeeded. Extracting image features for %d products…",
        success_count,
        len(probe),
        len(urls),
    )
    return extractor.extract(urls)


def find_similar_products(
    product_id: str,
    num_similar: int,
) -> list[str]:
    """Return the *num_similar* most similar product IDs to *product_id*.

    Uses the pre-computed search state installed at startup via
    :func:`set_search_state` — no heavy extraction per request.

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
    RuntimeError
        If called before :func:`set_search_state`.
    """
    if _search_state is None:
        raise RuntimeError(
            "Search state not initialised. Call set_search_state() first."
        )
    state = _search_state
    ids = state.ids
    if product_id not in ids:
        raise ValueError(
            f"product_id '{product_id}' not found in the provided DataFrame"
        )
    query_idx = ids.index(product_id)

    # Single row × matrix product → (N,) cosine similarities.  O(N·D).
    scores = state.normalized[query_idx] @ state.normalized.T
    scores = np.clip(scores, -1.0, 1.0)

    # Exclude query product, rank descending.
    order = np.argsort(-scores)
    ranked = [int(i) for i in order if i != query_idx]

    if state.prices is not None:
        # Stable tie-break by ascending price.
        ranked.sort(key=lambda i: (-float(scores[i]), float(state.prices[i])))

    return [ids[i] for i in ranked[:num_similar]]
