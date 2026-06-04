from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "get_settings"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="SAP_", extra="ignore"
    )

    # --- Data ---
    data_path: Path = Path(
        "data/marketing_sample_for_amazon_com-amazon_fashion_products"
        "__20200201_20200430__30k_data.ldjson"
    )

    # --- Preprocessing thresholds ---
    weight_sentinel: int = 999_999_999
    col_drop_threshold: float = 0.60
    row_drop_threshold: float = 0.50

    # --- Models ---
    text_model: str = "all-MiniLM-L6-v2"
    image_model: str = "efficientnet_b0"

    # --- Embedding dimensions ---
    text_embedding_dim: int = 384
    image_embedding_dim: int = 1280

    # --- Similarity weights ---
    weight_text: float = 0.4
    weight_image: float = 0.3
    weight_structured: float = 0.3

    # --- Fallback weights (no image available) ---
    fallback_weight_text: float = 0.6
    fallback_weight_structured: float = 0.4

    # --- Image cache ---
    image_cache_dir: Path = Path(".cache/images")
    image_cache_size: int = 10_000

    # --- Feature extraction ---
    skip_images: bool = True  # skip image download/extraction for fast startup

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
