"""Feature extraction contract tests — RED phase.

All three feature extractor modules do not yet exist; every test in this file
is expected to FAIL (ModuleNotFoundError) until Steps 5–7 (GREEN).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from sap_cxii_tech_ex_01.features.image import ImageExtractor  # noqa: F401 — RED
from sap_cxii_tech_ex_01.features.structured import StructuredExtractor  # noqa: F401 — RED
from sap_cxii_tech_ex_01.features.text import TextExtractor  # noqa: F401 — RED


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _make_fake_encode(dim: int):
    """Return a callable that produces a (N, dim) float32 array of ones."""

    def _encode(texts, **kwargs):
        n = len(texts) if hasattr(texts, "__len__") else 1
        return np.ones((n, dim), dtype=np.float32)

    return _encode


# ── Text fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def mock_sentence_transformer(settings):
    """Patch SentenceTransformer so tests run without model downloads."""
    dim = settings.text_embedding_dim
    with patch("sap_cxii_tech_ex_01.features.text.SentenceTransformer") as mock_cls:
        instance = MagicMock()
        instance.encode.side_effect = _make_fake_encode(dim)
        mock_cls.return_value = instance
        yield instance


@pytest.fixture
def text_extractor(settings, mock_sentence_transformer):
    return TextExtractor(settings)


# ── Image fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def image_extractor(settings):
    """ImageExtractor with model loading patched out."""
    with patch("sap_cxii_tech_ex_01.features.image._load_model") as mock_load:
        mock_load.return_value = MagicMock()
        extractor = ImageExtractor(settings)
    return extractor


# ── Text feature tests ─────────────────────────────────────────────────────────

def test_text_extract_returns_correct_shape(
    text_extractor: TextExtractor,
    settings,
) -> None:
    """extract(['text']) → shape (1, text_embedding_dim)."""
    result = text_extractor.extract(["Blue Denim Jacket"])
    assert result.shape == (1, settings.text_embedding_dim)


def test_text_batch_extraction_shape(
    text_extractor: TextExtractor,
    settings,
) -> None:
    """extract([...N texts...]) → shape (N, text_embedding_dim)."""
    texts = ["jacket", "dress", "shirt", "trousers", "coat"]
    result = text_extractor.extract(texts)
    assert result.shape == (len(texts), settings.text_embedding_dim)


def test_text_empty_string_returns_zero_vector(
    text_extractor: TextExtractor,
    settings,
) -> None:
    """Empty string must return a zero vector — no exception raised."""
    dim = settings.text_embedding_dim
    result = text_extractor.extract([""])
    assert result.shape == (1, dim)
    np.testing.assert_array_equal(result, np.zeros((1, dim)))


def test_text_none_input_returns_zero_vector(
    text_extractor: TextExtractor,
    settings,
) -> None:
    """None element in list must return a zero vector — no exception raised."""
    dim = settings.text_embedding_dim
    result = text_extractor.extract([None])
    assert result.shape == (1, dim)
    np.testing.assert_array_equal(result, np.zeros((1, dim)))


def test_text_output_dtype_float32(
    text_extractor: TextExtractor,
    settings,
) -> None:
    """Text embeddings must be float32 (sentence-transformers default)."""
    result = text_extractor.extract(["test"])
    assert result.dtype == np.float32


# ── Image feature tests ────────────────────────────────────────────────────────

def test_image_extract_returns_correct_shape(
    image_extractor: ImageExtractor,
    settings,
) -> None:
    """extract(['url']) → shape (1, image_embedding_dim)."""
    dim = settings.image_embedding_dim
    with patch.object(image_extractor, "_fetch_embedding") as mock_fetch:
        mock_fetch.return_value = np.ones(dim, dtype=np.float32)
        result = image_extractor.extract(["https://example.com/img.jpg"])
    assert result.shape == (1, dim)


def test_image_batch_extraction_shape(
    image_extractor: ImageExtractor,
    settings,
) -> None:
    """extract([...N urls...]) → shape (N, image_embedding_dim)."""
    dim = settings.image_embedding_dim
    urls = [f"https://example.com/img{i}.jpg" for i in range(4)]
    with patch.object(image_extractor, "_fetch_embedding") as mock_fetch:
        mock_fetch.return_value = np.ones(dim, dtype=np.float32)
        result = image_extractor.extract(urls)
    assert result.shape == (len(urls), dim)


def test_image_unavailable_url_returns_zero_vector(
    image_extractor: ImageExtractor,
    settings,
) -> None:
    """Failed image fetch must return a zero vector — no exception raised."""
    dim = settings.image_embedding_dim
    with patch.object(image_extractor, "_fetch_embedding") as mock_fetch:
        mock_fetch.side_effect = Exception("Network error / decode error")
        result = image_extractor.extract(["https://invalid.url/nonexistent.jpg"])
    assert result.shape == (1, dim)
    np.testing.assert_array_equal(result, np.zeros((1, dim)))


def test_image_none_url_returns_zero_vector(
    image_extractor: ImageExtractor,
    settings,
) -> None:
    """None URL must return a zero vector — no exception raised."""
    dim = settings.image_embedding_dim
    result = image_extractor.extract([None])
    assert result.shape == (1, dim)
    np.testing.assert_array_equal(result, np.zeros((1, dim)))


def test_image_cached_url_not_re_fetched(
    image_extractor: ImageExtractor,
    settings,
) -> None:
    """Calling extract twice with the same URL must not call _fetch_embedding twice."""
    dim = settings.image_embedding_dim
    url = "https://example.com/cached.jpg"
    with patch.object(image_extractor, "_fetch_embedding") as mock_fetch:
        mock_fetch.return_value = np.ones(dim, dtype=np.float32)
        image_extractor.extract([url])
        image_extractor.extract([url])
        assert mock_fetch.call_count == 1  # second call served from cache


def test_image_output_dtype_float32(
    image_extractor: ImageExtractor,
    settings,
) -> None:
    """Image embeddings must be float32 (torchvision default)."""
    dim = settings.image_embedding_dim
    with patch.object(image_extractor, "_fetch_embedding") as mock_fetch:
        mock_fetch.return_value = np.ones(dim, dtype=np.float32)
        result = image_extractor.extract(["https://example.com/img.jpg"])
    assert result.dtype == np.float32


# ── Structured feature tests ───────────────────────────────────────────────────

def test_structured_output_shape(
    settings,
    sample_df: pd.DataFrame,
) -> None:
    """fit_transform(df) → shape (N, D) with D > 0, N == len(df)."""
    s = StructuredExtractor(settings)
    result = s.fit_transform(sample_df)
    assert result.ndim == 2
    assert result.shape[0] == len(sample_df)
    assert result.shape[1] > 0


def test_structured_no_nan_in_output(
    settings,
    sample_df: pd.DataFrame,
) -> None:
    """Output array must contain no NaN values."""
    s = StructuredExtractor(settings)
    result = s.fit_transform(sample_df)
    assert not np.isnan(result).any()


def test_structured_output_dtype_float64(
    settings,
    sample_df: pd.DataFrame,
) -> None:
    """Output array must be float64 (sklearn convention)."""
    s = StructuredExtractor(settings)
    result = s.fit_transform(sample_df)
    assert result.dtype == np.float64


def test_structured_fit_transform_consistent(
    settings,
    sample_df: pd.DataFrame,
) -> None:
    """fit(df).transform(df) must produce identical output to fit_transform(df)."""
    s1 = StructuredExtractor(settings)
    combined = s1.fit_transform(sample_df)

    s2 = StructuredExtractor(settings)
    s2.fit(sample_df)
    separate = s2.transform(sample_df)

    np.testing.assert_array_equal(combined, separate)


def test_structured_numeric_columns_scaled(
    settings,
    sample_df: pd.DataFrame,
) -> None:
    """After StandardScaler, numeric feature columns should have mean ≈ 0."""
    s = StructuredExtractor(settings)
    result = s.fit_transform(sample_df)
    # First column is sales_price (scaled); mean should be close to 0
    assert abs(result[:, 0].mean()) == pytest.approx(0.0, abs=0.1)
