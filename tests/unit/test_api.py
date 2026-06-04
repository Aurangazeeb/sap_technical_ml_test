"""API endpoint contract tests (RED).

All tests in this file should FAIL until the route handlers are implemented
in Steps 3-5.  Tests define the contract — status codes, response schema,
and error behaviour — without relying on the real dataset or ML models.

Injection strategy (DR_0008):
- Patch ``sap_cxii_tech_ex_01.api.app.get_settings`` / ``load_dataset`` /
  ``build_search_state`` so the lifespan runs with test fixtures stored in
  ``app.state``.
- Patch ``sap_cxii_tech_ex_01.api.routes.find_similar_products`` to control
  what each scenario returns / raises.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from starlette.testclient import TestClient

import sap_cxii_tech_ex_01.api.app as _app_mod
import sap_cxii_tech_ex_01.api.routes as _routes_mod
from sap_cxii_tech_ex_01.api.app import app
from sap_cxii_tech_ex_01.config import Settings
from sap_cxii_tech_ex_01.search import SearchState

# ── Constants ─────────────────────────────────────────────────────────────────

KNOWN_ID = "prod_001"
UNKNOWN_ID = "unknown_product_xyz"
_MOCK_SIMILAR = ["prod_001", "prod_002", "prod_003"]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def test_settings() -> Settings:
    return Settings(data_path="data/test.ldjson")


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uniq_id": [f"prod_{i:03d}" for i in range(5)],
            "product_name": [f"Product {i}" for i in range(5)],
            "sales_price": [100.0 * (i + 1) for i in range(5)],
        }
    )


@pytest.fixture
def dummy_search_state() -> SearchState:
    n, dim = 5, 8
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    normalized = vecs / np.where(norms == 0.0, 1.0, norms)
    return SearchState(
        ids=[f"prod_{i:03d}" for i in range(n)],
        normalized=normalized,
        prices=None,
    )


@pytest.fixture
def api_client(test_settings: Settings, tiny_df: pd.DataFrame, dummy_search_state: SearchState):
    """TestClient whose lifespan is controlled by patches."""
    with patch.object(_app_mod, "get_settings", return_value=test_settings):
        with patch.object(_app_mod, "load_dataset", return_value=tiny_df):
            with patch.object(_app_mod, "build_search_state", return_value=dummy_search_state):
                with patch.object(
                    _routes_mod, "find_similar_products", return_value=_MOCK_SIMILAR
                ) as mock_fn:
                    with TestClient(app) as client:
                        client._mock_find_similar = mock_fn
                        yield client


@pytest.fixture
def api_client_no_raise(test_settings: Settings, tiny_df: pd.DataFrame, dummy_search_state: SearchState):
    """Like ``api_client`` but surfaces 500 responses instead of re-raising."""
    with patch.object(_app_mod, "get_settings", return_value=test_settings):
        with patch.object(_app_mod, "load_dataset", return_value=tiny_df):
            with patch.object(_app_mod, "build_search_state", return_value=dummy_search_state):
                with patch.object(
                    _routes_mod, "find_similar_products", return_value=_MOCK_SIMILAR
                ) as mock_fn:
                    with TestClient(app, raise_server_exceptions=False) as client:
                        client._mock_find_similar = mock_fn
                        yield client


# ── Happy-path tests ──────────────────────────────────────────────────────────

def test_find_similar_returns_200(api_client: TestClient) -> None:
    resp = api_client.get(f"/find_similar_products?product_id={KNOWN_ID}&num_similar=3")
    assert resp.status_code == 200


def test_response_schema_has_required_fields(api_client: TestClient) -> None:
    resp = api_client.get(f"/find_similar_products?product_id={KNOWN_ID}&num_similar=3")
    body = resp.json()
    assert "product_id" in body
    assert "similar_products" in body
    assert "count" in body


def test_response_echoes_product_id(api_client: TestClient) -> None:
    resp = api_client.get(f"/find_similar_products?product_id={KNOWN_ID}&num_similar=3")
    assert resp.json()["product_id"] == KNOWN_ID


def test_similar_products_is_list_of_str(api_client: TestClient) -> None:
    resp = api_client.get(f"/find_similar_products?product_id={KNOWN_ID}&num_similar=3")
    similar = resp.json()["similar_products"]
    assert isinstance(similar, list)
    assert all(isinstance(s, str) for s in similar)


def test_count_matches_list_length(api_client: TestClient) -> None:
    resp = api_client.get(f"/find_similar_products?product_id={KNOWN_ID}&num_similar=3")
    body = resp.json()
    assert body["count"] == len(body["similar_products"])


def test_num_similar_controls_result_length(api_client: TestClient) -> None:
    api_client._mock_find_similar.return_value = ["prod_001", "prod_002"]
    resp = api_client.get(f"/find_similar_products?product_id={KNOWN_ID}&num_similar=2")
    assert len(resp.json()["similar_products"]) == 2


# ── 404 — product not found ───────────────────────────────────────────────────

def test_unknown_product_returns_404(api_client: TestClient) -> None:
    api_client._mock_find_similar.side_effect = ValueError(
        f"product_id '{UNKNOWN_ID}' not found"
    )
    resp = api_client.get(f"/find_similar_products?product_id={UNKNOWN_ID}&num_similar=3")
    assert resp.status_code == 404


def test_not_found_body_has_detail(api_client: TestClient) -> None:
    api_client._mock_find_similar.side_effect = ValueError("not found")
    resp = api_client.get(f"/find_similar_products?product_id={UNKNOWN_ID}&num_similar=3")
    assert "detail" in resp.json()


# ── 400 — bad num_similar ─────────────────────────────────────────────────────

def test_num_similar_zero_returns_400(api_client: TestClient) -> None:
    resp = api_client.get(f"/find_similar_products?product_id={KNOWN_ID}&num_similar=0")
    assert resp.status_code == 400


def test_num_similar_negative_returns_400(api_client: TestClient) -> None:
    resp = api_client.get(f"/find_similar_products?product_id={KNOWN_ID}&num_similar=-5")
    assert resp.status_code == 400


def test_num_similar_exceeds_dataset_returns_400(api_client: TestClient, tiny_df: pd.DataFrame) -> None:
    too_many = len(tiny_df) + 100
    resp = api_client.get(
        f"/find_similar_products?product_id={KNOWN_ID}&num_similar={too_many}"
    )
    assert resp.status_code == 400


# ── 422 — FastAPI built-in validation ────────────────────────────────────────

def test_missing_product_id_returns_422(api_client: TestClient) -> None:
    resp = api_client.get("/find_similar_products?num_similar=3")
    assert resp.status_code == 422


def test_missing_num_similar_returns_422(api_client: TestClient) -> None:
    resp = api_client.get(f"/find_similar_products?product_id={KNOWN_ID}")
    assert resp.status_code == 422


def test_non_integer_num_similar_returns_422(api_client: TestClient) -> None:
    resp = api_client.get(
        f"/find_similar_products?product_id={KNOWN_ID}&num_similar=abc"
    )
    assert resp.status_code == 422


# ── 500 — unexpected errors ───────────────────────────────────────────────────

def test_unexpected_error_returns_500(api_client_no_raise: TestClient) -> None:
    api_client_no_raise._mock_find_similar.side_effect = RuntimeError("unexpected crash")
    resp = api_client_no_raise.get(
        f"/find_similar_products?product_id={KNOWN_ID}&num_similar=3"
    )
    assert resp.status_code == 500


def test_500_body_has_detail_no_traceback(api_client_no_raise: TestClient) -> None:
    api_client_no_raise._mock_find_similar.side_effect = RuntimeError("boom")
    resp = api_client_no_raise.get(
        f"/find_similar_products?product_id={KNOWN_ID}&num_similar=3"
    )
    body = resp.json()
    assert "detail" in body
    assert "traceback" not in body
    assert "Traceback" not in body.get("detail", "")


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_returns_200(api_client: TestClient) -> None:
    assert api_client.get("/health").status_code == 200


def test_health_schema_has_required_fields(api_client: TestClient) -> None:
    body = api_client.get("/health").json()
    assert "status" in body
    assert "dataset_size" in body
    assert "uptime_seconds" in body


def test_health_dataset_size_matches_fixture(
    api_client: TestClient, tiny_df: pd.DataFrame
) -> None:
    body = api_client.get("/health").json()
    assert body["dataset_size"] == len(tiny_df)


def test_health_uptime_is_non_negative(api_client: TestClient) -> None:
    body = api_client.get("/health").json()
    assert body["uptime_seconds"] >= 0.0


# ── /ready ────────────────────────────────────────────────────────────────────

def test_ready_returns_200(api_client: TestClient) -> None:
    assert api_client.get("/ready").status_code == 200


def test_ready_schema_has_ready_field(api_client: TestClient) -> None:
    body = api_client.get("/ready").json()
    assert "ready" in body


def test_ready_is_true_after_startup(api_client: TestClient) -> None:
    assert api_client.get("/ready").json()["ready"] is True
