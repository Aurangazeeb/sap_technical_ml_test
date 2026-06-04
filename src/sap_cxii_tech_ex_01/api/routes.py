"""Route handlers for the similarity search microservice."""
from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request

from sap_cxii_tech_ex_01.api.models import HealthResponse, ReadyResponse, SimilarProductsResponse
from sap_cxii_tech_ex_01.search import SearchState, find_similar_products

__all__ = ["router"]

router = APIRouter()


@router.get(
    "/find_similar_products",
    response_model=SimilarProductsResponse,
    summary="Find similar products",
)
def get_find_similar_products(
    product_id: str,
    num_similar: int,
    request: Request,
) -> SimilarProductsResponse:
    """Return the *num_similar* most similar products to *product_id*."""
    dataset_size = len(request.app.state.df)

    if num_similar <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"num_similar must be >= 1, got {num_similar}",
        )
    if num_similar > dataset_size:
        raise HTTPException(
            status_code=400,
            detail=(
                f"num_similar ({num_similar}) exceeds dataset size ({dataset_size})"
            ),
        )

    search_state: SearchState = request.app.state.search_state

    try:
        similar = find_similar_products(product_id, num_similar, search_state)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return SimilarProductsResponse(
        product_id=product_id,
        similar_products=similar,
        count=len(similar),
    )


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
def health(request: Request) -> HealthResponse:
    """Always returns 200 once the server is up."""
    uptime = time.monotonic() - request.app.state.startup_time
    return HealthResponse(
        status="ok",
        dataset_size=len(request.app.state.df),
        uptime_seconds=round(uptime, 3),
    )


@router.get("/ready", response_model=ReadyResponse, summary="Readiness probe")
def ready(request: Request) -> ReadyResponse:
    """Returns 200 with ready=True once data and models are loaded."""
    return ReadyResponse(ready=bool(request.app.state.ready))
