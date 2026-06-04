"""FastAPI application — similarity search microservice.

Endpoints
---------
GET /find_similar_products   Main similarity search
GET /health                  Liveness probe
GET /ready                   Readiness probe
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from sap_cxii_tech_ex_01.config import Settings, get_settings
from sap_cxii_tech_ex_01.search import load_dataset  # noqa: F401 — imported for patch target in tests

__all__ = ["app"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload dataset and settings once at startup; store in app.state."""
    settings: Settings = get_settings()
    df: pd.DataFrame = load_dataset(settings)
    app.state.settings = settings
    app.state.df = df
    app.state.startup_time = time.monotonic()
    app.state.ready = True
    yield
    app.state.ready = False


app = FastAPI(
    title="Product Similarity Search",
    description="Multimodal similarity search over Amazon Fashion products.",
    version="1.0.0",
    lifespan=lifespan,
)

# Routes are registered in separate modules and included here.
from sap_cxii_tech_ex_01.api import routes  # noqa: E402 — import after app creation

app.include_router(routes.router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, _exc: Exception) -> JSONResponse:
    """Return a structured 500 with no stack trace in production."""
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )


if __name__ == "__main__":
    import uvicorn

    s = get_settings()
    uvicorn.run("sap_cxii_tech_ex_01.api.app:app", host=str(s.api_host), port=s.api_port, reload=False)
