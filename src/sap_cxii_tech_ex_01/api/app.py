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
from fastapi import FastAPI

from sap_cxii_tech_ex_01.config import Settings, get_settings
from sap_cxii_tech_ex_01.search import load_dataset

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


if __name__ == "__main__":
    import uvicorn

    s = get_settings()
    uvicorn.run("sap_cxii_tech_ex_01.api.app:app", host=str(s.api_host), port=s.api_port, reload=False)
