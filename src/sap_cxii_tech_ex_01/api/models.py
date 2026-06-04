"""Pydantic v2 request/response and error models for the similarity search API."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["SimilarProductsResponse", "HealthResponse", "ReadyResponse", "ErrorResponse"]


class SimilarProductsResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "product_id": "abc123",
                    "similar_products": ["def456", "ghi789"],
                    "count": 2,
                }
            ]
        }
    )

    product_id: str = Field(..., description="The queried product ID")
    similar_products: list[str] = Field(..., description="Ordered list of similar product IDs")
    count: int = Field(..., description="Number of similar products returned")


class HealthResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"status": "ok", "dataset_size": 30000, "uptime_seconds": 42.1}]}
    )

    status: str = Field(..., description="Service status: 'ok' or 'degraded'")
    dataset_size: int = Field(..., description="Number of products loaded")
    uptime_seconds: float = Field(..., description="Seconds since startup")


class ReadyResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"ready": True}]}
    )

    ready: bool = Field(..., description="True when models and data are loaded")


class ErrorResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"detail": "product_id 'xyz' not found"}]}
    )

    detail: str = Field(..., description="Human-readable error message")
