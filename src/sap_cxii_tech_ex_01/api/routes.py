"""Route handlers — stubs only (implemented in Steps 3–5)."""
from __future__ import annotations

from fastapi import APIRouter

from sap_cxii_tech_ex_01.search import find_similar_products  # noqa: F401 — imported for patch target

__all__ = ["router"]

router = APIRouter()
