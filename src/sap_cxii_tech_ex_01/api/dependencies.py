"""Dependency injection for shared application state.

Provides FastAPI ``Depends()``-compatible callables that expose the loaded
dataset and settings to route handlers without global mutable state.
"""
from __future__ import annotations

from fastapi import Request

import pandas as pd

from sap_cxii_tech_ex_01.config import Settings

__all__ = ["get_df", "get_settings_dep"]


def get_df(request: Request) -> pd.DataFrame:
    """Return the preloaded product DataFrame from app state."""
    return request.app.state.df


def get_settings_dep(request: Request) -> Settings:
    """Return the Settings instance from app state."""
    return request.app.state.settings
