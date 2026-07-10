"""Shared facade access helpers for split weight analysis modules."""

from __future__ import annotations

from typing import Any


def weight_analysis_facade():
    from web.services import weight_analysis_service as facade

    return facade


def call(name: str, *args: Any, **kwargs: Any) -> Any:
    return getattr(weight_analysis_facade(), name)(*args, **kwargs)


def get(name: str) -> Any:
    return getattr(weight_analysis_facade(), name)
