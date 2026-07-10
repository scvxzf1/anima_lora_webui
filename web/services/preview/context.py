"""Shared facade access helpers for split WebUI preview modules."""

from __future__ import annotations

from typing import Any


def preview_facade():
    from web.services import preview_service as facade

    return facade


def call(name: str, *args: Any, **kwargs: Any) -> Any:
    return getattr(preview_facade(), name)(*args, **kwargs)


def get(name: str) -> Any:
    return getattr(preview_facade(), name)
