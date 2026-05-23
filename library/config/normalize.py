"""Helpers for turning config values into plain Python containers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def to_plain_config(value: Any) -> Any:
    """Recursively convert config mapping/list containers to pickle-safe types."""
    if isinstance(value, Mapping):
        return {key: to_plain_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_config(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_plain_config(item) for item in value)
    return value
