"""Shared facade access helpers for split WebUI training modules."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any


def training_facade():
    from web.services import training_service as facade

    return facade


def bind_legacy(module_globals: dict[str, Any], local_impl_names: Collection[str]) -> None:
    legacy = training_facade()
    local_names = set(local_impl_names)
    for name, value in vars(legacy).items():
        if name.startswith("__") or name in local_names:
            continue
        module_globals[name] = value
