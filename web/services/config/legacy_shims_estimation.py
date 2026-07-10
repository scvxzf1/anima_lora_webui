"""Estimation-domain legacy shim installers."""

from __future__ import annotations

from typing import Any


def install_estimation_public_shims(legacy_globals: dict[str, Any]) -> dict[str, Any]:
    """Install estimation public shims and return the shim map."""
    from web.services import config_service as _facade
    from web.services.config import estimation as _estimation

    sync_names = legacy_globals["_ESTIMATION_SHIM_SYNC_NAMES"]

    def _call_estimation_impl(name: str, *args, **kwargs):
        sync_state = {
            sync_name: legacy_globals[sync_name]
            for sync_name in sync_names
            if sync_name in legacy_globals
        }
        facade_previous = {
            sync_name: getattr(_facade, sync_name)
            for sync_name in sync_state
            if hasattr(_facade, sync_name)
        }
        facade_missing = set(sync_state) - set(facade_previous)
        estimation_previous = {
            sync_name: getattr(_estimation, sync_name)
            for sync_name in sync_state
            if hasattr(_estimation, sync_name)
        }
        estimation_missing = set(sync_state) - set(estimation_previous)
        try:
            for sync_name, value in sync_state.items():
                setattr(_facade, sync_name, value)
            _estimation._sync_from_facade()
            for sync_name, value in sync_state.items():
                if hasattr(_facade, sync_name):
                    setattr(_estimation, sync_name, getattr(_facade, sync_name))
                else:
                    setattr(_estimation, sync_name, value)
            exported = getattr(_estimation, name)
            impl = getattr(exported, "__wrapped__", exported)
            return impl(*args, **kwargs)
        finally:
            for sync_name, value in facade_previous.items():
                setattr(_facade, sync_name, value)
            for sync_name in facade_missing:
                if hasattr(_facade, sync_name):
                    delattr(_facade, sync_name)
            for sync_name, value in estimation_previous.items():
                setattr(_estimation, sync_name, value)
            for sync_name in estimation_missing:
                if hasattr(_estimation, sync_name):
                    delattr(_estimation, sync_name)

    shims = {}
    for estimation_name in legacy_globals["_ESTIMATION_SHIM_NAMES"]:
        def _make(name=estimation_name):
            def shim(*args, **kwargs):
                return _call_estimation_impl(name, *args, **kwargs)
            shim.__name__ = name
            shim.__qualname__ = name
            shim.__doc__ = f"Compatibility shim forwarding to web.services.config.estimation.{name}."
            return shim
        shims[estimation_name] = _make()
    for estimation_name, estimation_shim in shims.items():
        legacy_globals[estimation_name] = estimation_shim
    return shims
