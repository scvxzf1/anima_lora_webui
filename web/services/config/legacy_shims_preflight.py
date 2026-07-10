"""Preflight-domain legacy shim installers."""

from __future__ import annotations

from typing import Any

from web.services.config.legacy_shims_common import restore_raw_files_shims


def install_preflight_public_shims(legacy_globals: dict[str, Any]) -> dict[str, Any]:
    """Install preflight public shims and expose _call_preflight_impl on legacy globals."""
    from web.services import config_service as _facade
    from web.services.config import preflight as _preflight

    sync_names = legacy_globals["_PREFLIGHT_SHIM_SYNC_NAMES"]

    def _call_preflight_impl(name: str, *args, **kwargs):
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
        preflight_previous = {
            sync_name: getattr(_preflight, sync_name)
            for sync_name in sync_state
            if hasattr(_preflight, sync_name)
        }
        preflight_missing = set(sync_state) - set(preflight_previous)
        try:
            for sync_name, value in sync_state.items():
                setattr(_facade, sync_name, value)
            _preflight._sync_from_facade()
            for sync_name, value in sync_state.items():
                # Prefer the facade's current binding after sync so monkeypatches
                # on config_service remain visible to preflight helpers.
                if hasattr(_facade, sync_name):
                    setattr(_preflight, sync_name, getattr(_facade, sync_name))
                else:
                    setattr(_preflight, sync_name, value)
            restore_raw_files_shims(legacy_globals)
            exported = getattr(_preflight, name)
            impl = getattr(exported, "__wrapped__", exported)
            return impl(*args, **kwargs)
        finally:
            for sync_name, value in facade_previous.items():
                setattr(_facade, sync_name, value)
            for sync_name in facade_missing:
                if hasattr(_facade, sync_name):
                    delattr(_facade, sync_name)
            for sync_name, value in preflight_previous.items():
                setattr(_preflight, sync_name, value)
            for sync_name in preflight_missing:
                if hasattr(_preflight, sync_name):
                    delattr(_preflight, sync_name)
            restore_raw_files_shims(legacy_globals)

    legacy_globals["_call_preflight_impl"] = _call_preflight_impl

    shims = {}
    for preflight_name in legacy_globals["_PREFLIGHT_SHIM_NAMES"]:
        def _make(name=preflight_name):
            def shim(*args, **kwargs):
                return _call_preflight_impl(name, *args, **kwargs)
            shim.__name__ = name
            shim.__qualname__ = name
            shim.__doc__ = f"Compatibility shim forwarding to web.services.config.preflight.{name}."
            return shim
        shims[preflight_name] = _make()
    for preflight_name, preflight_shim in shims.items():
        legacy_globals[preflight_name] = preflight_shim

    for helper_name in legacy_globals.get("_PREFLIGHT_PRIVATE_HELPER_NAMES", ()):
        def _make_helper(name=helper_name):
            def shim(*args, **kwargs):
                return _call_preflight_impl(name, *args, **kwargs)

            shim.__name__ = name
            shim.__qualname__ = name
            shim.__doc__ = (
                f"Compatibility shim forwarding to web.services.config.preflight.{name}."
            )
            return shim

        legacy_globals[helper_name] = _make_helper()
    return shims
