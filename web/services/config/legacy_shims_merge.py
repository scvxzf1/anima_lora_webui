"""Merge-domain legacy shim installers."""

from __future__ import annotations

from typing import Any

from web.services.config.legacy_shims_common import restore_raw_files_shims


def install_merge_public_shims(legacy_globals: dict[str, Any]) -> dict[str, Any]:
    """Install merge public shims and return the shim map."""
    from web.services import config_service as _facade
    from web.services.config import merge as _merge

    sync_names = legacy_globals["_MERGE_SHIM_SYNC_NAMES"]

    def _call_merge_impl(name: str, *args, **kwargs):
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
        merge_previous = {
            sync_name: getattr(_merge, sync_name)
            for sync_name in sync_state
            if hasattr(_merge, sync_name)
        }
        merge_missing = set(sync_state) - set(merge_previous)
        try:
            for sync_name, value in sync_state.items():
                setattr(_facade, sync_name, value)
            _merge._sync_from_facade()
            restore_raw_files_shims(legacy_globals)
            for sync_name, value in sync_state.items():
                if hasattr(_facade, sync_name):
                    setattr(_merge, sync_name, getattr(_facade, sync_name))
                else:
                    setattr(_merge, sync_name, value)
            exported = getattr(_merge, name)
            impl = getattr(exported, "__wrapped__", exported)
            return impl(*args, **kwargs)
        finally:
            for sync_name, value in facade_previous.items():
                setattr(_facade, sync_name, value)
            for sync_name in facade_missing:
                if hasattr(_facade, sync_name):
                    delattr(_facade, sync_name)
            for sync_name, value in merge_previous.items():
                setattr(_merge, sync_name, value)
            for sync_name in merge_missing:
                if hasattr(_merge, sync_name):
                    delattr(_merge, sync_name)
            restore_raw_files_shims(legacy_globals)

    legacy_globals["_call_merge_impl"] = _call_merge_impl

    shims = {}
    for merge_name in legacy_globals["_MERGE_SHIM_NAMES"]:
        def _make(name=merge_name):
            def shim(*args, **kwargs):
                return _call_merge_impl(name, *args, **kwargs)
            shim.__name__ = name
            shim.__qualname__ = name
            shim.__doc__ = f"Compatibility shim forwarding to web.services.config.merge.{name}."
            return shim
        shims[merge_name] = _make()
    for merge_name, merge_shim in shims.items():
        legacy_globals[merge_name] = merge_shim

    # Private helpers if present
    for helper_name in legacy_globals.get("_MERGE_PRIVATE_HELPER_NAMES", ()):
        def _make_helper(name=helper_name):
            def shim(*args, **kwargs):
                return _call_merge_impl(name, *args, **kwargs)
            shim.__name__ = name
            shim.__qualname__ = name
            shim.__doc__ = f"Compatibility shim forwarding to web.services.config.merge.{name}."
            return shim
        legacy_globals[helper_name] = _make_helper()
    return shims
