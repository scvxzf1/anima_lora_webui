"""Raw-files-domain legacy shim installers."""

from __future__ import annotations

from typing import Any

from web.services.config.legacy_shims_common import restore_raw_files_shims


def install_raw_files_public_shims(legacy_globals: dict[str, Any]) -> dict[str, Any]:
    """Install raw_files public shims and expose _call_raw_files_impl."""
    from web.services import config_service as _facade
    from web.services.config import raw_files as _raw_files

    sync_names = legacy_globals["_RAW_FILES_SHIM_SYNC_NAMES"]
    legacy_helper_names = legacy_globals.get("_RAW_FILES_LEGACY_HELPER_NAMES", ())
    facade_helper_names = legacy_globals.get("_RAW_FILES_FACADE_HELPER_NAMES", ())

    def _call_raw_files_impl(name: str, *args, **kwargs):
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
        try:
            for sync_name, value in sync_state.items():
                setattr(_facade, sync_name, value)
            _raw_files._sync_from_facade()
            for sync_name, value in sync_state.items():
                setattr(_raw_files, sync_name, value)
            for helper_name in legacy_helper_names:
                if helper_name in legacy_globals:
                    setattr(_raw_files, helper_name, legacy_globals[helper_name])
            for helper_name in facade_helper_names:
                if hasattr(_facade, helper_name):
                    setattr(_raw_files, helper_name, getattr(_facade, helper_name))
                elif helper_name in legacy_globals:
                    setattr(_raw_files, helper_name, legacy_globals[helper_name])
            for sync_name, value in sync_state.items():
                legacy_globals[sync_name] = value
            restore_raw_files_shims(legacy_globals)
            exported = getattr(_raw_files, name)
            impl = getattr(exported, "__wrapped__", exported)
            return impl(*args, **kwargs)
        finally:
            for sync_name, value in facade_previous.items():
                setattr(_facade, sync_name, value)
            for sync_name in facade_missing:
                if hasattr(_facade, sync_name):
                    delattr(_facade, sync_name)
            restore_raw_files_shims(legacy_globals)

    legacy_globals["_call_raw_files_impl"] = _call_raw_files_impl

    shims = {}
    for raw_name in legacy_globals["_RAW_FILES_SHIM_NAMES"]:
        def _make(name=raw_name):
            def shim(*args, **kwargs):
                return _call_raw_files_impl(name, *args, **kwargs)

            shim.__name__ = name
            shim.__qualname__ = name
            shim.__doc__ = f"Compatibility shim forwarding to web.services.config.raw_files.{name}."
            return shim

        shims[raw_name] = _make()
    for raw_name, raw_shim in shims.items():
        legacy_globals[raw_name] = raw_shim
    return shims
