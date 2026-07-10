"""Common and shared legacy shim installers."""

from __future__ import annotations

from typing import Any


def restore_raw_files_shims(
    legacy_globals: dict[str, Any],
    *,
    file_groups_shims: dict[str, Any] | None = None,
    raw_files_shims: dict[str, Any] | None = None,
) -> None:
    """Re-bind raw file / file-group public shims after split-module sync calls."""
    groups = file_groups_shims if file_groups_shims is not None else legacy_globals.get("_FILE_GROUPS_SHIMS", {})
    raw = raw_files_shims if raw_files_shims is not None else legacy_globals.get("_RAW_FILES_SHIMS", {})
    for shim_name, shim in groups.items():
        legacy_globals[shim_name] = shim
    for shim_name, shim in raw.items():
        legacy_globals[shim_name] = shim

def install_common_public_shims(legacy_globals: dict[str, Any]) -> dict[str, Any]:
    """Install common public shims and expose _call_common_impl on legacy globals."""
    from web.services.config import common as _common

    sync_names = legacy_globals["_COMMON_SYNC_NAMES"]

    def _call_common_impl(name: str, *args, **kwargs):
        previous = {
            sync_name: getattr(_common, sync_name)
            for sync_name in sync_names
            if hasattr(_common, sync_name)
        }
        for sync_name in sync_names:
            if sync_name in legacy_globals:
                setattr(_common, sync_name, legacy_globals[sync_name])
        try:
            return getattr(_common, name)(*args, **kwargs)
        finally:
            for sync_name, value in previous.items():
                setattr(_common, sync_name, value)

    legacy_globals["_call_common_impl"] = _call_common_impl

    shims = {}
    for common_name in legacy_globals["_COMMON_SHIM_NAMES"]:
        def _make(name=common_name):
            def shim(*args, **kwargs):
                return _call_common_impl(name, *args, **kwargs)

            shim.__name__ = name
            shim.__qualname__ = name
            shim.__doc__ = f"Compatibility shim forwarding to web.services.config.common.{name}."
            return shim

        shims[common_name] = _make()
    for common_name, common_shim in shims.items():
        legacy_globals[common_name] = common_shim
    return shims
