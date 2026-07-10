"""File-groups-domain legacy shim installers."""

from __future__ import annotations

from typing import Any

from web.services.config.legacy_shims_common import restore_raw_files_shims


def install_file_groups_public_shims(legacy_globals: dict[str, Any]) -> dict[str, Any]:
    """Install file_groups public shims and expose _call_file_groups_impl."""
    from web.services import config_service as _facade
    from web.services.config import file_groups as _file_groups

    sync_names = legacy_globals["_FILE_GROUPS_SHIM_SYNC_NAMES"]

    def _call_file_groups_impl(name: str, *args, **kwargs):
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
        groups_previous = {
            sync_name: getattr(_file_groups, sync_name)
            for sync_name in sync_state
            if hasattr(_file_groups, sync_name)
        }
        groups_missing = set(sync_state) - set(groups_previous)
        try:
            for sync_name, value in sync_state.items():
                setattr(_facade, sync_name, value)
            _file_groups._sync_from_facade()
            for sync_name, value in sync_state.items():
                if hasattr(_facade, sync_name):
                    setattr(_file_groups, sync_name, getattr(_facade, sync_name))
                else:
                    setattr(_file_groups, sync_name, value)
            restore_raw_files_shims(legacy_globals)
            exported = getattr(_file_groups, name)
            impl = getattr(exported, "__wrapped__", exported)
            return impl(*args, **kwargs)
        finally:
            for sync_name, value in facade_previous.items():
                setattr(_facade, sync_name, value)
            for sync_name in facade_missing:
                if hasattr(_facade, sync_name):
                    delattr(_facade, sync_name)
            for sync_name, value in groups_previous.items():
                setattr(_file_groups, sync_name, value)
            for sync_name in groups_missing:
                if hasattr(_file_groups, sync_name):
                    delattr(_file_groups, sync_name)
            restore_raw_files_shims(legacy_globals)

    legacy_globals["_call_file_groups_impl"] = _call_file_groups_impl

    shims = {}
    for group_name in legacy_globals["_FILE_GROUPS_SHIM_NAMES"]:
        def _make(name=group_name):
            def shim(*args, **kwargs):
                return _call_file_groups_impl(name, *args, **kwargs)

            shim.__name__ = name
            shim.__qualname__ = name
            shim.__doc__ = f"Compatibility shim forwarding to web.services.config.file_groups.{name}."
            return shim

        shims[group_name] = _make()
    for group_name, group_shim in shims.items():
        legacy_globals[group_name] = group_shim
    return shims
