"""Dataset-domain legacy shim installers."""

from __future__ import annotations

from typing import Any

from web.services.config.legacy_shims_common import restore_raw_files_shims


def install_dataset_public_shims(legacy_globals: dict[str, Any]) -> dict[str, Any]:
    """Install dataset public shims and expose _call_dataset_impl on legacy globals."""
    from web.services import config_service as _facade
    from web.services.config import datasets as _datasets

    sync_names = legacy_globals["_DATASET_SHIM_SYNC_NAMES"]

    def _call_dataset_impl(name: str, *args, **kwargs):
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
            _datasets._sync_from_facade()
            restore_raw_files_shims(legacy_globals)
            return getattr(_datasets, name)(*args, **kwargs)
        finally:
            for sync_name, value in facade_previous.items():
                setattr(_facade, sync_name, value)
            for sync_name in facade_missing:
                if hasattr(_facade, sync_name):
                    delattr(_facade, sync_name)
            restore_raw_files_shims(legacy_globals)

    legacy_globals["_call_dataset_impl"] = _call_dataset_impl

    shims = {}
    for dataset_name in legacy_globals["_DATASET_SHIM_NAMES"]:
        def _make(name=dataset_name):
            def shim(*args, **kwargs):
                return _call_dataset_impl(name, *args, **kwargs)

            shim.__name__ = name
            shim.__qualname__ = name
            shim.__doc__ = f"Compatibility shim forwarding to web.services.config.datasets.{name}."
            return shim

        shims[dataset_name] = _make()
    for dataset_name, dataset_shim in shims.items():
        legacy_globals[dataset_name] = dataset_shim
    return shims
