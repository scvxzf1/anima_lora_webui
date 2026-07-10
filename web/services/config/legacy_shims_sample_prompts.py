"""Sample-prompts-domain legacy shim installers."""

from __future__ import annotations

from typing import Any

from web.services.config.legacy_shims_common import restore_raw_files_shims


def install_sample_prompts_public_shims(legacy_globals: dict[str, Any]) -> dict[str, Any]:
    """Install sample_prompts public shims and expose _call_sample_prompts_impl."""
    from web.services import config_service as _facade
    from web.services.config import sample_prompts as _sample_prompts

    sync_names = legacy_globals["_SAMPLE_PROMPTS_SHIM_SYNC_NAMES"]

    def _call_sample_prompts_impl(name: str, *args, **kwargs):
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
        for sync_name, value in sync_state.items():
            setattr(_facade, sync_name, value)
        _sample_prompts._sync_from_facade()
        for sync_name, value in sync_state.items():
            setattr(_sample_prompts, sync_name, value)
        restore_raw_files_shims(legacy_globals)
        exported = getattr(_sample_prompts, name)
        impl = getattr(exported, "__wrapped__", exported)
        try:
            return impl(*args, **kwargs)
        finally:
            for sync_name, value in facade_previous.items():
                setattr(_facade, sync_name, value)
            for sync_name in facade_missing:
                if hasattr(_facade, sync_name):
                    delattr(_facade, sync_name)
            restore_raw_files_shims(legacy_globals)

    legacy_globals["_call_sample_prompts_impl"] = _call_sample_prompts_impl

    shims = {}
    for sample_name in legacy_globals["_SAMPLE_PROMPTS_SHIM_NAMES"]:
        def _make(name=sample_name):
            def shim(*args, **kwargs):
                return _call_sample_prompts_impl(name, *args, **kwargs)

            shim.__name__ = name
            shim.__qualname__ = name
            shim.__doc__ = f"Compatibility shim forwarding to web.services.config.sample_prompts.{name}."
            return shim

        shims[sample_name] = _make()
    for sample_name, sample_shim in shims.items():
        legacy_globals[sample_name] = sample_shim
    return shims
