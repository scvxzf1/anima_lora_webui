"""Output-runs-domain legacy shim installers."""

from __future__ import annotations

from typing import Any

from web.services.config.legacy_shims_common import restore_raw_files_shims


def install_output_runs_public_shims(legacy_globals: dict[str, Any]) -> dict[str, Any]:
    """Install output_runs public shims and return the shim map."""
    from web.services import config_service as _facade
    from web.services.config import output_runs as _output_runs

    sync_names = legacy_globals["_OUTPUT_RUNS_SHIM_SYNC_NAMES"]
    helper_names = legacy_globals.get("_OUTPUT_RUNS_LEGACY_HELPER_NAMES", ())

    def _call_output_runs_impl(name: str, *args, **kwargs):
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
        output_previous = {
            sync_name: getattr(_output_runs, sync_name)
            for sync_name in sync_state
            if hasattr(_output_runs, sync_name)
        }
        output_missing = set(sync_state) - set(output_previous)
        try:
            for sync_name, value in sync_state.items():
                setattr(_facade, sync_name, value)
            _output_runs._sync_from_facade()
            for sync_name, value in sync_state.items():
                # Prefer the facade binding after sync so monkeypatches stay visible.
                if hasattr(_facade, sync_name):
                    setattr(_output_runs, sync_name, getattr(_facade, sync_name))
                else:
                    setattr(_output_runs, sync_name, value)
            # Do not rebind private helpers from legacy here: those helpers are also
            # installed as legacy shims, so writing them back would recurse.
            restore_raw_files_shims(legacy_globals)
            exported = getattr(_output_runs, name)
            impl = getattr(exported, "__wrapped__", exported)
            return impl(*args, **kwargs)
        finally:
            for sync_name, value in facade_previous.items():
                setattr(_facade, sync_name, value)
            for sync_name in facade_missing:
                if hasattr(_facade, sync_name):
                    delattr(_facade, sync_name)
            for sync_name, value in output_previous.items():
                setattr(_output_runs, sync_name, value)
            for sync_name in output_missing:
                if hasattr(_output_runs, sync_name):
                    delattr(_output_runs, sync_name)
            restore_raw_files_shims(legacy_globals)

    legacy_globals["_call_output_runs_impl"] = _call_output_runs_impl

    shims = {}
    for output_name in legacy_globals["_OUTPUT_RUNS_SHIM_NAMES"]:
        def _make(name=output_name):
            def shim(*args, **kwargs):
                return _call_output_runs_impl(name, *args, **kwargs)
            shim.__name__ = name
            shim.__qualname__ = name
            shim.__doc__ = f"Compatibility shim forwarding to web.services.config.output_runs.{name}."
            return shim
        shims[output_name] = _make()
    for output_name, output_shim in shims.items():
        legacy_globals[output_name] = output_shim

    # Private helpers stay available on the legacy module but are not part of the
    # public shim map / __all__-style contract.
    for helper_name in helper_names:
        def _make_helper(name=helper_name):
            def shim(*args, **kwargs):
                return _call_output_runs_impl(name, *args, **kwargs)

            shim.__name__ = name
            shim.__qualname__ = name
            shim.__doc__ = (
                f"Compatibility shim forwarding to web.services.config.output_runs.{name}."
            )
            return shim

        legacy_globals[helper_name] = _make_helper()
    return shims
