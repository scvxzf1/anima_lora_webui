"""Wiring layer for resume helpers used by history and queue code."""

from __future__ import annotations

from typing import Any

from web.services.training.common import _float_or_none, _format_ts, _int_or_none
from web.services.training.runtime_paths import (
    _display_project_path,
    _path_exists,
    _path_is_relative_to,
    _resolve_display_path,
    resolve_output_root,
)
from web.services.training.storage import _read_json


def _is_web_runtime_dir(*args, **kwargs):
    from web.services.training import runtime_config as _impl

    return _impl._is_web_runtime_dir(*args, **kwargs)


def _path_exists_wrapped(*args, **kwargs):
    return _path_exists(*args, **kwargs)


def _resolve_display_path_wrapped(*args, **kwargs):
    return _resolve_display_path(*args, **kwargs)


def _display_project_path_wrapped(*args, **kwargs):
    return _display_project_path(*args, **kwargs)


def _list_resume_checkpoints(task: dict[str, Any]) -> list[dict[str, Any]]:
    from web.services.training import resume as _resume
    from web.services.training.runtime_common import _load_config_file_config

    config_file = str(task.get("config_snapshot") or "").strip()
    config = _load_config_file_config(config_file) if config_file else {}

    return _resume._list_resume_checkpoints(
        task,
        scheduler_required=_resume._resume_scheduler_state_required(config),
        resolve_display_path=_resolve_display_path,
        display_project_path=_display_project_path,
        path_exists=_path_exists,
        read_json=_read_json,
        int_or_none=_int_or_none,
        float_or_none=_float_or_none,
        format_ts=_format_ts,
    )


def _resume_checkpoint_diagnostic(
    task: dict[str, Any],
    checkpoints: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from web.services.training import resume as _resume

    return _resume._resume_checkpoint_diagnostic(
        task,
        checkpoints,
        resolve_display_path=_resolve_display_path,
        display_project_path=_display_project_path,
        path_exists=_path_exists,
        float_or_none=_float_or_none,
        resolve_output_root=resolve_output_root,
        path_is_relative_to=_path_is_relative_to,
        is_web_runtime_dir=_is_web_runtime_dir,
    )


def _select_resume_checkpoint(
    checkpoints: list[dict[str, Any]],
    checkpoint: str | None,
) -> dict[str, Any] | None:
    from web.services.training import resume as _resume

    return _resume._select_resume_checkpoint(
        checkpoints,
        checkpoint,
        resolve_display_path=_resolve_display_path,
        display_project_path=_display_project_path,
    )
