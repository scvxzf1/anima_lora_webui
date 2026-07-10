"""Shared facade accessors for training history modules."""

from __future__ import annotations

from typing import Any


def _training_facade():
    from web.services import training_service as facade
    return facade


def _list_history_tasks(*args, **kwargs):
    return _training_facade()._list_history_tasks(*args, **kwargs)


def _delete_history_tasks(*args, **kwargs):
    return _training_facade()._delete_history_tasks(*args, **kwargs)


def _batch_archive_history_tasks(*args, **kwargs):
    from web.services.training import history_batch as _history_batch

    return _history_batch.batch_archive_history_tasks(*args, **kwargs)


def _batch_set_history_group(*args, **kwargs):
    from web.services.training import history_batch as _history_batch

    return _history_batch.batch_set_history_group(*args, **kwargs)


def _build_config_group_timeline(*args, **kwargs):
    return _training_facade()._build_config_group_timeline(*args, **kwargs)


def _clone_frozen_runtime_config(*args, **kwargs):
    return _training_facade()._clone_frozen_runtime_config(*args, **kwargs)


def _default_history_archived(*args, **kwargs):
    return _training_facade()._default_history_archived(*args, **kwargs)


def _display_project_path(*args, **kwargs):
    return _training_facade()._display_project_path(*args, **kwargs)


def _history_delete_run_key(*args, **kwargs):
    return _training_facade()._history_delete_run_key(*args, **kwargs)


def _history_delete_task_preview(*args, **kwargs):
    return _training_facade()._history_delete_task_preview(*args, **kwargs)


def _history_runtime_delete_dirs_for_tasks(*args, **kwargs):
    return _training_facade()._history_runtime_delete_dirs_for_tasks(*args, **kwargs)


def _json_safe_training_payload(*args, **kwargs):
    from web.services.training.live_utils import _json_safe_training_payload as impl
    return impl(*args, **kwargs)


def _list_resume_checkpoints(*args, **kwargs):
    from web.services.training import resume_facade as _resume_facade

    return _resume_facade._list_resume_checkpoints(*args, **kwargs)


def _path_exists(*args, **kwargs):
    from web.services.training import resume_facade as _resume_facade

    return _resume_facade._path_exists_wrapped(*args, **kwargs)


def _queue_runtime_delete_blockers(*args, **kwargs):
    return _training_facade()._queue_runtime_delete_blockers(*args, **kwargs)


def _resolve_display_path(*args, **kwargs):
    from web.services.training import resume_facade as _resume_facade

    return _resume_facade._resolve_display_path_wrapped(*args, **kwargs)


def _resume_checkpoint_diagnostic(*args, **kwargs):
    from web.services.training import resume_facade as _resume_facade

    return _resume_facade._resume_checkpoint_diagnostic(*args, **kwargs)


def _runtime_meta(*args, **kwargs):
    return _training_facade()._runtime_meta(*args, **kwargs)


def _select_resume_checkpoint(*args, **kwargs):
    from web.services.training import resume_facade as _resume_facade

    return _resume_facade._select_resume_checkpoint(*args, **kwargs)


def _history_dir():
    return _training_facade().HISTORY_DIR


def _history_collections_file():
    return _training_facade().HISTORY_COLLECTIONS_FILE
