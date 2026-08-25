"""Delegated training service methods for history.

Compatibility facade. Implementation lives in:

- ``history_runtime``: facade accessors shared by history domains
- ``history_resume``: resume payload / checkpoint selection
- ``history_ops``: list/update/delete/lifecycle helpers

The public ``TrainingService`` class keeps the same method names and delegates
here so HTTP routes and WebSocket payloads remain unchanged.
"""

from __future__ import annotations

from web.services.training.history_ops import (
    _append_history_jsonl,
    _batch_delete_history_tasks,
    _finish_history_task,
    _plan_history_delete,
    _reserve_history_task_dir,
    _start_history_task,
    batch_update_history_tasks,
    delete_history_task,
    get_config_group_timeline,
    get_history_artifact_path,
    get_history_collection_settings,
    get_history_log_path,
    get_history_task,
    get_history_task_summary,
    list_history_tasks,
    save_history_collection_settings,
    update_history_task,
)
from web.services.training.history_logs import find_history_log_match, get_history_log_page
from web.services.training.history_resume import (
    _annotate_resume_checkpoints,
    _build_resume_payload,
    _clone_resume_runtime,
    _ensure_resume_checkpoint_available,
    _positive_resume_duration_int,
    _resume_checkpoint_estimate,
    _resume_duration_override_requested,
    _resume_unavailable_reason,
    get_resume_options,
    resume_from_history_task,
)
from web.services.training.history_runtime import (
    _batch_archive_history_tasks,
    _batch_set_history_group,
    _build_config_group_timeline,
    _clone_frozen_runtime_config,
    _default_history_archived,
    _delete_history_tasks,
    _display_project_path,
    _history_collections_file,
    _history_delete_run_key,
    _history_delete_task_preview,
    _history_dir,
    _history_runtime_delete_dirs_for_tasks,
    _json_safe_training_payload,
    _list_history_tasks,
    _list_resume_checkpoints,
    _path_exists,
    _queue_runtime_delete_blockers,
    _resolve_display_path,
    _resume_checkpoint_diagnostic,
    _runtime_meta,
    _select_resume_checkpoint,
    _training_facade,
)

__all__ = [
    "resume_from_history_task",
    "_build_resume_payload",
    "list_history_tasks",
    "get_history_task",
    "get_history_task_summary",
    "get_history_log_path",
    "get_history_log_page",
    "find_history_log_match",
    "get_history_artifact_path",
    "get_config_group_timeline",
    "get_history_collection_settings",
    "save_history_collection_settings",
    "get_resume_options",
    "update_history_task",
    "batch_update_history_tasks",
    "delete_history_task",
    "_batch_delete_history_tasks",
    "_plan_history_delete",
    "_reserve_history_task_dir",
    "_start_history_task",
    "_finish_history_task",
    "_append_history_jsonl",
]
