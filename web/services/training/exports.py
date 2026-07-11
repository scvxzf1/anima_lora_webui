"""Compatibility re-exports for web.services.training_service.

Domain implementations live under web.services.training.*. This module only
preserves the historical import surface used by routes, tests, and split
helpers that still resolve symbols via the facade.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from web.services.training import anomalies as _training_anomalies
from web.services.training import history_meta as _history_meta_impl
from web.services.training import launch_support as _launch_support_impl
from web.services.training import progress_parser as _progress_parser
from web.services.training import runtime_config as _runtime_config_impl
from web.services.training import service_state as _service_state_impl

inspect_continue_lora_weight = _launch_support_impl.inspect_continue_lora_weight
_continue_lora_inspection_config = _launch_support_impl._continue_lora_inspection_config
_normalize_continue_lora_info = _launch_support_impl._normalize_continue_lora_info
_continue_lora_history_meta = _launch_support_impl._continue_lora_history_meta

_resolve_training_runtime_info = _runtime_config_impl._resolve_training_runtime_info
_ensure_training_data_dirs = _runtime_config_impl._ensure_training_data_dirs

_write_config_snapshot = _launch_support_impl._write_config_snapshot
_append_continue_lora_snapshot_note = _launch_support_impl._append_continue_lora_snapshot_note
_toml_comment_string = _launch_support_impl._toml_comment_string

_load_config_file_config = _runtime_config_impl._load_config_file_config
toml_dumps_sorted = _runtime_config_impl.toml_dumps_sorted
_prepare_web_runtime_config = _runtime_config_impl._prepare_web_runtime_config
_apply_runtime_env = _runtime_config_impl._apply_runtime_env
_runtime_meta = _runtime_config_impl._runtime_meta
_delete_queue_item_runtime_dir = _runtime_config_impl._delete_queue_item_runtime_dir
_queue_item_runtime_dir_label = _runtime_config_impl._queue_item_runtime_dir_label
_queue_item_runtime_delete_dir = _runtime_config_impl._queue_item_runtime_delete_dir
_validate_queue_runtime_dir_match = _runtime_config_impl._validate_queue_runtime_dir_match
_path_is_relative_to = _runtime_config_impl._path_is_relative_to
_write_runtime_run_meta = _runtime_config_impl._write_runtime_run_meta
_read_runtime_run_meta = _runtime_config_impl._read_runtime_run_meta
_runtime_from_config_file = _runtime_config_impl._runtime_from_config_file
_clone_frozen_runtime_config = _runtime_config_impl._clone_frozen_runtime_config
_clone_runtime_dataset_rows = _runtime_config_impl._clone_runtime_dataset_rows
_runtime_dataset_child_name = _runtime_config_impl._runtime_dataset_child_name
_bool_value_for_row = _runtime_config_impl._bool_value_for_row
_prepare_runtime_nl_tag_mix_source = _runtime_config_impl._prepare_runtime_nl_tag_mix_source
_prepare_runtime_trigger_clone_source = _runtime_config_impl._prepare_runtime_trigger_clone_source
_nl_tag_mix_caption_settings = _runtime_config_impl._nl_tag_mix_caption_settings
_build_nl_tag_mix_source = _runtime_config_impl._build_nl_tag_mix_source
_classify_nl_tag_mix_samples = _runtime_config_impl._classify_nl_tag_mix_samples
_nl_tag_mix_caption_entries = _runtime_config_impl._nl_tag_mix_caption_entries
_nl_tag_mix_source_counts = _runtime_config_impl._nl_tag_mix_source_counts
_nl_tag_mix_dominant_source = _runtime_config_impl._nl_tag_mix_dominant_source
_cycle_nl_tag_entries = _runtime_config_impl._cycle_nl_tag_entries
_select_nl_tag_caption_entries = _runtime_config_impl._select_nl_tag_caption_entries
_nl_tag_mix_relative_image_path = _runtime_config_impl._nl_tag_mix_relative_image_path
_select_nl_tag_mix_samples = _runtime_config_impl._select_nl_tag_mix_samples
_copy_nl_tag_caption_sidecars = _runtime_config_impl._copy_nl_tag_caption_sidecars
_copy_runtime_dataset_dir = _runtime_config_impl._copy_runtime_dataset_dir
_is_materialized_runtime_source_dir = _runtime_config_impl._is_materialized_runtime_source_dir
_unique_runtime_dir = _runtime_config_impl._unique_runtime_dir

_history_group_meta = _history_meta_impl._history_group_meta
_fill_history_group_meta = _history_meta_impl._fill_history_group_meta
_history_run_label_from_runtime = _history_meta_impl._history_run_label_from_runtime
_history_run_label_from_path = _history_meta_impl._history_run_label_from_path
_legacy_history_group_key = _history_meta_impl._legacy_history_group_key
_legacy_history_group_label = _history_meta_impl._legacy_history_group_label
_safe_run_stem = _runtime_config_impl._safe_run_stem

_load_training_queue_state = _service_state_impl._load_training_queue_state
_read_training_queue_state = _service_state_impl._read_training_queue_state
_write_training_queue_state = _service_state_impl._write_training_queue_state
_queue_backup_file = _service_state_impl._queue_backup_file
_load_history_collection_settings = _service_state_impl._load_history_collection_settings
_normalize_history_collection_settings = _service_state_impl._normalize_history_collection_settings
_normalize_unique_string_list = _service_state_impl._normalize_unique_string_list
_normalize_config_group_order = _service_state_impl._normalize_config_group_order
_normalize_queue_failure_policy = _service_state_impl._normalize_queue_failure_policy
_normalize_queue_retry_backoff = _service_state_impl._normalize_queue_retry_backoff
_normalize_queue_max_attempts = _service_state_impl._normalize_queue_max_attempts
_normalize_queue_auto_retry = _service_state_impl._normalize_queue_auto_retry
_queue_clearable_state_label = _service_state_impl._queue_clearable_state_label
_new_queue_item_id = _service_state_impl._new_queue_item_id
_normalize_history_task_ids = _service_state_impl._normalize_history_task_ids

_mark_orphaned_running_history_tasks = _history_meta_impl._mark_orphaned_running_history_tasks
_last_history_event_ts = _history_meta_impl._last_history_event_ts
_ensure_history_average_speed_meta = _history_meta_impl._ensure_history_average_speed_meta
_history_average_speed_from_logs = _history_meta_impl._history_average_speed_from_logs
_training_progress_log_sample = _history_meta_impl._training_progress_log_sample
_history_task_dir = _history_meta_impl._history_task_dir
_load_history_task = _history_meta_impl._load_history_task
_load_history_task_summary = _history_meta_impl._load_history_task_summary
_history_log_path = _history_meta_impl._history_log_path
_history_artifact_path = _history_meta_impl._history_artifact_path
_history_task_file_artifact_path = _history_meta_impl._history_task_file_artifact_path
_history_runtime_artifact_path = _history_meta_impl._history_runtime_artifact_path
_update_history_task = _history_meta_impl._update_history_task
_history_task_ids_for_delete = _history_meta_impl._history_task_ids_for_delete
_default_history_archived = _history_meta_impl._default_history_archived
_history_task_archived = _history_meta_impl._history_task_archived
_default_preprocess_history_name = _history_meta_impl._default_preprocess_history_name
_is_legacy_auto_preprocess_name = _history_meta_impl._is_legacy_auto_preprocess_name
_fill_history_runtime_meta = _history_meta_impl._fill_history_runtime_meta
_history_snapshot_path = _history_meta_impl._history_snapshot_path

from web.services.training.resume_facade import (
    _is_web_runtime_dir,
    _list_resume_checkpoints,
    _path_exists_wrapped as _path_exists,
    _resume_checkpoint_diagnostic,
    _select_resume_checkpoint,
)

def _default_sample_config() -> dict[str, Any]:
    return {
        "enabled": False,
        "sample_prompts": "",
        "sample_prompts_exists": False,
        "sample_every_n_epochs": None,
        "sample_every_n_steps": None,
        "sample_at_first": False,
        "sample_sampler": "euler",
        "message": "未启用训练中采样",
    }

def _sample_config_from_cfg(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._sample_config_from_cfg(*args, **kwargs)

def _cli_arg_overrides(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._cli_arg_overrides(*args, **kwargs)

def _step_rate_text_from_sample(
    last: tuple[float, int] | None,
    samples: deque[float],
    step: int,
    timestamp: float,
) -> tuple[str, tuple[float, int] | None]:
    return _progress_parser.step_rate_text_from_sample(last, samples, step, timestamp)

def _resolve_display_path(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._resolve_display_path(*args, **kwargs)

def _display_project_path(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._display_project_path(*args, **kwargs)

def _absolute_display_path(value: Any) -> str:
    path = _resolve_display_path(str(value or ""))
    return str(path) if path is not None else ""

_command_has_option = _launch_support_impl._command_has_option
_command_option_value = _launch_support_impl._command_option_value
_resolve_block_swap_profile_auto_arg = _launch_support_impl._resolve_block_swap_profile_auto_arg
_resolve_block_swap_profile_auto_config = _launch_support_impl._resolve_block_swap_profile_auto_config
_resolve_memory_probe_auto_arg = _launch_support_impl._resolve_memory_probe_auto_arg
_resolve_memory_probe_auto_config = _launch_support_impl._resolve_memory_probe_auto_config
_resolve_peak_probe_auto_arg = _launch_support_impl._resolve_peak_probe_auto_arg
_resolve_peak_probe_auto_config = _launch_support_impl._resolve_peak_probe_auto_config
_resolve_auto_path_arg = _launch_support_impl._resolve_auto_path_arg
_resolve_auto_path_config = _launch_support_impl._resolve_auto_path_config
_is_history_block_swap_profile_path = _launch_support_impl._is_history_block_swap_profile_path
_is_history_memory_probe_path = _launch_support_impl._is_history_memory_probe_path
_is_history_peak_probe_path = _launch_support_impl._is_history_peak_probe_path
_is_history_artifact_path = _launch_support_impl._is_history_artifact_path

def classify_training_error(text: str) -> str:
    return _training_anomalies.classify_training_error(text)

def format_training_anomaly(status_data: dict[str, Any]) -> str | None:
    return _training_anomalies.format_training_anomaly(status_data)

def _message_with_error_hint(message: str, hint: str) -> str:
    return _training_anomalies._message_with_error_hint(message, hint)

__all__ = [
    'inspect_continue_lora_weight',
    '_continue_lora_inspection_config',
    '_normalize_continue_lora_info',
    '_continue_lora_history_meta',
    '_resolve_training_runtime_info',
    '_ensure_training_data_dirs',
    '_write_config_snapshot',
    '_append_continue_lora_snapshot_note',
    '_toml_comment_string',
    '_load_config_file_config',
    'toml_dumps_sorted',
    '_prepare_web_runtime_config',
    '_apply_runtime_env',
    '_runtime_meta',
    '_delete_queue_item_runtime_dir',
    '_queue_item_runtime_dir_label',
    '_queue_item_runtime_delete_dir',
    '_validate_queue_runtime_dir_match',
    '_path_is_relative_to',
    '_write_runtime_run_meta',
    '_read_runtime_run_meta',
    '_runtime_from_config_file',
    '_clone_frozen_runtime_config',
    '_clone_runtime_dataset_rows',
    '_runtime_dataset_child_name',
    '_bool_value_for_row',
    '_prepare_runtime_nl_tag_mix_source',
    '_prepare_runtime_trigger_clone_source',
    '_nl_tag_mix_caption_settings',
    '_build_nl_tag_mix_source',
    '_classify_nl_tag_mix_samples',
    '_nl_tag_mix_caption_entries',
    '_nl_tag_mix_source_counts',
    '_nl_tag_mix_dominant_source',
    '_cycle_nl_tag_entries',
    '_select_nl_tag_caption_entries',
    '_nl_tag_mix_relative_image_path',
    '_select_nl_tag_mix_samples',
    '_copy_nl_tag_caption_sidecars',
    '_copy_runtime_dataset_dir',
    '_is_materialized_runtime_source_dir',
    '_unique_runtime_dir',
    '_history_group_meta',
    '_fill_history_group_meta',
    '_history_run_label_from_runtime',
    '_history_run_label_from_path',
    '_legacy_history_group_key',
    '_legacy_history_group_label',
    '_safe_run_stem',
    '_load_training_queue_state',
    '_read_training_queue_state',
    '_write_training_queue_state',
    '_queue_backup_file',
    '_load_history_collection_settings',
    '_normalize_history_collection_settings',
    '_normalize_unique_string_list',
    '_normalize_config_group_order',
    '_normalize_queue_failure_policy',
    '_normalize_queue_retry_backoff',
    '_normalize_queue_max_attempts',
    '_normalize_queue_auto_retry',
    '_queue_clearable_state_label',
    '_new_queue_item_id',
    '_normalize_history_task_ids',
    '_mark_orphaned_running_history_tasks',
    '_last_history_event_ts',
    '_ensure_history_average_speed_meta',
    '_history_average_speed_from_logs',
    '_training_progress_log_sample',
    '_history_task_dir',
    '_load_history_task',
    '_load_history_task_summary',
    '_history_log_path',
    '_history_artifact_path',
    '_history_task_file_artifact_path',
    '_history_runtime_artifact_path',
    '_update_history_task',
    '_history_task_ids_for_delete',
    '_default_history_archived',
    '_history_task_archived',
    '_default_preprocess_history_name',
    '_is_legacy_auto_preprocess_name',
    '_fill_history_runtime_meta',
    '_history_snapshot_path',
    '_default_sample_config',
    '_sample_config_from_cfg',
    '_cli_arg_overrides',
    '_step_rate_text_from_sample',
    '_resolve_display_path',
    '_display_project_path',
    '_absolute_display_path',
    '_command_has_option',
    '_command_option_value',
    '_resolve_block_swap_profile_auto_arg',
    '_resolve_block_swap_profile_auto_config',
    '_resolve_memory_probe_auto_arg',
    '_resolve_memory_probe_auto_config',
    '_resolve_peak_probe_auto_arg',
    '_resolve_peak_probe_auto_config',
    '_resolve_auto_path_arg',
    '_resolve_auto_path_config',
    '_is_history_block_swap_profile_path',
    '_is_history_memory_probe_path',
    '_is_history_peak_probe_path',
    '_is_history_artifact_path',
    'classify_training_error',
    'format_training_anomaly',
    '_message_with_error_hint',
    '_is_web_runtime_dir',
    '_list_resume_checkpoints',
    '_path_exists',
    '_resume_checkpoint_diagnostic',
    '_select_resume_checkpoint'
]
