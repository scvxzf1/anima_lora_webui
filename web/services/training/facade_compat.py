"""Extra historical imports exposed by web.services.training_service.

These are not domain implementations; they exist so older tests and helpers can still
``import web.services.training_service as training_service`` and monkeypatch or import
common helpers from the facade module.
"""

from __future__ import annotations

from typing import Any, Callable

COMPAT_EXPORTS: dict[str, Callable[[], Any]] = {}


def _register(name: str, loader: Callable[[], Any]) -> None:
    COMPAT_EXPORTS[name] = loader


def resolve_compat_export(name: str) -> Any:
    loader = COMPAT_EXPORTS[name]
    return loader()


def _lazy_from(module: str, attr: str):
    def _load():
        import importlib
        return getattr(importlib.import_module(module), attr)
    return _load


# stdlib / env surfaces used by tests via training_service.<name>
import os as _os
import shutil as _shutil
import sys as _sys
import time as _time
from datetime import datetime as _datetime

import psutil as _psutil

_register("os", lambda: _os)
_register("shutil", lambda: _shutil)
_register("sys", lambda: _sys)
_register("time", lambda: _time)
_register("datetime", lambda: _datetime)
_register("psutil", lambda: _psutil)

# Frequently imported helpers that used to live as re-exports on the facade.
_register("_apply_gpu_whitelist", _lazy_from("web.services.training.gpu", "apply_gpu_whitelist"))
_register("_normalize_gpu_whitelist", _lazy_from("web.services.training.gpu", "normalize_gpu_whitelist"))
_register("_get_gpu_stats", _lazy_from("web.services.training.gpu_async", "get_gpu_stats"))
_register("_list_available_gpus", _lazy_from("web.services.training.gpu_async", "list_available_gpus"))
_register("resolve_output_root", _lazy_from("web.services.settings_service", "resolve_output_root"))
_register("DynamicPath", _lazy_from("web.services._dynamic_path", "DynamicPath"))
_register("accelerate_training_command_prefix", _lazy_from("library.runtime.launch", "accelerate_training_command_prefix"))
_register("get_training_history_root", _lazy_from("library.env", "get_training_history_root"))
_register("get_training_queue_root", _lazy_from("library.env", "get_training_queue_root"))

# Common / storage / constants historically imported from the facade.
for _mod, _names in {
    "web.services.training.common": (
        "HISTORY_ARTIFACT_FILES",
        "HISTORY_RUNTIME_ARTIFACT_FIELDS",
        "RUN_META_FILE",
        "TRAINING_PROGRESS_LOG_RE",
        "_clean_history_text",
        "_float_or_none",
        "_format_ts",
        "_int_or_none",
        "_positive_int_or_none",
        "_safe_task_id",
    ),
    "web.services.training.constants": (
        "HISTORY_AVERAGE_SPEED_VERSION",
        "HISTORY_COLLECTIONS_FILE",
        "HISTORY_DIR",
        "MAX_HISTORY_DETAIL_LOG_RECORDS",
        "MAX_HISTORY_DETAIL_SYSTEM_RECORDS",
        "MAX_HISTORY_ITEMS",
        "MAX_LOG_RECORDS",
        "MAX_QUEUE_ITEMS",
        "MAX_TIMELINE_LOG_RECORDS",
        "MAX_TIMELINE_METRIC_RECORDS",
        "OUTPUT_READ_SIZE",
        "PROGRESS_RATE_SAMPLE_WINDOW",
        "SYSTEM_MONITOR_INTERVAL_SECONDS",
        "QUEUE_CLEARABLE_STATE_LABELS",
        "QUEUE_CLEARABLE_STATES",
        "QUEUE_DIR",
        "QUEUE_FAILURE_POLICIES",
        "QUEUE_FILE",
        "QUEUE_TERMINAL_STATES",
        "ROOT",
        "RUNTIME_META_KEYS",
        "TQDM_RE",
    ),
    "web.services.training.storage": (
        "_count_jsonl",
        "_read_json",
        "_read_json_object",
        "_read_jsonl",
        "_read_jsonl_limited",
        "_read_text_file",
        "_write_json",
        "_write_json_atomic",
    ),
    "web.services.training.history_batch": (
        "batch_archive_history_tasks",
        "batch_set_history_group",
        "bound_history_task_ids",
    ),
    "web.services.training.resume": (
        "_is_transient_resume_state_dir",
        "_resume_state_integrity",
        "_resume_state_integrity_unavailable_reason",
    ),
}.items():
    for _name in _names:
        # history_batch public names were imported with leading underscore aliases before.
        alias = {
            "batch_archive_history_tasks": "_batch_archive_history_tasks",
            "batch_set_history_group": "_batch_set_history_group",
            "bound_history_task_ids": "_bound_history_task_ids",
        }.get(_name, _name)
        _register(alias, _lazy_from(_mod, _name))

# history_store / timeline private helpers that tests may still resolve via facade.
for _name in (
    "_delete_history_task",
    "_delete_history_tasks",
    "_history_delete_run_key",
    "_history_delete_task_preview",
    "_history_meta_paths",
    "_history_runtime_delete_dirs_for_tasks",
    "_history_summary",
    "_is_deleting_history_dir",
    "_linked_preprocess_task_for_training",
    "_linked_preprocess_tasks_for_training",
    "_list_history_tasks",
    "_queue_runtime_delete_blockers",
    "_repair_history_meta",
    "_reserve_deleting_history_dir",
    "_safe_history_summary",
    "_sync_bound_history_collection_groups",
):
    _register(_name, _lazy_from("web.services.training.history_store", _name))

for _name in (
    "_build_config_group_timeline",
    "_format_step_rate",
    "_history_config_group",
    "_history_metrics_for_task",
    "_is_finite_number",
    "_metrics_from_progress_jsonl",
):
    _register(_name, _lazy_from("web.services.training.history_timeline", _name))

for _name in (
    "_clean_output_record",
    "_extract_float_metric",
    "_first_float_field",
    "_first_record_separator",
    "_json_safe_training_payload",
    "_live_metric_key",
    "_metric_from_progress_jsonl_event",
    "_progress_event_key",
    "_progress_event_loss",
    "_progress_event_lr",
    "_progress_event_wall_ts",
    "_progress_event_wall_ts_from_started_at",
):
    _register(_name, _lazy_from("web.services.training.live_utils", _name))

# settings display helper alias
_register("_display_settings_path", _lazy_from("web.services.settings_service", "display_path"))

# config_service helpers previously imported into the facade for convenience.
for _name in (
    "NL_TAG_MIX_CLASSIFICATION_METHOD",
    "_build_dataset_config_doc",
    "_classify_nl_tag_caption_text",
    "_dataset_rows_for_estimate",
    "_nl_tag_mix_caption_source",
    "_nl_tag_mix_image_files",
    "_normalize_nl_tag_mix",
    "_normalize_path_pattern",
    "_normalize_trigger_clone",
    "apply_global_model_path_defaults",
    "preflight_training_config",
    "training_sample_sampler_status",
):
    _register(_name, _lazy_from("web.services.config_service", _name))

for _name in ("DATASET_CAPTION_EXTS", "DATASET_IMAGE_EXTS"):
    _register(_name, _lazy_from("web.services.config.metadata", _name))

for _name in (
    "CAPTION_SOURCE_CAPTIONS_JSON",
    "CAPTIONS_JSON_FILE",
    "normalize_caption_source_mode",
):
    _register(_name, _lazy_from("library.preprocess.captions", _name))
