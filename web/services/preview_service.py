"""Preview image discovery and Web UI preview settings.

Listing/delete/weight listing logic lives under ``web.services.preview``;
this module remains the stable public import surface for routes, services,
and tests.
"""

from __future__ import annotations

from typing import Any

import toml

from web.services import settings_service
from web.services.atomic_io import atomic_write_text
from web.services.preview import common as _preview_common
from web.services.preview import images as _preview_images
from web.services.preview import weights as _preview_weights

ROOT = _preview_common.ROOT
CONFIGS_DIR = _preview_common.CONFIGS_DIR
SETTINGS_FILE = _preview_common.SETTINGS_FILE
IMAGE_EXTS = _preview_common.IMAGE_EXTS
WEIGHT_EXTS = _preview_common.WEIGHT_EXTS
DEFAULT_TRAINING_DIR = _preview_common.DEFAULT_TRAINING_DIR
DEFAULT_INFERENCE_DIR = _preview_common.DEFAULT_INFERENCE_DIR
DEFAULT_OUTPUT_ROOT = _preview_common.DEFAULT_OUTPUT_ROOT
MAX_IMAGE_LIMIT = _preview_common.MAX_IMAGE_LIMIT
MAX_WEIGHT_LIMIT = _preview_common.MAX_WEIGHT_LIMIT
SAMPLE_NAME_RE = _preview_common.SAMPLE_NAME_RE

_preview_task_label = _preview_common._preview_task_label
_load_settings = _preview_common._load_settings
_load_raw_settings = _preview_common._load_raw_settings
_read_safetensors_metadata = _preview_common._read_safetensors_metadata
_normalize_optional_preview_dir = _preview_common._normalize_optional_preview_dir
_normalize_preview_dir = _preview_common._normalize_preview_dir
_normalize_project_dir = _preview_common._normalize_project_dir
_normalize_project_file = _preview_common._normalize_project_file
_resolve_project_path = _preview_common._resolve_project_path
_resolve_preview_dir = _preview_common._resolve_preview_dir
_resolve_preview_file = _preview_common._resolve_preview_file
_resolve_weight_file = _preview_common._resolve_weight_file
_resolve_allowed_sample_dir = _preview_common._resolve_allowed_sample_dir
_allowed_external_preview_dirs = _preview_common._allowed_external_preview_dirs
_allowed_weight_dirs = _preview_common._allowed_weight_dirs
_resolve_training_output_dir = _preview_common._resolve_training_output_dir
_resolve_display_path = _preview_common._resolve_display_path
_int_or_none = _preview_common._int_or_none
_float_or_none = _preview_common._float_or_none
_display_path = _preview_common._display_path
_latest_runtime_sample_dir = _preview_common._latest_runtime_sample_dir
_runtime_sample_sort_ts = _preview_common._runtime_sample_sort_ts
_resolve_global_output_root = _preview_common._resolve_global_output_root

list_preview_images = _preview_images.list_preview_images
delete_preview_images = _preview_images.delete_preview_images
list_config_group_preview_images = _preview_images.list_config_group_preview_images
resolve_preview_image = _preview_images.resolve_preview_image
_normalize_preview_days = _preview_images._normalize_preview_days
_filter_preview_candidates_by_days = _preview_images._filter_preview_candidates_by_days
_task_image_match_score = _preview_images._task_image_match_score
_image_meta = _preview_images._image_meta
_sample_image_meta = _preview_images._sample_image_meta
_load_sample_prompt_entries = _preview_images._load_sample_prompt_entries
_parse_prompt_line = _preview_images._parse_prompt_line
_parse_prompt_toml = _preview_images._parse_prompt_toml
_parse_sample_image_name = _preview_images._parse_sample_image_name
_training_step_index = _preview_images._training_step_index
_empty_listing = _preview_images._empty_listing
_preview_settings_meta = _preview_images._preview_settings_meta
_normalize_preview_delete_files = _preview_images._normalize_preview_delete_files
_ensure_preview_delete_target = _preview_images._ensure_preview_delete_target
_preview_source_delete_label = _preview_images._preview_source_delete_label
_preview_delete_message = _preview_images._preview_delete_message
_preview_empty_message = _preview_images._preview_empty_message
_training_preview_label = _preview_images._training_preview_label

list_training_weights = _preview_weights.list_training_weights
list_config_group_training_weights = _preview_weights.list_config_group_training_weights
resolve_training_weight = _preview_weights.resolve_training_weight
_group_weight_match_score = _preview_weights._group_weight_match_score
_group_weight_scope_label = _preview_weights._group_weight_scope_label
_weight_meta = _preview_weights._weight_meta
_weight_kind = _preview_weights._weight_kind
_weight_sort_key = _preview_weights._weight_sort_key
_weight_scope = _preview_weights._weight_scope
_empty_weights_listing = _preview_weights._empty_weights_listing


def get_preview_settings(
    current_task_sample_dir: str | None = None,
    *,
    allow_latest_fallback: bool = True,
) -> dict[str, Any]:
    settings = _load_settings()
    task_dir = _normalize_optional_preview_dir(current_task_sample_dir)
    latest_run = _latest_runtime_sample_dir()
    if task_dir:
        training_dir = task_dir
        training_source = "current_task"
    elif allow_latest_fallback and latest_run:
        training_dir = latest_run["sample_dir"]
        training_source = "latest_run"
    elif not allow_latest_fallback:
        training_dir = ""
        training_source = "selected_task_missing"
    else:
        training_dir = settings["training_dir"]
        training_source = "saved_default"
    output_root = _resolve_global_output_root()
    return {
        "ok": True,
        "training_dir": settings["training_dir"],
        "inference_dir": settings["inference_dir"],
        "custom_dir": settings["custom_dir"],
        "training_output_root": settings_service.display_path(output_root),
        "current_task_sample_dir": task_dir,
        "latest_run_dir": latest_run["run_dir"] if latest_run else "",
        "latest_run_sample_dir": latest_run["sample_dir"] if latest_run else "",
        "effective_training_dir": training_dir,
        "effective_training_source": training_source,
        "defaults": {
            "training_dir": DEFAULT_TRAINING_DIR,
            "inference_dir": DEFAULT_INFERENCE_DIR,
            "custom_dir": "",
            "training_output_root": DEFAULT_OUTPUT_ROOT,
        },
    }


def save_preview_settings(data: dict[str, Any]) -> dict[str, Any]:
    current = _load_settings()
    next_settings = {
        "training_dir": _normalize_project_dir(
            data.get("training_dir", current["training_dir"]) or DEFAULT_TRAINING_DIR,
            allow_empty=False,
        ),
        "inference_dir": _normalize_preview_dir(
            data.get("inference_dir", current["inference_dir"]) or DEFAULT_INFERENCE_DIR,
            allow_empty=False,
        ),
        "custom_dir": _normalize_preview_dir(data.get("custom_dir", current["custom_dir"]) or "", allow_empty=True),
    }
    raw = _load_raw_settings()
    raw["preview"] = next_settings
    atomic_write_text(SETTINGS_FILE, toml.dumps(raw), encoding="utf-8")
    return {"ok": True, "message": "预览图路径设置已保存", **next_settings}
