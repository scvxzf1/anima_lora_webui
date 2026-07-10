"""Shared runtime for training preflight checks.

Owns path roots, facade dependency sync, and thin path wrappers so domain
modules can stay free of the config_service import cycle at module import time.
"""

from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any

from library.env import expand_env_vars, get_configs_root
from web.services.config import paths as _config_paths


def _missing_facade_dependency(*args, **kwargs):
    raise RuntimeError("preflight config helper was called before facade sync")


ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = get_configs_root()


def _safe_resolve(rel_path: str) -> Path | None:
    return _config_paths.safe_resolve(rel_path, root=ROOT, configs_dir=CONFIGS_DIR)


def _resolve_project_path(value: str) -> Path:
    return _config_paths.resolve_display_path(
        value,
        root=ROOT,
        configs_dir=CONFIGS_DIR,
        expand_env_vars_fn=expand_env_vars,
    )


def _display_path(path: Path) -> str:
    return _config_paths.display_path(path, root=ROOT, configs_dir=CONFIGS_DIR)


def _normalize_config_rel_path(rel_path: str) -> str:
    return _config_paths.normalize_config_rel_path(rel_path)


# Facade-filled helpers. Keep placeholders so direct imports fail loudly before sync.
_is_blank_output_name = _missing_facade_dependency
_dataset_config_path_from_cfg = _missing_facade_dependency
_bool_value = _missing_facade_dependency
_positive_int_or_none = _missing_facade_dependency
apply_auto_data_dirs = _missing_facade_dependency
load_merged_config = _missing_facade_dependency
_dataset_rows_for_estimate = _missing_facade_dependency
_normalize_path_pattern = _missing_facade_dependency
_dataset_image_files = _missing_facade_dependency
_caption_detection_counts_text = _missing_facade_dependency
_normalize_trigger_clone = _missing_facade_dependency
_count_source_images = _missing_facade_dependency
_nl_tag_mix_enabled = _missing_facade_dependency
_nl_tag_mix_caption_counts = _missing_facade_dependency
resolve_output_root = _missing_facade_dependency
_display_settings_path = _missing_facade_dependency
save_raw_file = _missing_facade_dependency
load_raw_file = _missing_facade_dependency
delete_raw_file = _missing_facade_dependency
patch_raw_file_values = _missing_facade_dependency
preview_raw_file_patch = _missing_facade_dependency
get_config_file_meta = _missing_facade_dependency
list_config_file_groups = _missing_facade_dependency
move_config_file_to_group = _missing_facade_dependency
_inspect_network_weight = _missing_facade_dependency
LOGGER = None
GUI_METHODS_DIR = CONFIGS_DIR / "gui-methods"
IMPORTED_CONFIGS_DIR = CONFIGS_DIR / "imported"
PRESETS_FILE = CONFIGS_DIR / "presets.toml"
WEB_FILE_GROUPS_FILE = CONFIGS_DIR / "web-file-groups.toml"
WEB_USER_LOCKS_FILE = CONFIGS_DIR / "web-user-locks.toml"
DATASET_PRESETS_DIR = CONFIGS_DIR / "datasets"

_SYNC_NAMES = (
    "ROOT",
    "CONFIGS_DIR",
    "GUI_METHODS_DIR",
    "IMPORTED_CONFIGS_DIR",
    "PRESETS_FILE",
    "WEB_FILE_GROUPS_FILE",
    "WEB_USER_LOCKS_FILE",
    "DATASET_PRESETS_DIR",
    "resolve_output_root",
    "_display_settings_path",
    "save_raw_file",
    "load_raw_file",
    "delete_raw_file",
    "patch_raw_file_values",
    "preview_raw_file_patch",
    "get_config_file_meta",
    "list_config_file_groups",
    "move_config_file_to_group",
    "_inspect_network_weight",
    "LOGGER",
    "_is_blank_output_name",
    "_dataset_config_path_from_cfg",
    "_bool_value",
    "_positive_int_or_none",
    "apply_auto_data_dirs",
    "load_merged_config",
    "_dataset_rows_for_estimate",
    "_normalize_path_pattern",
    "_dataset_image_files",
    "_caption_detection_counts_text",
    "_normalize_trigger_clone",
    "_count_source_images",
    "_nl_tag_mix_enabled",
    "_nl_tag_mix_caption_counts",
)


def _sync_from_facade() -> None:
    from web.services import config_service as _facade
    import sys

    if hasattr(_facade, "_sync_legacy_from_facade"):
        _facade._sync_legacy_from_facade()

    # Prefer the public preflight facade for exported-name protection.
    facade_mod = sys.modules.get("web.services.config.preflight")
    exported_names = set(getattr(facade_mod, "__all__", ()) or ()) if facade_mod is not None else set()

    for name in _SYNC_NAMES:
        if not hasattr(_facade, name):
            continue
        value = getattr(_facade, name)
        if name not in exported_names:
            globals()[name] = value
            if facade_mod is not None:
                setattr(facade_mod, name, value)
            # Keep domain modules that imported these names via re-export in sync
            # only for path roots / common helpers used across domains.
            for module_name in (
                "web.services.config.preflight_paths",
                "web.services.config.preflight_compat",
                "web.services.config.preflight_history",
                "web.services.config.preflight_dataset_checks",
            ):
                mod = sys.modules.get(module_name)
                if mod is not None and hasattr(mod, name):
                    setattr(mod, name, value)


def _exported(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        _sync_from_facade()
        return fn(*args, **kwargs)

    return wrapper


def _nonnegative_int_value(value: Any, fallback: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed >= 0 else fallback


def _nonnegative_float_value(value: Any, fallback: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed >= 0.0 else fallback


GLOBAL_MODEL_PATH_KEYS = (
    "pretrained_model_name_or_path",
    "qwen3",
    "vae",
)
