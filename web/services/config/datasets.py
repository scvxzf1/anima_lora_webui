"""Dataset presets, dataset editor, and dataset runtime document helpers.

Compatibility facade. Implementation lives in:

- ``dataset_rows`` / ``dataset_editor`` / ``dataset_presets_api``
- ``dataset_media`` for image listing, preview metadata, and counts
- ``dataset_nl_tag`` for natural-language / tag mix caption helpers

Routes/tests should keep importing from this module or ``config_service``.
"""

from __future__ import annotations

import logging
from functools import wraps
from pathlib import Path
from typing import Any

import toml

from library.env import expand_env_vars, get_configs_root, load_dotenv
from web.services.config import paths as _config_paths
from web.services.config.common import (
    _bool_value,
    _nonnegative_float,
    _nonnegative_int,
    _positive_int,
)
from web.services.config.metadata import (
    CAPTION_SOURCE_AUTO,
    CAPTION_SOURCE_CAPTIONS_JSON,
    CAPTION_SOURCE_JSON,
    CAPTION_SOURCE_MODE_LABELS,
    CAPTION_SOURCE_TXT,
    DATASET_CAPTION_MAX_CHARS,
    DATASET_IMAGE_EXTS,
    DATASET_PREVIEW_LIMIT,
    DATASET_SETTING_KEYS,
    DEFAULT_LORA_CACHE_DIR,
    DEFAULT_NL_TAG_MIX_TAG_RATIO,
    DEFAULT_RESIZED_IMAGE_DIR,
    HIDDEN_DATASET_PRESET_FILES,
    NL_TAG_MIX_ATTR_KEY,
    NL_TAG_MIX_CLASSIFICATION_METHOD,
    PREPROCESS_DATASET_SETTING_ORDER,
    RUNTIME_PREPROCESS_ATTR_KEY,
    SYSTEM_DATASET_PRESET_FILES,
    TRIGGER_CLONE_ATTR_KEY,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = get_configs_root()
GUI_METHODS_DIR = CONFIGS_DIR / "gui-methods"
IMPORTED_CONFIGS_DIR = CONFIGS_DIR / "imported"
PRESETS_FILE = CONFIGS_DIR / "presets.toml"
WEB_FILE_GROUPS_FILE = CONFIGS_DIR / "web-file-groups.toml"
WEB_USER_LOCKS_FILE = CONFIGS_DIR / "web-user-locks.toml"
DATASET_PRESETS_DIR = CONFIGS_DIR / "datasets"

load_dotenv()

LOGGER = logging.getLogger(__name__)


def _config_facade():
    from web.services import config_service as _facade

    return _facade


def save_raw_file(*args, **kwargs):
    return _config_facade().save_raw_file(*args, **kwargs)


def load_raw_file(*args, **kwargs):
    return _config_facade().load_raw_file(*args, **kwargs)


def delete_raw_file(*args, **kwargs):
    return _config_facade().delete_raw_file(*args, **kwargs)


def patch_raw_file_values(*args, **kwargs):
    return _config_facade().patch_raw_file_values(*args, **kwargs)


def preview_raw_file_patch(*args, **kwargs):
    return _config_facade().preview_raw_file_patch(*args, **kwargs)


def get_config_file_meta(*args, **kwargs):
    return _config_facade().get_config_file_meta(*args, **kwargs)


def list_config_file_groups(*args, **kwargs):
    return _config_facade().list_config_file_groups(*args, **kwargs)


def move_config_file_to_group(*args, **kwargs):
    return _config_facade().move_config_file_to_group(*args, **kwargs)


def _inspect_network_weight(*args, **kwargs):
    return _config_facade()._inspect_network_weight(*args, **kwargs)


def load_merged_config(*args, **kwargs):
    return _config_facade().load_merged_config(*args, **kwargs)


def apply_auto_data_dirs(*args, **kwargs):
    return _config_facade().apply_auto_data_dirs(*args, **kwargs)


def _prepare_raw_file_patch(*args, **kwargs):
    return _config_facade()._prepare_raw_file_patch(*args, **kwargs)


def _load_training_config_for_web_run(*args, **kwargs):
    return _config_facade()._load_training_config_for_web_run(*args, **kwargs)


def _normalize_config_rel_path(rel_path: str) -> str:
    return _config_paths.normalize_config_rel_path(rel_path)


def _safe_resolve(rel_path: str) -> Path | None:
    return _config_paths.safe_resolve(rel_path, root=ROOT, configs_dir=CONFIGS_DIR)


def _safe_config_subdir(subdir: str) -> Path | None:
    return _config_paths.safe_config_subdir(subdir, configs_dir=CONFIGS_DIR)


def _resolve_project_path(value: str) -> Path:
    return _config_paths.resolve_display_path(
        value,
        root=ROOT,
        configs_dir=CONFIGS_DIR,
        expand_env_vars_fn=expand_env_vars,
    )


def _display_path(path: Path) -> str:
    return _config_paths.display_path(path, root=ROOT, configs_dir=CONFIGS_DIR)


def _derived_data_dir(source_path: Path, suffix: str) -> Path:
    parent = source_path.parent if source_path.name else source_path
    name = source_path.name or "dataset"
    return (parent / f"{name}_{suffix}").resolve()


from web.services.config.dataset_rows import (  # noqa: E402,F401
    _build_dataset_config_doc,
    _dataset_defaults_from_config,
    _dataset_defaults_from_dataset,
    _dataset_path_value,
    _dataset_row_settings,
    _dataset_rows_from_config,
    _dataset_summary_from_rows,
    _dataset_training_defaults,
    _ensure_training_dataset_rows,
    _fill_missing_dataset_row_settings,
    _first_dataset_settings,
    _first_dataset_value,
    _first_training_dataset_row,
    _nl_tag_mix_enabled,
    _normalize_dataset_defaults,
    _normalize_dataset_row_settings,
    _normalize_dataset_rows,
    _normalize_nl_tag_mix,
    _normalize_path_pattern,
    _normalize_preprocess_dataset_settings,
    _normalize_trigger_clone,
    _preprocess_settings_for_runtime_attrs,
    _preprocess_settings_from_custom_attributes,
    _safe_file_stem,
    _single_dataset_config_from_cfg,
    _trigger_clone_should_persist,
)
from web.services.config.dataset_editor import (  # noqa: E402,F401
    _dataset_config_path_from_cfg,
    _dataset_config_rel_path,
    _is_allowed_dataset_config_path,
    _restore_dataset_config_after_failed_train_patch,
    _training_config_rel_path,
    load_dataset_editor,
    save_dataset_editor,
)
from web.services.config.dataset_preset_paths import (  # noqa: E402,F401
    _is_dataset_preset_readonly,
    _normalize_dataset_preset_path,
)
from web.services.config.dataset_presets_api import (  # noqa: E402,F401
    _dataset_preset_groups_for_ui,
    _dataset_preset_summary,
    _is_dataset_group_for_ui,
    apply_dataset_preset_to_training_config,
    delete_dataset_preset,
    diagnose_dataset_presets,
    import_dataset_preset,
    list_dataset_preset_images,
    list_dataset_presets,
    load_dataset_preset,
    resolve_dataset_preview_image,
    save_dataset_preset,
    save_dataset_preset_as,
)
from web.services.config.dataset_media import (  # noqa: E402,F401
    _caption_detection_counts_text,
    _caption_extension_for_detected_mode,
    _caption_source_mode_label,
    _count_images,
    _count_source_images,
    _dataset_caption_detection_summary,
    _dataset_caption_meta,
    _dataset_image_dimensions,
    _dataset_image_files,
    _dataset_image_preview_meta,
    _dataset_num_repeats,
    _dataset_preview_empty_message,
    _format_caption_preview_text,
    _list_dataset_image_files,
)
from web.services.config.dataset_nl_tag import (  # noqa: E402,F401
    _classify_nl_tag_caption_text,
    _nl_tag_mix_available_count,
    _nl_tag_mix_caption_counts,
    _nl_tag_mix_caption_path_and_text,
    _nl_tag_mix_caption_source,
    _nl_tag_mix_image_files,
)
from web.services.config.file_groups import _lock_reason_message  # noqa: E402,F401

_SYNC_NAMES = (
    "ROOT",
    "CONFIGS_DIR",
    "DATASET_PRESETS_DIR",
    "LOGGER",
    "load_raw_file",
    "save_raw_file",
    "delete_raw_file",
    "patch_raw_file_values",
    "preview_raw_file_patch",
    "get_config_file_meta",
    "list_config_file_groups",
    "move_config_file_to_group",
)


def _sync_from_facade() -> None:
    from web.services import config_service as _facade
    from web.services.config import dataset_media as _dataset_media

    _exported_names = set(globals().get("__all__", ()) or ())
    for _name in _SYNC_NAMES:
        if not hasattr(_facade, _name):
            continue
        _value = getattr(_facade, _name)
        if _name not in _exported_names:
            globals()[_name] = _value
    # Keep media helpers on the same path roots as this facade.
    _dataset_media.ROOT = ROOT
    _dataset_media.CONFIGS_DIR = CONFIGS_DIR


def _exported(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        _sync_from_facade()
        return fn(*args, **kwargs)

    return wrapper


def _dataset_rows_for_estimate(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_path = _dataset_config_path_from_cfg(cfg)
    if dataset_path and dataset_path.exists():
        try:
            data = toml.loads(dataset_path.read_text(encoding="utf-8"))
        except toml.TomlDecodeError:
            data = _single_dataset_config_from_cfg(cfg)
    else:
        data = _single_dataset_config_from_cfg(cfg)
    return _dataset_rows_from_config(data, cfg)


__all__ = [
    "list_dataset_presets",
    "diagnose_dataset_presets",
    "load_dataset_preset",
    "save_dataset_preset",
    "save_dataset_preset_as",
    "import_dataset_preset",
    "delete_dataset_preset",
    "apply_dataset_preset_to_training_config",
    "list_dataset_preset_images",
    "resolve_dataset_preview_image",
    "load_dataset_editor",
    "save_dataset_editor",
    "_dataset_config_path_from_cfg",
    "_dataset_rows_for_estimate",
    "_dataset_rows_from_config",
    "_normalize_dataset_rows",
    "_normalize_dataset_defaults",
    "_normalize_nl_tag_mix",
    "_normalize_trigger_clone",
    "_normalize_path_pattern",
    "_build_dataset_config_doc",
    "_nl_tag_mix_caption_source",
    "_nl_tag_mix_image_files",
    "_classify_nl_tag_caption_text",
]

for _name in __all__:
    globals()[_name] = _exported(globals()[_name])
