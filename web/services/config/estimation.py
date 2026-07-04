"""Training step estimation for WebUI config forms.

This module is loaded by ``web.services.config_service`` as part of the
compatibility facade.  It keeps facade access lazy so the module can also be
imported directly without pulling in the legacy facade.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

def _missing_facade_dependency(*args, **kwargs):
    raise RuntimeError("config estimation helper was called before facade sync")


DEFAULT_MAX_TRAIN_STEPS = 0
_load_training_config_for_web_run = _missing_facade_dependency
_normalize_config_rel_path = _missing_facade_dependency
_dataset_rows_for_estimate = _missing_facade_dependency
_resolve_project_path = _missing_facade_dependency
_display_path = _missing_facade_dependency
_positive_int = _missing_facade_dependency
_positive_float = _missing_facade_dependency
_nonnegative_int = _missing_facade_dependency
_bool_value = _missing_facade_dependency
_normalize_nl_tag_mix = _missing_facade_dependency
_normalize_trigger_clone = _missing_facade_dependency
_normalize_path_pattern = _missing_facade_dependency
_nl_tag_mix_available_count = _missing_facade_dependency
_count_source_images = _missing_facade_dependency
_count_images = _missing_facade_dependency

_SYNC_NAMES = (
    "ROOT",
    "CONFIGS_DIR",
    "GUI_METHODS_DIR",
    "IMPORTED_CONFIGS_DIR",
    "PRESETS_FILE",
    "WEB_FILE_GROUPS_FILE",
    "WEB_USER_LOCKS_FILE",
    "DATASET_PRESETS_DIR",
    "DEFAULT_MAX_TRAIN_STEPS",
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
    "_load_training_config_for_web_run",
    "_normalize_config_rel_path",
    "_dataset_rows_for_estimate",
    "_resolve_project_path",
    "_display_path",
    "_positive_int",
    "_positive_float",
    "_nonnegative_int",
    "_bool_value",
    "_normalize_nl_tag_mix",
    "_normalize_trigger_clone",
    "_normalize_path_pattern",
    "_nl_tag_mix_available_count",
    "_count_source_images",
    "_count_images",
)

_LEGACY_STATE_NAMES = (
    "ROOT",
    "CONFIGS_DIR",
    "GUI_METHODS_DIR",
    "IMPORTED_CONFIGS_DIR",
    "PRESETS_FILE",
    "WEB_FILE_GROUPS_FILE",
    "WEB_USER_LOCKS_FILE",
    "DATASET_PRESETS_DIR",
    "DEFAULT_MAX_TRAIN_STEPS",
    "resolve_output_root",
    "_display_settings_path",
    "LOGGER",
)


def _sync_from_facade() -> None:
    from web.services import config_service as _facade

    _exported_names = set(globals().get("__all__", ()))
    _legacy_module = getattr(_facade, "_legacy", None)
    for _name in _SYNC_NAMES:
        if not hasattr(_facade, _name):
            continue
        _value = getattr(_facade, _name)
        if _name not in _exported_names:
            globals()[_name] = _value
        if _legacy_module is not None and _name in _LEGACY_STATE_NAMES:
            setattr(_legacy_module, _name, _value)


def _exported(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        _sync_from_facade()
        return fn(*args, **kwargs)

    return wrapper

__all__ = ['estimate_training_steps']

def estimate_training_steps(
    variant: str,
    preset: str,
    methods_subdir: str = "gui-methods",
    *,
    config_file: str | None = None,
    dataset_config: str | None = None,
) -> dict[str, Any]:
    cfg = _load_training_config_for_web_run(
        variant,
        preset,
        methods_subdir,
        config_file=config_file,
    )
    if dataset_config is not None:
        dataset_rel = _normalize_config_rel_path(str(dataset_config or ""))
        if dataset_rel:
            cfg["dataset_config"] = dataset_rel
        else:
            cfg.pop("dataset_config", None)
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    dataset_rows = _dataset_rows_for_estimate(cfg)
    detail_rows: list[dict[str, Any]] = []
    source_images = 0
    resized_images = 0
    train_images = 0
    weighted_images = 0
    dataset_repeats = 0
    for idx, row in enumerate(dataset_rows):
        source_dir = _resolve_project_path(str(row.get("source_dir") or ""))
        resized_dir = _resolve_project_path(str(row.get("image_dir") or ""))
        repeats = _positive_int(row.get("num_repeats"), 1)
        mix = _normalize_nl_tag_mix(row.get("nl_tag_mix"))
        trigger_clone = _normalize_trigger_clone(row.get("trigger_clone"))
        recursive = _bool_value(row.get("recursive"), True)
        path_pattern = _normalize_path_pattern(row.get("path_pattern"))
        mix_count = (
            _nl_tag_mix_available_count(
                source_dir,
                image_exts,
                recursive=recursive,
                path_pattern=path_pattern,
            )
            if mix["enabled"]
            else None
        )
        src_count = (
            mix_count
            if mix_count is not None
            else _count_source_images(
                source_dir,
                image_exts,
                recursive=recursive,
                path_pattern=path_pattern,
            )
        )
        resized_count = _count_images(
            resized_dir,
            image_exts,
            recursive=recursive,
            path_pattern=path_pattern,
        )
        used_count = resized_count or src_count
        trigger_clone_image_count = (
            _count_source_images(
                source_dir,
                image_exts,
                recursive=recursive,
                path_pattern=path_pattern,
            )
            if trigger_clone["enabled"]
            else 0
        )
        trigger_clone_repeats = trigger_clone["num_repeats"] if trigger_clone["enabled"] else 0
        trigger_clone_weighted = trigger_clone_image_count * trigger_clone_repeats
        source_images += src_count
        resized_images += resized_count
        train_images += used_count + trigger_clone_image_count
        weighted_images += used_count * repeats + trigger_clone_weighted
        dataset_repeats += repeats
        detail_rows.append({
            "index": idx + 1,
            "source_dir": _display_path(source_dir),
            "image_dir": _display_path(resized_dir),
            "cache_dir": _display_path(_resolve_project_path(str(row.get("cache_dir") or ""))),
            "source_image_count": src_count,
            "resized_image_count": resized_count,
            "train_image_count": used_count,
            "num_repeats": repeats,
            "weighted_image_count": used_count * repeats,
            "trigger_clone": trigger_clone,
            "trigger_clone_image_count": trigger_clone_image_count,
            "trigger_clone_weighted_image_count": trigger_clone_weighted,
            "uses_preprocessed_images": resized_count > 0,
            "recursive": recursive,
            "path_pattern": path_pattern,
            "nl_tag_mix": mix,
            "nl_tag_mix_missing": mix["enabled"] and mix_count is None,
        })

    sample_ratio = _positive_float(cfg.get("sample_ratio"), 1.0)
    explicit_epochs = cfg.get("max_train_epochs") not in (None, "")
    epochs = _positive_int(cfg.get("max_train_epochs"), 0) if explicit_epochs else None
    max_train_steps = _nonnegative_int(cfg.get("max_train_steps"), DEFAULT_MAX_TRAIN_STEPS)
    batch_size = _positive_int(cfg.get("train_batch_size"), 1)
    grad_accum = _positive_int(cfg.get("gradient_accumulation_steps"), 1)
    effective_batch = max(1, batch_size * grad_accum)
    repeated_images = int(weighted_images * sample_ratio)
    steps_per_epoch = (repeated_images + effective_batch - 1) // effective_batch if repeated_images else 0
    if epochs is not None:
        total_steps = steps_per_epoch * epochs
        duration_mode = "epochs"
    elif max_train_steps > 0:
        total_steps = max_train_steps
        duration_mode = "steps"
    else:
        total_steps = 0
        duration_mode = "unset"
    first_row = detail_rows[0] if detail_rows else {}

    return {
        "ok": True,
        "variant": variant,
        "preset": preset,
        "methods_subdir": methods_subdir,
        "source_image_count": source_images,
        "resized_image_count": resized_images,
        "train_image_count": train_images,
        "dataset_count": len(detail_rows),
        "dataset_num_repeats": dataset_repeats or 1,
        "weighted_image_count": weighted_images,
        "sample_ratio": sample_ratio,
        "max_train_epochs": epochs,
        "max_train_steps": max_train_steps,
        "uses_max_train_epochs": epochs is not None,
        "duration_configured": duration_mode != "unset",
        "duration_mode": duration_mode,
        "train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size": effective_batch,
        "repeated_image_count": repeated_images,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "uses_preprocessed_images": bool(detail_rows) and all(row["uses_preprocessed_images"] for row in detail_rows),
        "source_dir": first_row.get("source_dir", ""),
        "resized_dir": first_row.get("image_dir", ""),
        "lora_cache_dir": first_row.get("cache_dir", ""),
        "datasets": detail_rows,
    }




for _name in __all__:
    globals()[_name] = _exported(globals()[_name])
