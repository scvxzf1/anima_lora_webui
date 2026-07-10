"""Dataset row normalize/document helpers extracted from datasets service.

Pure-ish helpers for converting dataset editor rows, defaults, and TOML docs.
Path display/resolution stays local so this module can be shared by the editor
and preset APIs without depending on the config facade.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import tomlkit

from library.env import expand_env_vars, get_configs_root, load_dotenv
from library.preprocess.captions import normalize_caption_source_mode
from web.services.config import paths as _config_paths
from web.services.config.common import (
    _bool_value,
    _nonnegative_float,
    _nonnegative_int,
    _positive_int,
)
from web.services.config.metadata import (
    CAPTION_SOURCE_AUTO,
    CAPTION_SOURCE_JSON,
    DATASET_SETTING_KEYS,
    DEFAULT_LORA_CACHE_DIR,
    DEFAULT_NL_TAG_MIX_TAG_RATIO,
    DEFAULT_RESIZED_IMAGE_DIR,
    NL_TAG_MIX_ATTR_KEY,
    PREPROCESS_DATASET_SETTING_ORDER,
    RUNTIME_PREPROCESS_ATTR_KEY,
    TRIGGER_CLONE_ATTR_KEY,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = get_configs_root()

load_dotenv()


def _sync_from_facade() -> None:
    """Keep path roots aligned with the config_service facade / test patches.

    Avoid importing the facade here so pure helper callers can stay facade-free.
    """
    import sys

    global ROOT, CONFIGS_DIR
    facade = sys.modules.get("web.services.config_service")
    if facade is None:
        return
    if hasattr(facade, "ROOT"):
        ROOT = facade.ROOT
    if hasattr(facade, "CONFIGS_DIR"):
        CONFIGS_DIR = facade.CONFIGS_DIR


def _normalize_config_rel_path(rel_path: str) -> str:
    return _config_paths.normalize_config_rel_path(rel_path)


def _safe_resolve(rel_path: str) -> Path | None:
    _sync_from_facade()
    return _config_paths.safe_resolve(rel_path, root=ROOT, configs_dir=CONFIGS_DIR)


def _safe_config_subdir(subdir: str) -> Path | None:
    _sync_from_facade()
    return _config_paths.safe_config_subdir(subdir, configs_dir=CONFIGS_DIR)


def _resolve_project_path(value: str) -> Path:
    _sync_from_facade()
    return _config_paths.resolve_display_path(
        value,
        root=ROOT,
        configs_dir=CONFIGS_DIR,
        expand_env_vars_fn=expand_env_vars,
    )


def _display_path(path: Path) -> str:
    _sync_from_facade()
    return _config_paths.display_path(path, root=ROOT, configs_dir=CONFIGS_DIR)


def _derived_data_dir(source_path: Path, suffix: str) -> Path:
    parent = source_path.parent if source_path.name else source_path
    name = source_path.name or "dataset"
    return (parent / f"{name}_{suffix}").resolve()


def _single_dataset_config_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    source_dir = str(cfg.get("source_image_dir") or "image_dataset")
    image_dir = str(cfg.get("resized_image_dir") or DEFAULT_RESIZED_IMAGE_DIR)
    cache_dir = str(cfg.get("lora_cache_dir") or DEFAULT_LORA_CACHE_DIR)
    return {
        "general": {
            "caption_extension": ".txt",
            "keep_tokens": 3,
            "prefer_json_caption": False,
            "caption_source_mode": CAPTION_SOURCE_AUTO,
        },
        "datasets": [
            {
                "resolution": 1024,
                "batch_size": 1,
                "enable_bucket": True,
                "min_bucket_reso": 256,
                "max_bucket_reso": 1024,
                "bucket_reso_steps": 64,
                "bucket_no_upscale": False,
                "validation_split": 0.0,
                "validation_seed": 42,
                "subsets": [
                    {
                        "image_dir": image_dir,
                        "cache_dir": cache_dir,
                        "num_repeats": 1,
                        "custom_attributes": {"source_dir": source_dir},
                    }
                ],
            }
        ],
    }

def _dataset_defaults_from_config(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "resolution": _positive_int(_first_dataset_value(data, "resolution"), 1024),
        "batch_size": _positive_int(_first_dataset_value(data, "batch_size"), 1),
        "prior_loss_weight": _nonnegative_float(_first_dataset_value(data, "prior_loss_weight", 1.0), 1.0),
        "enable_bucket": bool(_first_dataset_value(data, "enable_bucket", True)),
        "min_bucket_reso": _positive_int(_first_dataset_value(data, "min_bucket_reso"), 256),
        "max_bucket_reso": _positive_int(_first_dataset_value(data, "max_bucket_reso"), 1024),
        "bucket_reso_steps": _positive_int(_first_dataset_value(data, "bucket_reso_steps"), 64),
        "bucket_no_upscale": bool(_first_dataset_value(data, "bucket_no_upscale", False)),
        "validation_split": _nonnegative_float(_first_dataset_value(data, "validation_split", 0.0), 0.0),
        "validation_split_num": _nonnegative_int(_first_dataset_value(data, "validation_split_num", 0), 0),
        "validation_seed": _nonnegative_int(_first_dataset_value(data, "validation_seed", 42), 42),
        "caption_extension": str(_first_dataset_value(
            data,
            "caption_extension",
            (data.get("general") or {}).get("caption_extension") or ".txt",
        )),
        "keep_tokens": _nonnegative_int((data.get("general") or {}).get("keep_tokens"), 3),
        "prefer_json_caption": _bool_value(_first_dataset_value(data, "prefer_json_caption"), False),
        "caption_source_mode": normalize_caption_source_mode(
            _first_dataset_value(data, "caption_source_mode"),
            _bool_value(_first_dataset_value(data, "prefer_json_caption"), False),
        ),
    }

def _dataset_defaults_from_dataset(dataset: dict[str, Any], data: dict[str, Any] | None = None) -> dict[str, Any]:
    source: dict[str, Any] = {"datasets": [dataset]}
    if isinstance(data, dict) and isinstance(data.get("general"), dict):
        source["general"] = data["general"]
    return _dataset_defaults_from_config(source)

def _dataset_summary_from_rows(rows: list[dict[str, Any]], defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    clean_rows = _normalize_dataset_rows(rows)
    clean_defaults = _normalize_dataset_defaults(defaults or _first_dataset_settings(clean_rows))
    first = _first_training_dataset_row(clean_rows) if clean_rows else {}
    repeats = sum(_positive_int(row.get("num_repeats"), 1) for row in clean_rows) if clean_rows else 0
    reg_rows = [row for row in clean_rows if _bool_value(row.get("is_reg"), False)]
    train_rows = [row for row in clean_rows if not _bool_value(row.get("is_reg"), False)]
    return {
        "ok": True,
        "dataset_count": len(clean_rows),
        "train_dataset_count": len(train_rows),
        "reg_dataset_count": len(reg_rows),
        "repeat_total": repeats,
        "reg_repeat_total": sum(_positive_int(row.get("num_repeats"), 1) for row in reg_rows),
        "source_dir": first.get("source_dir", ""),
        "image_dir": first.get("image_dir", ""),
        "cache_dir": first.get("cache_dir", ""),
        "resolution": clean_defaults.get("resolution", 1024),
        "batch_size": clean_defaults.get("batch_size", 1),
        "enable_bucket": clean_defaults.get("enable_bucket", True),
        "prior_loss_weight": clean_defaults.get("prior_loss_weight", 1.0),
    }

def _first_training_dataset_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if not _bool_value(row.get("is_reg"), False):
            return row
    return rows[0] if rows else {}

def _ensure_training_dataset_rows(rows: list[dict[str, Any]]) -> None:
    if rows and not any(not _bool_value(row.get("is_reg"), False) for row in rows):
        raise ValueError("至少需要一组普通训练数据集，正则化数据集只能作为辅助保留集")

def _dataset_rows_from_config(data: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    datasets = data.get("datasets") if isinstance(data, dict) else []
    if not isinstance(datasets, list):
        datasets = []

    fallback_source = str(cfg.get("source_image_dir") or "")
    fallback_image = str(cfg.get("resized_image_dir") or fallback_source)
    fallback_cache = str(cfg.get("lora_cache_dir") or "")
    fallback_path_pattern = cfg.get("path_pattern")

    for dataset in datasets:
        if not isinstance(dataset, dict):
            continue
        subsets = dataset.get("subsets") or []
        if not isinstance(subsets, list):
            continue
        for subset in subsets:
            if not isinstance(subset, dict):
                continue
            attrs = subset.get("custom_attributes")
            if not isinstance(attrs, dict):
                attrs = {}
            image_dir = _dataset_path_value(subset.get("image_dir") or fallback_image, cfg)
            cache_dir = _dataset_path_value(subset.get("cache_dir") or fallback_cache, cfg)
            source_dir = _dataset_path_value(attrs.get("source_dir") or fallback_source or image_dir, cfg)
            settings = _dataset_defaults_from_dataset(dataset, data)
            settings.update(_preprocess_settings_from_custom_attributes(attrs))
            rows.append({
                "source_dir": source_dir,
                "image_dir": image_dir,
                "cache_dir": cache_dir,
                "num_repeats": _positive_int(subset.get("num_repeats"), 1),
                "is_reg": _bool_value(subset.get("is_reg"), False),
                "recursive": _bool_value(subset.get("recursive", dataset.get("recursive")), True),
                "path_pattern": _normalize_path_pattern(
                    subset.get("path_pattern", dataset.get("path_pattern", fallback_path_pattern))
                ),
                "nl_tag_mix": _normalize_nl_tag_mix(attrs.get(NL_TAG_MIX_ATTR_KEY)),
                "trigger_clone": _normalize_trigger_clone(attrs.get(TRIGGER_CLONE_ATTR_KEY)),
                "settings": settings,
            })

    if not rows:
        rows = _normalize_dataset_rows([
            {
                "source_dir": fallback_source,
                "image_dir": fallback_image,
                "cache_dir": fallback_cache,
                "num_repeats": 1,
            }
        ])
    return rows

def _normalize_dataset_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean_rows: list[dict[str, Any]] = []
    for raw in rows or []:
        if not isinstance(raw, dict):
            continue
        source = str(raw.get("source_dir") or raw.get("source_image_dir") or "").strip()
        image = str(raw.get("image_dir") or raw.get("resized_image_dir") or "").strip()
        cache = str(raw.get("cache_dir") or raw.get("lora_cache_dir") or "").strip()
        if not source and not image and not cache:
            continue
        if not source:
            source = image
        source_path = _resolve_project_path(source)
        image_path = _resolve_project_path(image) if image else _derived_data_dir(source_path, "resized")
        cache_path = _resolve_project_path(cache) if cache else _derived_data_dir(source_path, "lora_cache")
        clean_rows.append({
            "source_dir": _display_path(source_path),
            "image_dir": _display_path(image_path),
            "cache_dir": _display_path(cache_path),
            "num_repeats": _positive_int(raw.get("num_repeats"), 1),
            "is_reg": _bool_value(raw.get("is_reg"), False),
            "recursive": _bool_value(raw.get("recursive"), True),
            "path_pattern": _normalize_path_pattern(raw.get("path_pattern")),
            "nl_tag_mix": _normalize_nl_tag_mix(raw.get(NL_TAG_MIX_ATTR_KEY) or raw.get("nl_tag_mix")),
            "trigger_clone": _normalize_trigger_clone(
                raw.get(TRIGGER_CLONE_ATTR_KEY) or raw.get("trigger_clone")
            ),
            "settings": _normalize_dataset_row_settings(raw),
        })
    return clean_rows

def _normalize_dataset_row_settings(raw: dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw.get("settings"), dict):
        settings = dict(raw["settings"])
        for key in DATASET_SETTING_KEYS:
            if key in raw and key not in settings:
                settings[key] = raw[key]
        return settings
    if any(key in raw for key in DATASET_SETTING_KEYS):
        return {key: raw[key] for key in DATASET_SETTING_KEYS if key in raw}
    return {}

def _fill_missing_dataset_row_settings(rows: list[dict[str, Any]], defaults: dict[str, Any]) -> list[dict[str, Any]]:
    fallback = _normalize_dataset_defaults(defaults)
    next_rows: list[dict[str, Any]] = []
    for row in rows:
        next_row = dict(row)
        settings = next_row.get("settings")
        if isinstance(settings, dict) and settings:
            merged = dict(fallback)
            merged.update(settings)
            next_row["settings"] = _normalize_dataset_defaults(merged)
        else:
            next_row["settings"] = fallback
        next_rows.append(next_row)
    return next_rows

def _normalize_dataset_defaults(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    out["resolution"] = _positive_int(raw.get("resolution"), 1024)
    out["batch_size"] = _positive_int(raw.get("batch_size"), 1)
    out["prior_loss_weight"] = _nonnegative_float(raw.get("prior_loss_weight"), 1.0)
    out["enable_bucket"] = str(raw.get("enable_bucket", True)).lower() not in {"0", "false", "no", "off"}
    out["min_bucket_reso"] = _positive_int(raw.get("min_bucket_reso"), 256)
    out["max_bucket_reso"] = _positive_int(raw.get("max_bucket_reso"), 1024)
    out["bucket_reso_steps"] = _positive_int(raw.get("bucket_reso_steps"), 64)
    out["bucket_no_upscale"] = str(raw.get("bucket_no_upscale", False)).lower() in {"1", "true", "yes", "on"}
    if raw.get("validation_split_num") not in (None, ""):
        out["validation_split_num"] = _nonnegative_int(raw.get("validation_split_num"), 0)
    out["validation_split"] = _nonnegative_float(raw.get("validation_split"), 0.0)
    out["validation_seed"] = _nonnegative_int(raw.get("validation_seed"), 42)
    out["caption_extension"] = str(raw.get("caption_extension") or ".txt").strip() or ".txt"
    out["keep_tokens"] = _nonnegative_int(raw.get("keep_tokens"), 3)
    prefer_json = _bool_value(raw.get("prefer_json_caption"), False)
    out["prefer_json_caption"] = prefer_json
    out["caption_source_mode"] = normalize_caption_source_mode(
        raw.get("caption_source_mode"),
        prefer_json,
    )
    return out

def _normalize_preprocess_dataset_settings(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    if "resolution" in raw:
        out["resolution"] = _positive_int(raw.get("resolution"), 1024)
    if "enable_bucket" in raw:
        out["enable_bucket"] = str(raw.get("enable_bucket", True)).lower() not in {"0", "false", "no", "off"}
    if "min_bucket_reso" in raw:
        out["min_bucket_reso"] = _positive_int(raw.get("min_bucket_reso"), 256)
    if "max_bucket_reso" in raw:
        out["max_bucket_reso"] = _positive_int(raw.get("max_bucket_reso"), 1024)
    if "bucket_reso_steps" in raw:
        out["bucket_reso_steps"] = _positive_int(raw.get("bucket_reso_steps"), 64)
    if "bucket_no_upscale" in raw:
        out["bucket_no_upscale"] = str(raw.get("bucket_no_upscale", False)).lower() in {"1", "true", "yes", "on"}
    return out

def _normalize_nl_tag_mix(raw: Any) -> dict[str, Any]:
    source = raw if isinstance(raw, dict) else {}
    enabled = _bool_value(source.get("enabled"), False)
    try:
        tag_ratio = float(source.get("tag_ratio", DEFAULT_NL_TAG_MIX_TAG_RATIO))
    except (TypeError, ValueError):
        tag_ratio = DEFAULT_NL_TAG_MIX_TAG_RATIO
    if tag_ratio > 1:
        tag_ratio = tag_ratio / 100
    tag_ratio = min(1.0, max(0.0, tag_ratio))
    return {
        "enabled": enabled,
        "tag_ratio": tag_ratio,
    }

def _normalize_trigger_clone(raw: Any) -> dict[str, Any]:
    source = raw if isinstance(raw, dict) else {}
    return {
        "enabled": _bool_value(source.get("enabled"), False),
        "prompt": str(source.get("prompt") or "").strip(),
        "num_repeats": _positive_int(source.get("num_repeats"), 1),
    }

def _trigger_clone_should_persist(clone: dict[str, Any]) -> bool:
    normalized = _normalize_trigger_clone(clone)
    return (
        bool(normalized["enabled"])
        or bool(normalized["prompt"])
        or _positive_int(normalized.get("num_repeats"), 1) != 1
    )

def _nl_tag_mix_enabled(row: dict[str, Any]) -> bool:
    return bool(_normalize_nl_tag_mix(row.get("nl_tag_mix")).get("enabled"))

def _preprocess_settings_from_custom_attributes(attrs: dict[str, Any]) -> dict[str, Any]:
    raw = attrs.get(RUNTIME_PREPROCESS_ATTR_KEY) if isinstance(attrs, dict) else None
    return _normalize_preprocess_dataset_settings(raw) if isinstance(raw, dict) else {}

def _preprocess_settings_for_runtime_attrs(row_cfg: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_dataset_defaults(row_cfg)
    return {key: normalized[key] for key in PREPROCESS_DATASET_SETTING_ORDER if key in normalized}

def _build_dataset_config_doc(
    clean_rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    *,
    prefer_train_batch_size: bool = False,
    include_preprocess_settings: bool = True,
) -> str:
    doc = tomlkit.document()
    doc.add(tomlkit.comment("Web UI 自动生成的数据集配置。"))
    doc.add(tomlkit.comment("原始数据集路径保存在 custom_attributes.source_dir，训练读取 image_dir/cache_dir。"))

    general = tomlkit.table()
    general.add("caption_extension", str(cfg.get("caption_extension") or ".txt"))
    general.add("keep_tokens", _nonnegative_int(cfg.get("keep_tokens"), 3))
    doc.add("general", general)

    datasets = tomlkit.aot()
    for row in clean_rows:
        row_cfg = _dataset_row_settings(row, cfg)
        dataset = tomlkit.table()
        if include_preprocess_settings:
            dataset.add("resolution", _positive_int(row_cfg.get("resolution"), 1024))
        batch_size = row_cfg.get("batch_size")
        if prefer_train_batch_size and cfg.get("train_batch_size") not in (None, ""):
            batch_size = cfg.get("train_batch_size")
        dataset.add("batch_size", _positive_int(batch_size, 1))
        dataset.add("prior_loss_weight", _nonnegative_float(row_cfg.get("prior_loss_weight"), 1.0))
        caption_source_mode = normalize_caption_source_mode(
            row_cfg.get("caption_source_mode"),
            _bool_value(row_cfg.get("prefer_json_caption"), False),
        )
        dataset.add("caption_source_mode", caption_source_mode)
        dataset.add("caption_extension", str(row_cfg.get("caption_extension") or cfg.get("caption_extension") or ".txt"))
        dataset.add(
            "prefer_json_caption",
            caption_source_mode == CAPTION_SOURCE_JSON
            or _bool_value(row_cfg.get("prefer_json_caption"), False),
        )
        if include_preprocess_settings:
            dataset.add("enable_bucket", bool(row_cfg.get("enable_bucket", True)))
            dataset.add("min_bucket_reso", _positive_int(row_cfg.get("min_bucket_reso"), 256))
            dataset.add("max_bucket_reso", _positive_int(row_cfg.get("max_bucket_reso"), 1024))
            dataset.add("bucket_reso_steps", _positive_int(row_cfg.get("bucket_reso_steps"), 64))
            dataset.add("bucket_no_upscale", bool(row_cfg.get("bucket_no_upscale", False)))
        validation_split_num = _nonnegative_int(row_cfg.get("validation_split_num"), 0)
        if validation_split_num > 0:
            dataset.add("validation_split_num", validation_split_num)
        dataset.add("validation_split", _nonnegative_float(row_cfg.get("validation_split"), 0.0))
        dataset.add("validation_seed", _nonnegative_int(row_cfg.get("validation_seed"), 42))

        subsets = tomlkit.aot()
        subset = tomlkit.table()
        subset.add("image_dir", row["image_dir"])
        subset.add("cache_dir", row["cache_dir"])
        subset.add("num_repeats", _positive_int(row.get("num_repeats"), 1))
        if _bool_value(row.get("is_reg"), False):
            subset.add("is_reg", True)
        if not _bool_value(row.get("recursive"), True):
            subset.add("recursive", False)
        path_pattern = _normalize_path_pattern(row.get("path_pattern"))
        if path_pattern != "*":
            subset.add("path_pattern", path_pattern)
        attrs = tomlkit.inline_table()
        attrs.add("source_dir", row["source_dir"])
        mix = _normalize_nl_tag_mix(row.get("nl_tag_mix"))
        if mix["enabled"]:
            mix_attrs = tomlkit.inline_table()
            mix_attrs.add("enabled", True)
            mix_attrs.add("tag_ratio", mix["tag_ratio"])
            attrs.add(NL_TAG_MIX_ATTR_KEY, mix_attrs)
        trigger_clone = _normalize_trigger_clone(row.get("trigger_clone"))
        if _trigger_clone_should_persist(trigger_clone):
            clone_attrs = tomlkit.inline_table()
            clone_attrs.add("enabled", bool(trigger_clone["enabled"]))
            clone_attrs.add("prompt", trigger_clone["prompt"])
            clone_attrs.add("num_repeats", _positive_int(trigger_clone.get("num_repeats"), 1))
            attrs.add(TRIGGER_CLONE_ATTR_KEY, clone_attrs)
        if not include_preprocess_settings:
            preprocess_attrs = tomlkit.inline_table()
            for key, value in _preprocess_settings_for_runtime_attrs(row_cfg).items():
                preprocess_attrs.add(key, value)
            attrs.add(RUNTIME_PREPROCESS_ATTR_KEY, preprocess_attrs)
        subset.add("custom_attributes", attrs)
        subsets.append(subset)
        dataset.add("subsets", subsets)
        datasets.append(dataset)
    doc.add("datasets", datasets)
    return tomlkit.dumps(doc)

def _dataset_row_settings(row: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("settings")
    if isinstance(raw, dict):
        return _normalize_dataset_defaults(raw)
    return _normalize_dataset_defaults(fallback)

def _dataset_training_defaults(rows: list[dict[str, Any]], fallback: dict[str, Any]) -> dict[str, Any]:
    defaults = _normalize_dataset_defaults(fallback)
    for row in rows:
        if not _bool_value(row.get("is_reg"), False):
            continue
        settings = row.get("settings")
        if isinstance(settings, dict) and "prior_loss_weight" in settings:
            defaults["prior_loss_weight"] = _nonnegative_float(settings.get("prior_loss_weight"), defaults["prior_loss_weight"])
            break
    return defaults

def _first_dataset_settings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if rows and isinstance(rows[0].get("settings"), dict):
        return rows[0]["settings"]
    return {}

def _first_dataset_value(data: dict[str, Any], key: str, default: Any = None) -> Any:
    datasets = data.get("datasets") if isinstance(data, dict) else []
    if isinstance(datasets, list) and datasets and isinstance(datasets[0], dict):
        if key in datasets[0]:
            return datasets[0].get(key)
    general = data.get("general") if isinstance(data, dict) else {}
    if isinstance(general, dict) and key in general:
        return general.get(key)
    return default

def _dataset_path_value(value: Any, cfg: dict[str, Any]) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    for key, raw in cfg.items():
        if isinstance(raw, str):
            text = text.replace("{" + key + "}", raw)
    return _display_path(_resolve_project_path(expand_env_vars(text)))

def _safe_file_stem(value: str) -> str:
    stem = Path(str(value or "").replace("\\", "/")).stem
    chars: list[str] = []
    for ch in stem:
        if ch.isascii() and (ch.isalnum() or ch in {"-", "_"}):
            chars.append(ch)
        elif ch.isspace():
            chars.append("_")
    return "".join(chars).strip("_-") or "dataset"

def _normalize_path_pattern(value: Any) -> str:
    return str(value or "*").strip() or "*"

