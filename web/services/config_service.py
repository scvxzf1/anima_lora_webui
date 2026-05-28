"""Configuration loading, merging, and saving."""

from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import tomllib
from typing import Any
from urllib.parse import quote

import toml
import tomlkit
from PIL import Image, UnidentifiedImageError

from library.env import expand_env_vars, expand_env_vars_in_obj, load_dotenv
from library.preprocess.captions import load_json_caption
from web.services.settings_service import display_path as _display_settings_path
from web.services.settings_service import resolve_output_root

ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = ROOT / "configs"
GUI_METHODS_DIR = CONFIGS_DIR / "gui-methods"
IMPORTED_CONFIGS_DIR = CONFIGS_DIR / "imported"
PRESETS_FILE = CONFIGS_DIR / "presets.toml"
WEB_FILE_GROUPS_FILE = CONFIGS_DIR / "web-file-groups.toml"
WEB_USER_LOCKS_FILE = CONFIGS_DIR / "web-user-locks.toml"
DEFAULT_SAMPLE_PROMPTS_FILE = "configs/sample_prompts.txt"
DEFAULT_RESIZED_IMAGE_DIR = "post_image_dataset/resized"
DEFAULT_LORA_CACHE_DIR = "post_image_dataset/lora"
DEFAULT_MAX_TRAIN_STEPS = 0
PREPROCESS_ENV_CHECK_KEY = "preprocess_environment"
PREPROCESS_ENV_REQUIRED_FILES = (
    "tasks.py",
    "library/__init__.py",
    "library/preprocess/__init__.py",
    "scripts/__init__.py",
    "scripts/tasks/__init__.py",
    "scripts/tasks/preprocess.py",
    "scripts/preprocess/resize_images.py",
    "scripts/preprocess/cache_latents.py",
    "scripts/preprocess/cache_text_embeddings.py",
)
UI_ONLY_CONFIG_FIELDS = {
    "dataset_config_picker",
}
DATASET_PRESETS_DIR = CONFIGS_DIR / "datasets"
DATASET_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})
DATASET_PREVIEW_LIMIT = 120
DATASET_CAPTION_MAX_CHARS = 20000
OUTPUT_RUN_CONFIG_FILES = {
    "original": ("config.original.toml", "原始配置"),
    "runtime": ("config.runtime.toml", "运行时配置"),
    "dataset": ("dataset.runtime.toml", "数据集配置"),
}
DATASET_SETTING_KEYS = frozenset({
    "resolution",
    "enable_bucket",
    "min_bucket_reso",
    "max_bucket_reso",
    "bucket_reso_steps",
    "bucket_no_upscale",
    "validation_split",
    "validation_split_num",
    "validation_seed",
    "prefer_json_caption",
})
PREPROCESS_DATASET_SETTING_ORDER = (
    "resolution",
    "enable_bucket",
    "min_bucket_reso",
    "max_bucket_reso",
    "bucket_reso_steps",
    "bucket_no_upscale",
)
PREPROCESS_DATASET_SETTING_KEYS = frozenset(PREPROCESS_DATASET_SETTING_ORDER)
RUNTIME_PREPROCESS_ATTR_KEY = "preprocess"
SYSTEM_PRESET_FILES = frozenset({
    "configs/base.toml",
    "configs/presets.toml",
})
SYSTEM_DATASET_PRESET_FILES = frozenset({
    "configs/datasets/easycontrol.toml",
    "configs/datasets/ip_adapter.toml",
})
HIDDEN_DATASET_PRESET_FILES = frozenset({
    "configs/datasets/easycontrol.toml",
    "configs/datasets/ip_adapter.toml",
})
HIDDEN_CONFIG_FILES = frozenset({
    "configs/gui-methods/postfix_ortho_cond.toml",
    "configs/methods/postfix.toml",
})
SYSTEM_PRESET_PREFIXES = ("configs/methods/", "configs/gui-methods/")
SYSTEM_MANAGED_FILES = frozenset({
    "configs/web-file-groups.toml",
    "configs/web-user-locks.toml",
})
CONFIG_FILE_LABELS_ZH = {
    "configs/base.toml": "基础公共配置",
    "configs/presets.toml": "训练预设集合",
    "configs/web-file-groups.toml": "Web 配置分组表",
    "configs/datasets/easycontrol.toml": "EasyControl 数据集蓝图",
    "configs/datasets/ip_adapter.toml": "IP-Adapter 数据集蓝图",
    "configs/gui-methods/chimera_hydra.toml": "Chimera Hydra 训练变体",
    "configs/gui-methods/easycontrol.toml": "EasyControl 训练变体",
    "configs/gui-methods/hydralora-8gb.toml": "HydraLoRA 低显存变体",
    "configs/gui-methods/hydralora.toml": "HydraLoRA 训练变体",
    "configs/gui-methods/ip_adapter.toml": "IP-Adapter 训练变体",
    "configs/gui-methods/lokr.toml": "LoKr 训练变体",
    "configs/gui-methods/lora-8gb.toml": "LoRA 低显存变体",
    "configs/gui-methods/lora.toml": "LoRA 标准训练变体",
    "configs/gui-methods/reft.toml": "ReFT 训练变体",
    "configs/gui-methods/soft_tokens.toml": "Soft Tokens 训练变体",
    "configs/gui-methods/tlora-8gb.toml": "T-LoRA 低显存变体",
    "configs/gui-methods/tlora.toml": "T-LoRA 训练变体",
    "configs/gui-methods/tlora_ortho_reft.toml": "T-LoRA + Ortho + ReFT 组合变体",
    "configs/methods/chimera.toml": "Chimera 内置方法配置",
    "configs/methods/easycontrol.toml": "EasyControl 内置方法配置",
    "configs/methods/ip_adapter.toml": "IP-Adapter 内置方法配置",
    "configs/methods/lora.toml": "LoRA 内置方法配置",
    "configs/methods/soft_tokens.toml": "Soft Tokens 内置方法配置",
    "configs/methods/spd.toml": "SPD 实验配置",
    "configs/methods/turbo.toml": "Turbo 内置方法配置",
}
SYSTEM_CONFIG_GROUP_IDS = frozenset({
    "web_config",
    "presets",
    "methods",
    "gui_methods",
    "rokkotsu_goddess",
    "imported",
    "datasets",
})
FIXED_SYSTEM_CONFIG_GROUP_IDS = frozenset({
    "web_config",
    "presets",
    "methods",
    "gui_methods",
})
FILE_MOVE_TARGET_GROUPS = frozenset({
    "imported",
    "rokkotsu_goddess",
    "datasets",
})
USER_LOCKABLE_GROUPS = frozenset({
    "imported",
    "rokkotsu_goddess",
    "datasets",
})

load_dotenv()


def list_methods() -> list[str]:
    return [
        "lora", "lokr", "ortholora", "tlora", "hydralora",
        "reft", "chimera", "soft_tokens", "ip_adapter", "easycontrol",
        "spd",
    ]


def list_variants(method: str) -> list[str]:
    if method == "spd":
        spd_config = CONFIGS_DIR / "methods" / "spd.toml"
        return ["spd"] if spd_config.exists() else []
    if not GUI_METHODS_DIR.exists():
        return []
    variants = [stem for _order, stem in _builtin_variants_by_family().get(method, [])]
    if not variants:
        variants = _legacy_exact_variant_for_method(method)
    variants.extend(_custom_gui_variants())
    return variants


def _builtin_variants_by_family() -> dict[str, list[tuple[int, str]]]:
    by_family: dict[str, list[tuple[int, str]]] = {}
    if not GUI_METHODS_DIR.exists():
        return by_family
    for path in GUI_METHODS_DIR.glob("*.toml"):
        if _display_path(path) in HIDDEN_CONFIG_FILES:
            continue
        meta = _read_variant_metadata(path)
        family = meta.get("family")
        if not isinstance(family, str) or not family:
            continue
        order = meta.get("order")
        order_int = order if isinstance(order, int) else 100
        by_family.setdefault(family, []).append((order_int, path.stem))
    for entries in by_family.values():
        entries.sort(key=lambda item: (item[0], item[1]))
    return by_family


def _read_variant_metadata(path: Path) -> dict[str, Any]:
    try:
        data = toml.loads(path.read_text(encoding="utf-8"))
    except (OSError, toml.TomlDecodeError):
        return {}
    meta = data.get("variant")
    return meta if isinstance(meta, dict) else {}


def _legacy_exact_variant_for_method(method: str) -> list[str]:
    path = GUI_METHODS_DIR / f"{method}.toml"
    if not path.is_file() or _display_path(path) in HIDDEN_CONFIG_FILES:
        return []
    return [method]


def _custom_gui_variants() -> list[str]:
    custom_dir = GUI_METHODS_DIR / "custom"
    if not custom_dir.exists():
        return []
    return [f"custom/{p.stem}" for p in sorted(custom_dir.glob("*.toml"))]


def list_all_variants() -> list[str]:
    if not GUI_METHODS_DIR.exists():
        return []
    return sorted(
        p.stem for p in GUI_METHODS_DIR.glob("*.toml")
        if _display_path(p) not in HIDDEN_CONFIG_FILES
    )


def list_presets() -> list[str]:
    if not PRESETS_FILE.exists():
        return []
    data = toml.loads(PRESETS_FILE.read_text(encoding="utf-8"))
    return sorted(k for k, v in data.items() if isinstance(v, dict))


def load_merged_config(variant: str, preset: str, methods_subdir: str = "gui-methods") -> dict[str, Any]:
    methods_dir = _safe_config_subdir(methods_subdir)
    if methods_dir is None:
        raise ValueError("配置目录不合法")
    base = _load(CONFIGS_DIR / "base.toml")
    presets_data = _load(PRESETS_FILE)
    pset = presets_data.get(preset, {}) if isinstance(presets_data.get(preset), dict) else {}
    meth = _load(methods_dir / f"{variant}.toml")

    merged: dict[str, Any] = {}
    for k, v in base.items():
        if k not in ("general", "datasets"):
            merged[k] = v
    for k, v in pset.items():
        merged[k] = v
    for k, v in meth.items():
        merged[k] = v
    merged.setdefault("max_train_steps", DEFAULT_MAX_TRAIN_STEPS)
    merged = expand_env_vars_in_obj(merged)
    return apply_auto_data_dirs(merged)


def suggest_data_dirs(source_image_dir: str) -> dict[str, Any]:
    source_path = _resolve_project_path(str(source_image_dir or ""))
    if not str(source_image_dir or "").strip():
        return {"ok": False, "error": "请先填写源图像目录 / source_image_dir"}
    return {
        "ok": True,
        "source_image_dir": _display_path(source_path),
        "resized_image_dir": _display_path(_derived_data_dir(source_path, "resized")),
        "lora_cache_dir": _display_path(_derived_data_dir(source_path, "lora_cache")),
    }


def suggest_dataset_dirs(source_dirs: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for idx, raw in enumerate(source_dirs):
        source = str(raw or "").strip()
        if not source:
            continue
        source_path = _resolve_project_path(source)
        rows.append({
            "index": idx,
            "source_dir": _display_path(source_path),
            "image_dir": _display_path(_derived_data_dir(source_path, "resized")),
            "cache_dir": _display_path(_derived_data_dir(source_path, "lora_cache")),
        })
    if not rows:
        return {"ok": False, "error": "请至少填写一个原始数据集路径"}
    return {"ok": True, "datasets": rows}


def list_dataset_presets() -> dict[str, Any]:
    DATASET_PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    presets: list[dict[str, Any]] = []
    for path in sorted(DATASET_PRESETS_DIR.glob("*.toml")):
        rel_path = _display_path(path)
        if rel_path in HIDDEN_DATASET_PRESET_FILES:
            continue
        meta = get_config_file_meta(rel_path)
        summary = _dataset_preset_summary(rel_path)
        presets.append({
            **meta,
            "readonly": _is_dataset_preset_readonly(rel_path),
            "system_preset": rel_path in SYSTEM_DATASET_PRESET_FILES,
            "summary": summary,
        })
    return {"ok": True, "presets": presets}


def load_dataset_preset(rel_path: str) -> dict[str, Any]:
    normalized = _normalize_dataset_preset_path(rel_path, must_exist=True)
    path = _safe_resolve(normalized)
    if path is None or not path.exists():
        raise ValueError("数据集预设不存在")
    content = path.read_text(encoding="utf-8")
    data = toml.loads(content)
    rows = _dataset_rows_from_config(data, {})
    defaults = _dataset_defaults_from_config(data)
    return {
        "ok": True,
        "file": normalized,
        "name": Path(normalized).stem,
        "content": content,
        "datasets": rows,
        "defaults": defaults,
        "readonly": _is_dataset_preset_readonly(normalized),
        "meta": get_config_file_meta(normalized),
        "summary": _dataset_summary_from_rows(rows, defaults),
    }


def save_dataset_preset(
    rel_path: str,
    rows: list[dict[str, Any]],
    defaults: dict[str, Any] | None = None,
    *,
    overwrite: bool = True,
) -> dict[str, Any]:
    normalized = _normalize_dataset_preset_path(rel_path, must_exist=False)
    if _is_dataset_preset_readonly(normalized):
        raise ValueError("系统数据集预设为只读，请复制后编辑")
    path = _safe_resolve(normalized)
    if path is None:
        raise ValueError("数据集预设路径不合法")
    if path.exists() and not overwrite:
        raise ValueError("数据集预设已存在，请换一个名称")

    clean_rows = _fill_missing_dataset_row_settings(_normalize_dataset_rows(rows), _normalize_dataset_defaults(defaults or {}))
    if not clean_rows:
        raise ValueError("请至少填写一个数据集路径")
    cfg = _normalize_dataset_defaults(defaults or {})
    content = _build_dataset_config_doc(clean_rows, cfg)
    ok, msg = save_raw_file(normalized, content, overwrite=overwrite)
    if not ok:
        raise ValueError(msg)
    return {
        "ok": True,
        "message": f"已保存数据集预设 {Path(normalized).name}",
        "file": normalized,
        "datasets": clean_rows,
        "defaults": _normalize_dataset_defaults(cfg),
        "content": content,
        "summary": _dataset_summary_from_rows(clean_rows, _normalize_dataset_defaults(cfg)),
    }


def save_dataset_preset_as(
    name: str,
    rows: list[dict[str, Any]],
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stem = _safe_file_stem(name)
    return save_dataset_preset(f"configs/datasets/{stem}.toml", rows, defaults, overwrite=False)


def delete_dataset_preset(rel_path: str) -> dict[str, Any]:
    normalized = _normalize_dataset_preset_path(rel_path, must_exist=True)
    if _is_dataset_preset_readonly(normalized):
        raise ValueError("系统数据集预设为只读，不能删除")
    ok, msg = delete_raw_file(normalized)
    if not ok:
        raise ValueError(msg)
    return {"ok": True, "message": "数据集预设已删除", "file": normalized}


def apply_dataset_preset_to_training_config(
    dataset_file: str,
    train_file: str,
    train_content: str | None = None,
) -> dict[str, Any]:
    dataset_rel = _normalize_dataset_preset_path(dataset_file, must_exist=True)
    train_rel = _normalize_config_rel_path(train_file)
    train_path = _safe_resolve(train_rel)
    if train_path is None or train_path.suffix.lower() != ".toml":
        raise ValueError("训练配置路径不合法")
    if not train_path.exists():
        raise ValueError("训练配置不存在")

    preset = load_dataset_preset(dataset_rel)
    rows = _normalize_dataset_rows(preset.get("datasets", []))
    if not rows:
        raise ValueError("数据集预设里没有可用路径")
    first = rows[0]
    values = {
        "dataset_config": dataset_rel,
        "source_image_dir": first["source_dir"],
        "resized_image_dir": first["image_dir"],
        "lora_cache_dir": first["cache_dir"],
    }
    ok, msg, _path, next_content, changed = _prepare_raw_file_patch(train_rel, values, content=train_content)
    if not ok:
        raise ValueError(msg)
    ok, msg = save_raw_file(train_rel, next_content, overwrite=True)
    if not ok:
        raise ValueError(msg)
    return {
        "ok": True,
        "message": "已应用数据集预设",
        "dataset_config": dataset_rel,
        "datasets": rows,
        "defaults": preset.get("defaults") or {},
        "train_content": next_content,
        "changed": changed,
        "values": values,
        "summary": preset.get("summary") or _dataset_summary_from_rows(rows, preset.get("defaults") or {}),
    }


def list_dataset_preset_images(
    dataset_file: str,
    dataset_index: int = 0,
    *,
    source: str = "training",
    limit: int = DATASET_PREVIEW_LIMIT,
) -> dict[str, Any]:
    preset = load_dataset_preset(dataset_file)
    rows = _normalize_dataset_rows(preset.get("datasets", []))
    if not rows:
        raise ValueError("数据集预设里没有可预览路径")
    if dataset_index < 0 or dataset_index >= len(rows):
        raise ValueError("数据集序号不在范围内")

    row = rows[dataset_index]
    defaults = _normalize_dataset_defaults(preset.get("defaults") or {})
    settings = _normalize_dataset_defaults(row.get("settings") or defaults)
    caption_extension = str(settings.get("caption_extension") or ".txt").strip() or ".txt"
    if not caption_extension.startswith("."):
        caption_extension = f".{caption_extension}"
    prefer_json_caption = _bool_value(settings.get("prefer_json_caption"), False)
    source_kind = "source" if str(source or "").strip().lower() == "source" else "training"
    image_dir_raw = row.get("source_dir") if source_kind == "source" else row.get("image_dir")
    image_dir = _resolve_project_path(str(image_dir_raw or ""))
    source_dir = _resolve_project_path(str(row.get("source_dir") or ""))
    train_dir = _resolve_project_path(str(row.get("image_dir") or ""))

    listing = _list_dataset_image_files(image_dir, limit)
    images = [
        _dataset_image_preview_meta(
            path,
            preset_file=preset["file"],
            dataset_index=dataset_index,
            source=source_kind,
            caption_extension=caption_extension,
            prefer_json_caption=prefer_json_caption,
            source_dir=source_dir,
            train_dir=train_dir,
        )
        for path in listing["items"]
    ]
    directory_exists = image_dir.is_dir()
    return {
        "ok": True,
        "file": preset["file"],
        "dataset_index": dataset_index,
        "dataset_label": f"第 {dataset_index + 1} 组数据集",
        "source": source_kind,
        "source_label": "原始图目录" if source_kind == "source" else "训练图目录",
        "directory": _display_path(image_dir),
        "directory_exists": directory_exists,
        "caption_extension": caption_extension,
        "prefer_json_caption": prefer_json_caption,
        "count": len(images),
        "total": listing["total"],
        "limit": listing["limit"],
        "images": images,
        "row": row,
        "settings": settings,
        "message": "" if images else _dataset_preview_empty_message(image_dir, source_kind),
    }


def resolve_dataset_preview_image(
    dataset_file: str,
    dataset_index: int,
    image_file: str,
    *,
    source: str = "training",
) -> Path:
    preset = load_dataset_preset(dataset_file)
    rows = _normalize_dataset_rows(preset.get("datasets", []))
    if dataset_index < 0 or dataset_index >= len(rows):
        raise ValueError("数据集序号不在范围内")
    row = rows[dataset_index]
    source_kind = "source" if str(source or "").strip().lower() == "source" else "training"
    root = _resolve_project_path(str(row.get("source_dir") if source_kind == "source" else row.get("image_dir") or ""))
    if not root.is_dir():
        raise FileNotFoundError("数据集图片目录不存在")
    path = _resolve_project_path(str(image_file or ""))
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("图片不属于当前数据集路径") from exc
    if path.suffix.lower() not in DATASET_IMAGE_EXTS:
        raise ValueError("只允许读取数据集图片")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError("图片不存在")
    return path


def load_dataset_editor(variant: str, preset: str, methods_subdir: str = "gui-methods") -> dict[str, Any]:
    cfg = apply_auto_data_dirs(load_merged_config(variant, preset, methods_subdir))
    dataset_path = _dataset_config_path_from_cfg(cfg)
    if dataset_path and dataset_path.exists():
        data = toml.loads(dataset_path.read_text(encoding="utf-8"))
    else:
        data = _single_dataset_config_from_cfg(cfg)
    rows = _dataset_rows_from_config(data, cfg)
    return {
        "ok": True,
        "dataset_config": _display_path(dataset_path) if dataset_path else "",
        "datasets": rows,
        "defaults": _dataset_defaults_from_config(data),
    }


def save_dataset_editor(
    variant: str,
    preset: str,
    methods_subdir: str,
    rows: list[dict[str, Any]],
    defaults: dict[str, Any] | None = None,
    train_file: str | None = None,
    train_content: str | None = None,
    prefer_existing_dataset_config: bool = True,
) -> dict[str, Any]:
    cfg = apply_auto_data_dirs(load_merged_config(variant, preset, methods_subdir))
    if defaults:
        cfg.update(_normalize_dataset_defaults(defaults))
    clean_rows = _fill_missing_dataset_row_settings(_normalize_dataset_rows(rows), _normalize_dataset_defaults(cfg))
    if not clean_rows:
        raise ValueError("请至少填写一个数据集路径")

    train_rel = _normalize_config_rel_path(train_file) if train_file else _training_config_rel_path(variant, methods_subdir)
    dataset_variant = Path(train_rel).stem if train_rel else variant
    dataset_rel = _dataset_config_rel_path(
        cfg,
        dataset_variant,
        methods_subdir,
        prefer_existing=prefer_existing_dataset_config,
    )
    dataset_path = _safe_resolve(dataset_rel)
    if dataset_path is None:
        raise ValueError("数据集配置路径不合法")

    if train_rel and get_config_file_meta(train_rel).get("locked"):
        raise ValueError(f"{_lock_reason_message(get_config_file_meta(train_rel))}，请使用新名称保存新配置后编辑")

    next_content = ""
    if train_rel:
        first = clean_rows[0]
        values = {
            "dataset_config": dataset_rel,
            "source_image_dir": first["source_dir"],
            "resized_image_dir": first["image_dir"],
            "lora_cache_dir": first["cache_dir"],
        }
        ok, msg, _train_path, next_content, _changed = _prepare_raw_file_patch(train_rel, values, content=train_content)
        if not ok:
            raise ValueError(msg)

    dataset_doc = _build_dataset_config_doc(
        clean_rows,
        cfg,
        prefer_train_batch_size=True,
    )
    dataset_existed = dataset_path.exists()
    previous_dataset_doc = dataset_path.read_text(encoding="utf-8") if dataset_existed else ""
    ok, msg = save_raw_file(dataset_rel, dataset_doc, overwrite=True)
    if not ok:
        raise ValueError(msg)
    if train_rel:
        ok, msg = save_raw_file(train_rel, next_content, overwrite=True)
        if not ok:
            _restore_dataset_config_after_failed_train_patch(dataset_path, dataset_existed, previous_dataset_doc)
            raise ValueError(msg)

    return {
        "ok": True,
        "message": f"已保存 {len(clean_rows)} 个数据集路径",
        "dataset_config": dataset_rel,
        "datasets": clean_rows,
        "defaults": _normalize_dataset_defaults(cfg),
        "train_content": next_content,
    }


def apply_auto_data_dirs(cfg: dict[str, Any], *, create: bool = False) -> dict[str, Any]:
    next_cfg = dict(cfg)
    source_raw = str(next_cfg.get("source_image_dir") or "").strip()
    if not source_raw:
        return next_cfg
    source_path = _resolve_project_path(source_raw)
    resized_path = _auto_data_dir_for_key(next_cfg.get("resized_image_dir"), source_path, "resized")
    cache_path = _auto_data_dir_for_key(next_cfg.get("lora_cache_dir"), source_path, "lora_cache")
    next_cfg["resized_image_dir"] = _display_path(resized_path)
    next_cfg["lora_cache_dir"] = _display_path(cache_path)
    if create:
        resized_path.mkdir(parents=True, exist_ok=True)
        cache_path.mkdir(parents=True, exist_ok=True)
    return next_cfg


def preflight_training_config(
    variant: str,
    preset: str,
    methods_subdir: str = "gui-methods",
    *,
    config_file: str | None = None,
) -> dict[str, Any]:
    cfg = _load_training_config_for_web_run(variant, preset, methods_subdir, config_file=config_file)
    checks: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    runtime_config = is_web_runtime_config(config_file) or _looks_like_web_runtime_config(cfg)

    def add(level: str, key: str, message: str, path: Path | None = None) -> None:
        item = {
            "level": level,
            "key": key,
            "message": message,
        }
        if path is not None:
            item["path"] = _display_path(path)
        checks.append(item)
        if level == "error":
            errors.append(item)
        elif level == "warning":
            warnings.append(item)

    def check_file(key: str, label: str, suffixes: tuple[str, ...] = ()) -> None:
        raw = cfg.get(key)
        if not raw:
            add("error", key, f"{label} 未填写")
            return
        path = _resolve_project_path(str(raw))
        if not path.exists():
            add("error", key, f"{label} 不存在", path)
            return
        if not path.is_file():
            add("error", key, f"{label} 不是文件", path)
            return
        if suffixes and path.suffix.lower() not in suffixes:
            add("warning", key, f"{label} 后缀不是常见格式 {', '.join(suffixes)}", path)
            return
        add("ok", key, f"{label} 存在", path)

    def check_dir(key: str, label: str, *, must_exist: bool, warn_empty: bool = False) -> None:
        raw = cfg.get(key)
        if not raw:
            add("error", key, f"{label} 未填写")
            return
        path = _resolve_project_path(str(raw))
        if not path.exists():
            if must_exist:
                add("error", key, f"{label} 不存在", path)
            else:
                add("warning", key, f"{label} 不存在，训练/预处理可能会创建它", path)
            return
        if not path.is_dir():
            add("error", key, f"{label} 不是目录", path)
            return
        if warn_empty and not any(path.iterdir()):
            add("warning", key, f"{label} 为空", path)
            return
        add("ok", key, f"{label} 存在", path)

    check_file("pretrained_model_name_or_path", "基础 DiT 模型", (".safetensors", ".pt", ".pth", ".ckpt"))
    check_file("qwen3", "Qwen3 文本编码器", (".safetensors", ".pt", ".pth", ".bin"))
    check_file("vae", "VAE 模型", (".safetensors", ".pt", ".pth", ".ckpt"))
    dataset_config_path = _dataset_config_path_from_cfg(cfg)
    if cfg.get("dataset_config") and (runtime_config or (dataset_config_path and dataset_config_path.exists())):
        check_file("dataset_config", "数据集配置", (".toml",))

    _check_dataset_source_paths(cfg, add)
    _check_dataset_paths(cfg, add, check_runtime_dirs=runtime_config)
    if not runtime_config:
        _check_web_preprocess_environment(add)
    if runtime_config:
        _check_training_images(cfg, add)
        _check_cache_sidecars(cfg, add)

    return {
        "ok": not errors,
        "variant": variant,
        "preset": preset,
        "methods_subdir": methods_subdir,
        "summary": {
            "errors": len(errors),
            "warnings": len(warnings),
            "checks": len(checks),
        },
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }


def _load_training_config_for_web_run(
    variant: str,
    preset: str,
    methods_subdir: str,
    *,
    config_file: str | None = None,
) -> dict[str, Any]:
    cfg = apply_auto_data_dirs(load_merged_config(variant, preset, methods_subdir))
    source = _config_file_path(config_file)
    if source is not None:
        try:
            cfg.update(expand_env_vars_in_obj(toml.loads(source.read_text(encoding="utf-8"))))
        except toml.TomlDecodeError as exc:
            raise ValueError(f"训练配置 TOML 解析失败: {config_file}") from exc
    return apply_auto_data_dirs(cfg)


def _config_file_path(config_file: str | None) -> Path | None:
    raw = str(config_file or "").strip()
    if not raw:
        return None
    path = Path(raw.replace("\\", "/"))
    if ".." in path.parts:
        raise ValueError("训练配置路径不能包含 ..")
    if path.is_absolute():
        resolved = path.resolve()
    else:
        normalized = _normalize_config_rel_path(raw)
        resolved = (ROOT / normalized).resolve()
    if _is_output_run_snapshot_config(resolved) and resolved.name != OUTPUT_RUN_CONFIG_FILES["runtime"][0]:
        raise ValueError("训练输出目录只能使用 config.runtime.toml 作为训练配置")
    if not _is_allowed_training_config_path(resolved):
        raise ValueError("训练配置必须在项目目录或全局输出文件夹内")
    if not resolved.is_file():
        raise FileNotFoundError(f"训练配置不存在: {config_file}")
    if resolved.suffix.lower() != ".toml":
        raise ValueError("训练配置必须是 TOML 文件")
    return resolved


def _is_allowed_training_config_path(path: Path) -> bool:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
        return True
    except ValueError:
        pass
    try:
        rel_to_output = resolved.relative_to(resolve_output_root().resolve())
    except ValueError:
        return _is_web_runtime_config_tree(resolved)
    return (
        len(rel_to_output.parts) == 2
        and rel_to_output.name == "config.runtime.toml"
        and _is_web_runtime_config_tree(resolved)
    )


def _is_web_runtime_config_tree(path: Path) -> bool:
    run_dir = path.parent
    return (
        path.name == "config.runtime.toml"
        and path.is_file()
        and _has_web_runtime_dirs(run_dir)
    )


def _is_output_run_snapshot_config(path: Path) -> bool:
    resolved = path.resolve()
    if resolved.name not in {filename for filename, _label in OUTPUT_RUN_CONFIG_FILES.values()}:
        return False
    try:
        rel_to_output = resolved.relative_to(resolve_output_root().resolve())
        if len(rel_to_output.parts) == 2:
            return True
    except ValueError:
        pass
    return _has_web_runtime_dirs(resolved.parent)


def _has_web_runtime_dirs(run_dir: Path) -> bool:
    return (
        (run_dir / "model_cache").is_dir()
        and (run_dir / "dataset_cache").is_dir()
        and (run_dir / "training_output").is_dir()
    )


def is_web_runtime_config(config_file: str | None) -> bool:
    path = _config_file_path(config_file)
    if path is None:
        return False
    run_dir = path.parent
    return (
        path.name == "config.runtime.toml"
        and (run_dir / "model_cache").is_dir()
        and (run_dir / "dataset_cache").is_dir()
        and (run_dir / "training_output").is_dir()
    )


def _looks_like_web_runtime_config(cfg: dict[str, Any]) -> bool:
    output_root = resolve_output_root().resolve()
    for key in ("output_dir", "logging_dir", "dataset_config", "resized_image_dir", "lora_cache_dir"):
        raw = str(cfg.get(key) or "").strip()
        if not raw:
            continue
        path = _resolve_project_path(raw)
        try:
            path.relative_to(output_root)
            return True
        except ValueError:
            continue
    return False


def _check_web_preprocess_environment(add) -> None:
    python_exe = Path(_web_python_executable())
    if not python_exe.is_file():
        add(
            "error",
            PREPROCESS_ENV_CHECK_KEY,
            f"预处理启动环境异常: Python 解释器不存在 {python_exe}",
            python_exe,
        )
        return
    missing = [rel for rel in PREPROCESS_ENV_REQUIRED_FILES if not (ROOT / rel).is_file()]
    if missing:
        add(
            "error",
            PREPROCESS_ENV_CHECK_KEY,
            f"预处理启动环境异常: 缺少 {', '.join(missing)}",
            ROOT / missing[0],
        )
        return
    add("ok", PREPROCESS_ENV_CHECK_KEY, "预处理启动环境文件检查通过", ROOT)


def _web_python_executable() -> str:
    venv_python = ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def estimate_training_steps(
    variant: str,
    preset: str,
    methods_subdir: str = "gui-methods",
    dataset_config: str | None = None,
) -> dict[str, Any]:
    cfg = apply_auto_data_dirs(load_merged_config(variant, preset, methods_subdir))
    if dataset_config is not None:
        dataset_rel = _normalize_config_rel_path(str(dataset_config or ""))
        if dataset_rel:
            cfg["dataset_config"] = dataset_rel
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
        src_count = _count_images(source_dir, image_exts)
        resized_count = _count_images(resized_dir, image_exts)
        used_count = resized_count or src_count
        source_images += src_count
        resized_images += resized_count
        train_images += used_count
        weighted_images += used_count * repeats
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
            "uses_preprocessed_images": resized_count > 0,
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


def _dataset_config_path_from_cfg(cfg: dict[str, Any]) -> Path | None:
    rel_path = str(cfg.get("dataset_config") or "").strip()
    if not rel_path:
        return None
    path = _resolve_project_path(rel_path)
    if path.suffix.lower() != ".toml":
        return None
    if not _is_allowed_dataset_config_path(path):
        return None
    return path


def _is_allowed_dataset_config_path(path: Path) -> bool:
    resolved = path.resolve()
    for root in (ROOT.resolve(), resolve_output_root().resolve()):
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    if resolved.name == "dataset.runtime.toml":
        run_dir = resolved.parent
        return (
            resolved.is_file()
            and (run_dir / "config.runtime.toml").is_file()
            and (run_dir / "model_cache").is_dir()
            and (run_dir / "dataset_cache").is_dir()
            and (run_dir / "training_output").is_dir()
        )
    return False


def _dataset_config_rel_path(
    cfg: dict[str, Any],
    variant: str,
    methods_subdir: str,
    *,
    prefer_existing: bool = True,
) -> str:
    existing = str(cfg.get("dataset_config") or "").strip()
    if prefer_existing and existing:
        normalized = _normalize_config_rel_path(existing)
        path = _safe_resolve(normalized)
        if path is not None and normalized.startswith("configs/datasets/"):
            return normalized
    stem = _safe_file_stem(variant or methods_subdir or "dataset")
    return f"configs/datasets/{stem}.toml"


def _training_config_rel_path(variant: str, methods_subdir: str) -> str:
    methods_dir = _safe_config_subdir(methods_subdir)
    if methods_dir is None:
        return ""
    stem = _safe_file_stem(variant)
    path = methods_dir / f"{stem}.toml"
    if not path.exists():
        return ""
    return _display_path(path)


def _single_dataset_config_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    source_dir = str(cfg.get("source_image_dir") or "image_dataset")
    image_dir = str(cfg.get("resized_image_dir") or DEFAULT_RESIZED_IMAGE_DIR)
    cache_dir = str(cfg.get("lora_cache_dir") or DEFAULT_LORA_CACHE_DIR)
    return {
        "general": {
            "caption_extension": ".txt",
            "keep_tokens": 3,
            "prefer_json_caption": False,
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
                "validation_split": 0.025,
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
        "enable_bucket": bool(_first_dataset_value(data, "enable_bucket", True)),
        "min_bucket_reso": _positive_int(_first_dataset_value(data, "min_bucket_reso"), 256),
        "max_bucket_reso": _positive_int(_first_dataset_value(data, "max_bucket_reso"), 1024),
        "bucket_reso_steps": _positive_int(_first_dataset_value(data, "bucket_reso_steps"), 64),
        "bucket_no_upscale": bool(_first_dataset_value(data, "bucket_no_upscale", False)),
        "validation_split": _positive_float(_first_dataset_value(data, "validation_split", 0.025), 0.025),
        "validation_split_num": _positive_int(_first_dataset_value(data, "validation_split_num", 0), 0),
        "validation_seed": _positive_int(_first_dataset_value(data, "validation_seed", 42), 42),
        "caption_extension": str((data.get("general") or {}).get("caption_extension") or ".txt"),
        "keep_tokens": _positive_int((data.get("general") or {}).get("keep_tokens"), 3),
        "prefer_json_caption": _bool_value((data.get("general") or {}).get("prefer_json_caption"), False),
    }


def _dataset_defaults_from_dataset(dataset: dict[str, Any], data: dict[str, Any] | None = None) -> dict[str, Any]:
    source: dict[str, Any] = {"datasets": [dataset]}
    if isinstance(data, dict) and isinstance(data.get("general"), dict):
        source["general"] = data["general"]
    return _dataset_defaults_from_config(source)


def _dataset_preset_summary(rel_path: str) -> dict[str, Any]:
    try:
        preset = load_dataset_preset(rel_path)
    except Exception as e:
        return {"ok": False, "error": str(e), "dataset_count": 0}
    return preset.get("summary") or {}


def _dataset_summary_from_rows(rows: list[dict[str, Any]], defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    clean_rows = _normalize_dataset_rows(rows)
    clean_defaults = _normalize_dataset_defaults(defaults or _first_dataset_settings(clean_rows))
    first = clean_rows[0] if clean_rows else {}
    repeats = sum(_positive_int(row.get("num_repeats"), 1) for row in clean_rows) if clean_rows else 0
    return {
        "ok": True,
        "dataset_count": len(clean_rows),
        "repeat_total": repeats,
        "source_dir": first.get("source_dir", ""),
        "image_dir": first.get("image_dir", ""),
        "cache_dir": first.get("cache_dir", ""),
        "resolution": clean_defaults.get("resolution", 1024),
        "batch_size": clean_defaults.get("batch_size", 1),
        "enable_bucket": clean_defaults.get("enable_bucket", True),
    }


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


def _dataset_rows_from_config(data: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    datasets = data.get("datasets") if isinstance(data, dict) else []
    if not isinstance(datasets, list):
        datasets = []

    fallback_source = str(cfg.get("source_image_dir") or "")
    fallback_image = str(cfg.get("resized_image_dir") or fallback_source)
    fallback_cache = str(cfg.get("lora_cache_dir") or "")

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
            "settings": _normalize_dataset_row_settings(raw),
        })
    return clean_rows


def _normalize_dataset_row_settings(raw: dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw.get("settings"), dict):
        return _normalize_dataset_defaults(raw["settings"])
    if any(key in raw for key in DATASET_SETTING_KEYS):
        return _normalize_dataset_defaults(raw)
    return {}


def _fill_missing_dataset_row_settings(rows: list[dict[str, Any]], defaults: dict[str, Any]) -> list[dict[str, Any]]:
    fallback = _normalize_dataset_defaults(defaults)
    next_rows: list[dict[str, Any]] = []
    for row in rows:
        next_row = dict(row)
        settings = next_row.get("settings")
        next_row["settings"] = _normalize_dataset_defaults(settings) if isinstance(settings, dict) and settings else fallback
        next_rows.append(next_row)
    return next_rows


def _normalize_dataset_defaults(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    out["resolution"] = _positive_int(raw.get("resolution"), 1024)
    out["batch_size"] = _positive_int(raw.get("batch_size"), 1)
    out["enable_bucket"] = str(raw.get("enable_bucket", True)).lower() not in {"0", "false", "no", "off"}
    out["min_bucket_reso"] = _positive_int(raw.get("min_bucket_reso"), 256)
    out["max_bucket_reso"] = _positive_int(raw.get("max_bucket_reso"), 1024)
    out["bucket_reso_steps"] = _positive_int(raw.get("bucket_reso_steps"), 64)
    out["bucket_no_upscale"] = str(raw.get("bucket_no_upscale", False)).lower() in {"1", "true", "yes", "on"}
    if raw.get("validation_split_num") not in (None, ""):
        out["validation_split_num"] = _positive_int(raw.get("validation_split_num"), 0)
    out["validation_split"] = _positive_float(raw.get("validation_split"), 0.025)
    out["validation_seed"] = _positive_int(raw.get("validation_seed"), 42)
    out["caption_extension"] = str(raw.get("caption_extension") or ".txt").strip() or ".txt"
    out["keep_tokens"] = _positive_int(raw.get("keep_tokens"), 3)
    out["prefer_json_caption"] = _bool_value(raw.get("prefer_json_caption"), False)
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
    general.add("keep_tokens", _positive_int(cfg.get("keep_tokens"), 3))
    if _bool_value(cfg.get("prefer_json_caption"), False):
        general.add("prefer_json_caption", True)
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
        if include_preprocess_settings:
            dataset.add("enable_bucket", bool(row_cfg.get("enable_bucket", True)))
            dataset.add("min_bucket_reso", _positive_int(row_cfg.get("min_bucket_reso"), 256))
            dataset.add("max_bucket_reso", _positive_int(row_cfg.get("max_bucket_reso"), 1024))
            dataset.add("bucket_reso_steps", _positive_int(row_cfg.get("bucket_reso_steps"), 64))
            dataset.add("bucket_no_upscale", bool(row_cfg.get("bucket_no_upscale", False)))
        validation_split_num = _positive_int(row_cfg.get("validation_split_num"), 0)
        if validation_split_num > 0:
            dataset.add("validation_split_num", validation_split_num)
        dataset.add("validation_split", _positive_float(row_cfg.get("validation_split"), 0.025))
        dataset.add("validation_seed", _positive_int(row_cfg.get("validation_seed"), 42))

        subsets = tomlkit.aot()
        subset = tomlkit.table()
        subset.add("image_dir", row["image_dir"])
        subset.add("cache_dir", row["cache_dir"])
        subset.add("num_repeats", _positive_int(row.get("num_repeats"), 1))
        attrs = tomlkit.inline_table()
        attrs.add("source_dir", row["source_dir"])
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


def _list_dataset_image_files(directory: Path, limit: int) -> dict[str, Any]:
    clean_limit = max(1, min(_positive_int(limit, DATASET_PREVIEW_LIMIT), DATASET_PREVIEW_LIMIT))
    if not directory.is_dir():
        return {"items": [], "total": 0, "limit": clean_limit}
    items = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in DATASET_IMAGE_EXTS
    ]
    items.sort(key=lambda path: (path.name.lower(), path.name))
    return {"items": items[:clean_limit], "total": len(items), "limit": clean_limit}


def _dataset_image_preview_meta(
    path: Path,
    *,
    preset_file: str,
    dataset_index: int,
    source: str,
    caption_extension: str,
    prefer_json_caption: bool,
    source_dir: Path,
    train_dir: Path,
) -> dict[str, Any]:
    stat = path.stat()
    caption = _dataset_caption_meta(
        path,
        caption_extension,
        source_dir,
        train_dir,
        prefer_json_caption=prefer_json_caption,
    )
    dimensions = _dataset_image_dimensions(path)
    rel_path = _display_path(path)
    url = (
        "/api/config/dataset-presets/image"
        f"?file={quote(preset_file)}"
        f"&dataset_index={dataset_index}"
        f"&source={quote(source)}"
        f"&image={quote(rel_path)}"
    )
    return {
        "file": rel_path,
        "name": path.name,
        "url": url,
        "mtime": stat.st_mtime,
        "mtime_text": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "size_bytes": stat.st_size,
        "width": dimensions.get("width"),
        "height": dimensions.get("height"),
        "total_pixels": dimensions.get("total_pixels"),
        "caption": caption,
    }


def _dataset_image_dimensions(path: Path) -> dict[str, int]:
    try:
        with Image.open(path) as image:
            width, height = image.size
    except (OSError, UnidentifiedImageError):
        return {}
    return {
        "width": int(width),
        "height": int(height),
        "total_pixels": int(width) * int(height),
    }


def _dataset_caption_meta(
    path: Path,
    caption_extension: str,
    source_dir: Path,
    train_dir: Path,
    *,
    prefer_json_caption: bool = False,
) -> dict[str, Any]:
    extension = caption_extension if caption_extension.startswith(".") else f".{caption_extension}"
    directories = []
    for directory in (path.parent, source_dir, train_dir):
        if not directory:
            continue
        directory = directory.resolve()
        if directory not in directories:
            directories.append(directory)

    if prefer_json_caption:
        for directory in directories:
            candidate = directory / f"{path.stem}.json"
            if not candidate.exists() or not candidate.is_file():
                continue
            try:
                text = load_json_caption(candidate).render()
            except Exception:
                continue
            truncated = len(text) > DATASET_CAPTION_MAX_CHARS
            if truncated:
                text = text[:DATASET_CAPTION_MAX_CHARS]
            return {
                "ok": True,
                "file": _display_path(candidate),
                "extension": ".json",
                "text": text,
                "truncated": truncated,
                "length": len(text),
            }

    candidates = []
    for directory in directories:
        candidate = (directory / f"{path.stem}{extension}").resolve()
        if candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        if not candidate.exists() or not candidate.is_file():
            continue
        try:
            text = candidate.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = candidate.read_text(encoding="utf-8", errors="replace")
        truncated = len(text) > DATASET_CAPTION_MAX_CHARS
        if truncated:
            text = text[:DATASET_CAPTION_MAX_CHARS]
        return {
            "ok": True,
            "file": _display_path(candidate),
            "extension": extension,
            "text": text,
            "truncated": truncated,
            "length": len(text),
        }
    return {
        "ok": False,
        "file": "",
        "extension": extension,
        "text": "",
        "truncated": False,
        "length": 0,
    }


def _dataset_preview_empty_message(directory: Path, source: str) -> str:
    label = "原始图目录" if source == "source" else "训练图目录"
    if not directory.exists():
        return f"{label}不存在"
    if not directory.is_dir():
        return f"{label}不是目录"
    return f"{label}里没有可预览图片"


def _safe_file_stem(value: str) -> str:
    stem = Path(str(value or "").replace("\\", "/")).stem
    chars: list[str] = []
    for ch in stem:
        if ch.isascii() and (ch.isalnum() or ch in {"-", "_"}):
            chars.append(ch)
        elif ch.isspace():
            chars.append("_")
    return "".join(chars).strip("_-") or "dataset"


def _count_images(path: Path, image_exts: set[str]) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in image_exts)


def _dataset_num_repeats(cfg: dict[str, Any]) -> int:
    dataset_config = cfg.get("dataset_config")
    if dataset_config:
        path = _safe_resolve(_normalize_config_rel_path(str(dataset_config)))
        if path is not None and path.exists():
            try:
                data = toml.loads(path.read_text(encoding="utf-8"))
                repeats = []
                for dataset in data.get("datasets") or []:
                    for subset in dataset.get("subsets") or []:
                        repeats.append(_positive_int(subset.get("num_repeats"), 1))
                return max(1, sum(repeats) or 1)
            except Exception:
                return 1
    return 1


def _positive_int(value: Any, fallback: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return fallback
    return n if n > 0 else fallback


def _nonnegative_int(value: Any, fallback: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return fallback
    return n if n >= 0 else fallback


def _positive_float(value: Any, fallback: float) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return fallback
    return n if n > 0 else fallback


def _bool_value(value: Any, fallback: bool = False) -> bool:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def list_output_runs(limit: int = 200) -> dict[str, Any]:
    output_root = resolve_output_root()
    root_display = _display_settings_path(output_root)
    if not output_root.exists():
        return {
            "ok": True,
            "output_root": root_display,
            "output_root_abs": str(output_root),
            "runs": [],
        }
    if not output_root.is_dir():
        raise ValueError(f"输出文件夹不是目录: {root_display}")

    runs: list[dict[str, Any]] = []
    for child in output_root.iterdir():
        if not child.is_dir():
            continue
        try:
            child.resolve().relative_to(output_root.resolve())
        except ValueError:
            continue
        summary = _output_run_summary(child)
        if summary["files"]:
            runs.append(summary)

    runs.sort(key=lambda item: (float(item.get("mtime") or 0), str(item.get("name") or "")), reverse=True)
    return {
        "ok": True,
        "output_root": root_display,
        "output_root_abs": str(output_root),
        "runs": runs[:max(1, int(limit or 200))],
    }


def load_output_run_config(run: str, kind: str) -> dict[str, Any]:
    run_dir = _resolve_output_run_dir(run)
    file_path = _output_run_config_path(run_dir, kind)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"运行配置不存在: {run_dir.name}/{file_path.name}")
    return {
        "ok": True,
        "run": run_dir.name,
        "kind": kind,
        "label": OUTPUT_RUN_CONFIG_FILES[kind][1],
        "file": _display_settings_path(file_path),
        "content": file_path.read_text(encoding="utf-8", errors="replace"),
        "readonly": True,
    }


def save_output_run_config_as(run: str, name: str, target_group: str | None = None) -> dict[str, Any]:
    run_dir = _resolve_output_run_dir(run)
    original_path = run_dir / OUTPUT_RUN_CONFIG_FILES["original"][0]
    if not original_path.exists() or not original_path.is_file():
        raise ValueError("这个运行目录没有 config.original.toml，不能复制为项目预设")
    content = original_path.read_text(encoding="utf-8", errors="replace")
    try:
        tomllib.loads(content)
        tomlkit.parse(content)
    except (tomllib.TOMLDecodeError, tomlkit.exceptions.TOMLKitError) as e:
        raise ValueError(f"TOML 语法错误: {e}") from e

    target = _normalize_output_run_save_as_path(name, fallback_stem=run_dir.name)
    normalized_group = _normalize_group_id(target_group or "")
    if normalized_group:
        groups = {str(group.get("id") or ""): group for group in list_config_file_groups()}
        group = groups.get(normalized_group)
        if not group or not group.get("movable") or group.get("locked"):
            raise ValueError("目标分组不可用或已锁定")

    ok, msg = save_raw_file(target, content, overwrite=False)
    if not ok:
        raise ValueError(msg)

    group_meta = None
    if normalized_group:
        moved, move_msg, group_meta = move_config_file_to_group(target, normalized_group)
        if not moved:
            raise ValueError(move_msg)

    return {
        "ok": True,
        "message": "已复制为新项目预设",
        "run": run_dir.name,
        "file": target,
        "meta": get_config_file_meta(target),
        "group": group_meta,
    }


def _output_run_summary(run_dir: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    mtimes = [_safe_mtime(run_dir)]
    for kind, (filename, label) in OUTPUT_RUN_CONFIG_FILES.items():
        path = run_dir / filename
        if not path.is_file():
            continue
        mtime = _safe_mtime(path)
        mtimes.append(mtime)
        files.append({
            "kind": kind,
            "label": label,
            "filename": filename,
            "file": _display_settings_path(path),
            "mtime": mtime,
            "mtime_text": _format_file_time(mtime),
        })
    mtime = max(mtimes) if mtimes else 0.0
    return {
        "name": run_dir.name,
        "path": _display_settings_path(run_dir),
        "mtime": mtime,
        "mtime_text": _format_file_time(mtime),
        "files": files,
        "has_original": any(item["kind"] == "original" for item in files),
        "has_runtime": any(item["kind"] == "runtime" for item in files),
        "has_dataset": any(item["kind"] == "dataset" for item in files),
    }


def _resolve_output_run_dir(run: str) -> Path:
    name = _normalize_output_run_name(run)
    root = resolve_output_root()
    candidate = root / name
    if not candidate.exists() or not candidate.is_dir():
        raise FileNotFoundError(f"运行目录不存在: {name}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("运行目录必须位于输出文件夹内") from exc
    return resolved


def _normalize_output_run_name(run: str) -> str:
    name = str(run or "").replace("\\", "/").strip()
    if not name or "/" in name or name in {".", ".."} or ".." in Path(name).parts:
        raise ValueError("run 参数只允许输出文件夹下的直接目录名")
    return name


def _output_run_config_path(run_dir: Path, kind: str) -> Path:
    normalized = str(kind or "").strip()
    if normalized not in OUTPUT_RUN_CONFIG_FILES:
        raise ValueError("kind 只能是 original、runtime 或 dataset")
    return run_dir / OUTPUT_RUN_CONFIG_FILES[normalized][0]


def _normalize_output_run_save_as_path(value: str, *, fallback_stem: str) -> str:
    raw = str(value or "").replace("\\", "/").strip()
    if not raw:
        raw = fallback_stem
    path = Path(raw)
    if path.is_absolute():
        try:
            path = path.resolve().relative_to(ROOT.resolve())
        except ValueError as exc:
            raise ValueError("新项目预设必须保存在项目目录内") from exc
    if ".." in path.parts:
        raise ValueError("新项目预设路径不能包含 ..")
    if path.suffix.lower() != ".toml":
        path = path.with_suffix(".toml")
    if len(path.parts) == 1:
        path = Path("configs") / "imported" / path.name
    normalized = path.as_posix().lstrip("/")
    if not normalized.startswith("configs/imported/") or Path(normalized).name in {"", ".toml"}:
        raise ValueError("新项目预设必须保存到 configs/imported/ 下")
    safe_path = _safe_resolve(normalized)
    if safe_path is None:
        raise ValueError("新项目预设路径不合法")
    return normalized


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _format_file_time(value: float) -> str:
    if not value:
        return ""
    return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def load_raw_file(rel_path: str) -> str:
    path = _safe_resolve(_normalize_config_rel_path(rel_path))
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def load_sample_prompts_file(rel_path: str | None = None) -> dict[str, Any]:
    normalized = _normalize_prompt_file_path(rel_path or DEFAULT_SAMPLE_PROMPTS_FILE)
    path = (ROOT / normalized).resolve()
    if not path.exists():
        return {"ok": True, "file": normalized, "content": "", "prompts": []}
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    prompts = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    return {
        "ok": True,
        "file": normalized,
        "content": content,
        "prompts": prompts,
    }


def save_sample_prompts_file(
    content: str,
    rel_path: str | None = None,
    *,
    train_config_file: str | None = None,
) -> dict[str, Any]:
    normalized = _normalize_prompt_file_path(rel_path or DEFAULT_SAMPLE_PROMPTS_FILE)
    if train_config_file:
        normalized = _sample_prompts_path_for_config(train_config_file)
    text = str(content or "")
    lines = text.splitlines()
    prompts = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    path = (ROOT / normalized).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return {
        "ok": True,
        "file": normalized,
        "content": text,
        "prompts": prompts,
        "message": f"已保存 {len(prompts)} 条预览提示词",
    }


def save_raw_file(
    rel_path: str,
    content: str,
    *,
    allow_locked: bool = False,
    overwrite: bool = True,
) -> tuple[bool, str]:
    normalized = _normalize_config_rel_path(rel_path)
    path = _safe_resolve(normalized)
    if path is None:
        return False, "路径不合法"
    if path.exists() and not overwrite:
        return False, "配置文件已存在，请换一个新的名称"
    meta = get_config_file_meta(normalized)
    if meta.get("locked") and not allow_locked:
        return False, f"{_lock_reason_message(meta)}，请使用新名称保存新配置后编辑"
    try:
        toml.loads(content)
    except toml.TomlDecodeError as e:
        return False, f"TOML 语法错误: {e}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True, "保存成功"


def delete_raw_file(rel_path: str) -> tuple[bool, str]:
    normalized = _normalize_config_rel_path(rel_path)
    path = _safe_resolve(normalized)
    if path is None or path.suffix.lower() != ".toml":
        return False, "路径不合法，只能删除 configs/ 下的 TOML 文件"
    if not path.exists():
        return False, "配置文件不存在或已被删除"
    if not path.is_file():
        return False, "目标不是文件，已拒绝删除"

    meta = get_config_file_meta(normalized)
    if meta.get("locked"):
        return False, f"{_lock_reason_message(meta)}，不能删除"

    try:
        path.unlink()
    except OSError as e:
        return False, f"删除失败: {e}"

    user_locks, user_group_locks = _load_user_locks()
    if normalized in user_locks:
        user_locks.discard(normalized)
        _save_user_locks(user_locks, user_group_locks)

    return True, "删除成功"


def patch_raw_file_values(
    rel_path: str,
    values: dict[str, Any],
    *,
    content: str | None = None,
) -> tuple[bool, str, str, list[str]]:
    ok, msg, path, next_content, changed = _prepare_raw_file_patch(rel_path, values, content=content)
    if not ok or path is None:
        return False, msg, "", []
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(next_content, encoding="utf-8")
    return True, "保存成功", next_content, changed


def preview_raw_file_patch(
    rel_path: str,
    values: dict[str, Any],
    *,
    content: str | None = None,
) -> tuple[bool, str, str, list[str]]:
    ok, msg, _path, next_content, changed = _prepare_raw_file_patch(rel_path, values, content=content)
    if not ok:
        return False, msg, "", []
    return True, "预览成功", next_content, changed


def _prepare_raw_file_patch(
    rel_path: str,
    values: dict[str, Any],
    *,
    content: str | None = None,
) -> tuple[bool, str, Path | None, str, list[str]]:
    normalized = _normalize_config_rel_path(rel_path)
    path = _safe_resolve(normalized)
    if path is None:
        return False, "路径不合法", None, "", []
    meta = get_config_file_meta(normalized)
    if meta.get("locked"):
        return False, f"{_lock_reason_message(meta)}，请使用新名称保存新配置后编辑", None, "", []
    if not isinstance(values, dict):
        return False, "字段补丁格式不合法", None, "", []
    values = {
        key: value
        for key, value in values.items()
        if key not in UI_ONLY_CONFIG_FIELDS
    }

    source = content if content is not None else load_raw_file(rel_path)
    try:
        next_content = _patch_toml_top_level(source, values)
        toml.loads(next_content)
    except Exception as e:
        return False, f"TOML 更新失败: {e}", None, "", []

    return True, "保存成功", path, next_content, sorted(values.keys())


def _restore_dataset_config_after_failed_train_patch(path: Path, existed: bool, previous_content: str) -> None:
    if existed:
        path.write_text(previous_content, encoding="utf-8")
        return
    try:
        path.unlink()
    except FileNotFoundError:
        return


def set_user_file_lock(rel_path: str, locked: bool) -> tuple[bool, str, dict[str, Any]]:
    normalized = _normalize_config_rel_path(rel_path)
    path = _safe_resolve(normalized)
    if path is None or path.suffix != ".toml":
        return False, "路径不合法，只能锁定 configs/ 下的 TOML 文件", {}
    if not path.exists():
        return False, "只能锁定已经存在的 TOML 文件", {}

    meta = get_config_file_meta(normalized)
    if meta.get("system_locked"):
        return False, "系统预设为内置只读，不能手动锁定或解锁", meta
    if meta.get("group_locked"):
        return False, "该文件属于只读分组，不能手动锁定或解锁", meta
    if meta.get("user_group_locked"):
        return False, "该文件所在分组已锁定，请先解除分组锁定", meta

    user_locks, user_group_locks = _load_user_locks()
    if locked:
        user_locks.add(normalized)
    else:
        user_locks.discard(normalized)
    _save_user_locks(user_locks, user_group_locks)

    next_meta = get_config_file_meta(normalized)
    return True, ("已锁定当前文件" if locked else "已解除用户锁定"), next_meta


def set_user_group_lock(group_id: str, locked: bool) -> tuple[bool, str, dict[str, Any]]:
    normalized = _normalize_group_id(group_id)
    if not normalized:
        return False, "缺少 group 参数", {}

    group = _get_config_file_group(normalized)
    if group is None:
        return False, "分组不存在", {}
    if normalized not in _lockable_group_ids():
        return False, "该分组属于系统或只读参考，不能手动锁定或解锁", group
    if any(item.get("system_locked") for item in group.get("files", [])):
        return False, "该分组包含系统预设，不能手动锁定或解锁", group

    user_locks, user_group_locks = _load_user_locks()
    if locked:
        user_group_locks.add(normalized)
    else:
        user_group_locks.discard(normalized)
    _save_user_locks(user_locks, user_group_locks)

    next_group = _get_config_file_group(normalized) or group
    return True, ("已锁定当前分组" if locked else "已解除分组锁定"), next_group


def create_config_file_group(label: str) -> tuple[bool, str, dict[str, Any] | None]:
    clean_label = _normalize_group_label(label)
    if not clean_label:
        return False, "分组名称不能为空", None

    specs = _load_config_file_group_specs()
    group_id = _unique_group_id(_slugify_group_label(clean_label), specs)
    spec = _new_user_config_group_spec(group_id, clean_label)
    specs.append(spec)
    _save_config_file_group_specs(specs)
    return True, "分组已创建", _build_config_file_group(spec)


def rename_config_file_group(group_id: str, label: str) -> tuple[bool, str, dict[str, Any] | None]:
    normalized = _normalize_group_id(group_id)
    clean_label = _normalize_group_label(label)
    if not normalized:
        return False, "缺少 group 参数", None
    if not clean_label:
        return False, "分组名称不能为空", None

    specs = _load_config_file_group_specs()
    spec = _find_config_group_spec(specs, normalized)
    if spec is None:
        return False, "分组不存在", None
    if not _is_renamable_config_group(spec):
        return False, "系统固定或只读分组不能重命名", _build_config_file_group(spec)

    spec["label"] = clean_label
    _save_config_file_group_specs(specs)
    return True, "分组已重命名", _build_config_file_group(spec)


def delete_config_file_group(group_id: str) -> tuple[bool, str]:
    normalized = _normalize_group_id(group_id)
    if not normalized:
        return False, "缺少 group 参数"

    specs = _load_config_file_group_specs()
    spec = _find_config_group_spec(specs, normalized)
    if spec is None:
        return False, "分组不存在"
    if _is_user_group_locked(normalized):
        return False, "该分组已锁定，请先解除分组锁定后再删除"
    if not _is_deletable_config_group(spec):
        return False, "系统或只读分组不能删除"

    released_files = {item["path"] for item in _build_config_file_group(spec).get("files", [])}
    released_files.update(spec.get("files", []))
    released_files.update(spec.get("order", []))
    specs = [item for item in specs if item["id"] != normalized]
    if released_files:
        for item in specs:
            item["exclude"] = set(item.get("exclude", set())) - released_files
        _move_orphaned_config_files_to_fallback_groups(specs, sorted(released_files))
    user_locks, user_group_locks = _load_user_locks()
    if normalized in user_group_locks:
        user_group_locks.discard(normalized)
        _save_user_locks(user_locks, user_group_locks)
    _save_config_file_group_specs(specs)
    return True, "分组已删除，TOML 文件已保留在其他可见分组中"


def reorder_config_file_group(group_id: str, direction: str) -> tuple[bool, str, dict[str, Any] | None]:
    normalized = _normalize_group_id(group_id)
    clean_direction = str(direction or "").strip().lower()
    if not normalized:
        return False, "缺少 group 参数", None
    if clean_direction not in {"up", "down"}:
        return False, "排序方向必须是 up 或 down", None

    specs = _load_config_file_group_specs()
    spec = _find_config_group_spec(specs, normalized)
    if spec is None:
        return False, "分组不存在", None
    if _is_fixed_config_group(spec):
        return False, "系统分组不能调整顺序", _build_config_file_group(spec)

    movable_indices = [
        idx for idx, item in enumerate(specs)
        if not _is_fixed_config_group(item)
    ]
    current_pos = next((idx for idx, item_idx in enumerate(movable_indices) if specs[item_idx]["id"] == normalized), -1)
    if current_pos < 0:
        return False, "分组不在可排序列表中", _build_config_file_group(spec)

    next_pos = current_pos - 1 if clean_direction == "up" else current_pos + 1
    if next_pos < 0 or next_pos >= len(movable_indices):
        return True, "分组顺序未变化", _build_config_file_group(spec)

    current_index = movable_indices[current_pos]
    next_index = movable_indices[next_pos]
    specs[current_index], specs[next_index] = specs[next_index], specs[current_index]
    _save_config_file_group_specs(specs)
    moved = _find_config_group_spec(specs, normalized)
    return True, "分组顺序已更新", _build_config_file_group(moved) if moved else None


def move_config_file_to_group(rel_path: str, group_id: str) -> tuple[bool, str, dict[str, Any] | None]:
    normalized_file = _normalize_config_rel_path(rel_path)
    target_group_id = _normalize_group_id(group_id)
    path = _safe_resolve(normalized_file)
    if path is None or path.suffix.lower() != ".toml" or not path.exists():
        return False, "配置文件不存在或路径不合法", None
    if _is_system_locked_path(normalized_file):
        return False, "系统预设和 Web 管理配置不能移动分组", None
    if not normalized_file.startswith(("configs/imported/", "configs/datasets/")):
        return False, "当前仅支持移动导入配置和数据集配置", None

    specs = _load_config_file_group_specs()
    target = _find_config_group_spec(specs, target_group_id)
    if target is None:
        return False, "目标分组不存在", None
    if not _is_move_target_group(target):
        return False, "只能移动到导入配置、数据集配置或用户自定义分组", _build_config_file_group(target)
    if target.get("locked") or _is_user_group_locked(target_group_id):
        return False, "目标分组已锁定，不能移入配置", _build_config_file_group(target)

    for spec in specs:
        files = [item for item in spec.get("files", []) if item != normalized_file]
        spec["files"] = files
        spec["order"] = [item for item in spec.get("order", []) if item != normalized_file]
        exclude = [item for item in spec.get("exclude", []) if item != normalized_file]
        if _group_patterns_include_file(spec, normalized_file) and spec["id"] != target_group_id:
            exclude.append(normalized_file)
        spec["exclude"] = sorted(dict.fromkeys(exclude))

    target.setdefault("files", [])
    if normalized_file not in target["files"]:
        target["files"].append(normalized_file)
    target["files"] = list(dict.fromkeys(target["files"]))
    target.setdefault("order", [])
    target["order"] = [item for item in target["order"] if item != normalized_file]
    target["order"].append(normalized_file)
    if normalized_file in target.get("exclude", []):
        target["exclude"] = [item for item in target["exclude"] if item != normalized_file]

    _save_config_file_group_specs(specs)
    return True, "配置已移动到分组", _build_config_file_group(target)


def reorder_config_file_in_group(
    rel_path: str,
    group_id: str,
    direction: str,
) -> tuple[bool, str, dict[str, Any] | None]:
    normalized_file = _normalize_config_rel_path(rel_path)
    target_group_id = _normalize_group_id(group_id)
    clean_direction = str(direction or "").strip().lower()
    if clean_direction not in {"up", "down"}:
        return False, "排序方向必须是 up 或 down", None

    path = _safe_resolve(normalized_file)
    if path is None or path.suffix.lower() != ".toml" or not path.exists():
        return False, "配置文件不存在或路径不合法", None

    specs = _load_config_file_group_specs()
    spec = _find_config_group_spec(specs, target_group_id)
    if spec is None:
        return False, "分组不存在", None

    files = [item["path"] for item in _build_config_file_group(spec).get("files", [])]
    if normalized_file not in files:
        return False, "配置文件不在该分组中", _build_config_file_group(spec)

    index = files.index(normalized_file)
    next_index = index - 1 if clean_direction == "up" else index + 1
    if next_index < 0 or next_index >= len(files):
        return True, "排序未变化", _build_config_file_group(spec)

    files[index], files[next_index] = files[next_index], files[index]
    spec["order"] = files
    _save_config_file_group_specs(specs)
    return True, "配置排序已更新", _build_config_file_group(spec)


def restore_system_presets(files: list[str] | None = None) -> dict[str, Any]:
    targets = _list_system_preset_files() if files is None else files
    normalized_targets: list[str] = []
    errors: list[dict[str, str]] = []
    seen: set[str] = set()

    for raw in targets:
        normalized = _normalize_config_rel_path(raw)
        path = _safe_resolve(normalized)
        if path is None or path.suffix != ".toml":
            errors.append({"file": normalized, "reason": "路径不合法"})
            continue
        if not _is_system_preset_path(normalized):
            errors.append({"file": normalized, "reason": "不是系统预设文件"})
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        normalized_targets.append(normalized)

    if errors:
        return {
            "ok": False,
            "error": "还原请求包含不合法文件",
            "restored": [],
            "skipped": [],
            "errors": errors,
            "backup_dir": "",
        }

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_root = CONFIGS_DIR / ".restore-backups" / timestamp
    restored: list[str] = []
    skipped: list[dict[str, str]] = []

    for rel_path in normalized_targets:
        path = _safe_resolve(rel_path)
        if path is None or not path.exists():
            skipped.append({"file": rel_path, "reason": "当前文件不存在"})
            continue

        baseline = _read_git_head_file(rel_path)
        if baseline is None:
            skipped.append({"file": rel_path, "reason": "没有可还原的系统基线"})
            continue

        backup_path = backup_root / _backup_relative_path(rel_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_path)
        path.write_text(baseline, encoding="utf-8")
        restored.append(rel_path)

    return {
        "ok": True,
        "restored": restored,
        "skipped": skipped,
        "errors": [],
        "backup_dir": _display_path(backup_root) if restored else "",
    }


def list_config_files() -> list[str]:
    return [item["path"] for group in list_config_file_groups() for item in group["files"]]


def list_config_file_groups() -> list[dict[str, Any]]:
    specs = _sort_config_file_group_specs_for_display(_load_config_file_group_specs())
    return [_build_config_file_group(spec) for spec in specs]


def _get_config_file_group(group_id: str) -> dict[str, Any] | None:
    normalized = _normalize_group_id(group_id)
    for group in list_config_file_groups():
        if group.get("id") == normalized:
            return group
    return None


def get_config_file_meta(
    rel_path: str,
    group_id: str | None = None,
    group_label: str | None = None,
    locked: bool | None = None,
    trainable: bool | None = None,
    methods_subdir: str | None = None,
) -> dict[str, Any]:
    normalized = _normalize_config_rel_path(rel_path)
    inferred = (
        {
            "id": group_id,
            "label": group_label,
            "locked": locked,
            "open": True,
            "trainable": bool(trainable),
            "methods_subdir": methods_subdir or "",
        }
        if group_id and group_label and locked is not None and trainable is not None
        else _infer_config_file_group(normalized)
    )
    stem = Path(normalized).stem
    group_locked = bool(inferred["locked"] if locked is None else locked)
    system_locked = _is_system_locked_path(normalized)
    user_locked = _is_user_locked(normalized)
    user_group_locked = _is_user_group_locked(group_id or inferred["id"])
    effective_locked = system_locked or user_locked or user_group_locked or group_locked
    lock_reason = ""
    if system_locked:
        lock_reason = "system"
    elif user_locked:
        lock_reason = "user"
    elif user_group_locked:
        lock_reason = "user_group"
    elif group_locked:
        lock_reason = "group"
    return {
        "path": normalized,
        "label": CONFIG_FILE_LABELS_ZH.get(normalized, Path(normalized).name),
        "filename": Path(normalized).name,
        "group": group_id or inferred["id"],
        "group_label": group_label or inferred["label"],
        "locked": effective_locked,
        "group_locked": group_locked,
        "user_group_locked": user_group_locked,
        "system_locked": system_locked,
        "user_locked": user_locked,
        "lock_reason": lock_reason,
        "lock_reason_label": _lock_reason_label(lock_reason),
        "restorable": _is_system_preset_path(normalized),
        "open": inferred["open"],
        "trainable": inferred["trainable"] if trainable is None else trainable,
        "method": stem,
        "methods_subdir": methods_subdir or inferred["methods_subdir"],
    }


def _infer_config_file_group(rel_path: str) -> dict[str, Any]:
    for group in list_config_file_groups():
        for item in group["files"]:
            if item["path"] == rel_path:
                return {
                    "id": group["id"],
                    "label": group["label"],
                    "locked": group["locked"],
                    "open": group["open"],
                    "trainable": group["trainable"],
                    "methods_subdir": group["methods_subdir"],
                }
    if rel_path.startswith("configs/gui-methods/"):
        return _group_defaults("gui_methods", "可训练方法变体", False, True, "gui-methods", True)
    if rel_path.startswith("configs/methods/"):
        return _group_defaults("methods", "系统内置方法配置（锁定只读）", True, True, "methods", False)
    if rel_path.startswith("configs/imported/"):
        return _group_defaults("imported", "导入配置", False, True, "imported", True)
    if rel_path.startswith("configs/datasets/"):
        return _group_defaults("datasets", "数据集配置", False, False, "", False)
    if rel_path in {"configs/base.toml", "configs/presets.toml"}:
        return _group_defaults("presets", "系统预设配置（锁定只读）", True, False, "", False)
    return _group_defaults("custom", "自定义配置", False, False, "", True)


def _load_config_file_group_specs() -> list[dict[str, Any]]:
    data = _load(WEB_FILE_GROUPS_FILE)
    groups = data.get("groups")
    if not isinstance(groups, list):
        groups = _default_config_file_group_specs()
    specs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in groups:
        if not isinstance(raw, dict):
            continue
        group_id = str(raw.get("id") or "").strip()
        if not group_id or group_id in seen_ids:
            continue
        seen_ids.add(group_id)
        specs.append({
            "id": group_id,
            "label": str(raw.get("label") or group_id),
            "open": bool(raw.get("open", True)),
            "locked": bool(raw.get("locked", False)),
            "trainable": bool(raw.get("trainable", False)),
            "methods_subdir": str(raw.get("methods_subdir") or ""),
            "user_managed": bool(raw.get("user_managed", False)),
            "files": _string_list(raw.get("files")),
            "order": _string_list(raw.get("order")),
            "patterns": _string_list(raw.get("patterns")),
            "exclude": set(_string_list(raw.get("exclude"))),
        })
    return specs


def _sort_config_file_group_specs_for_display(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item
        for _idx, item in sorted(
            enumerate(specs),
            key=lambda pair: (1 if _is_fixed_config_group(pair[1]) else 0, pair[0]),
        )
    ]


def _save_config_file_group_specs(specs: list[dict[str, Any]]) -> None:
    doc = tomlkit.document()
    doc.add(tomlkit.comment("Web UI 配置文件管理注册表，由 WebUI 自动维护。"))
    doc.add(tomlkit.comment("系统分组请谨慎修改；user_managed=true 的分组可在 WebUI 中重命名/删除。"))
    group_array = tomlkit.aot()
    for spec in specs:
        table = tomlkit.table()
        table.add("id", spec["id"])
        table.add("label", spec["label"])
        table.add("open", bool(spec.get("open", True)))
        table.add("locked", bool(spec.get("locked", False)))
        table.add("trainable", bool(spec.get("trainable", False)))
        if spec.get("methods_subdir"):
            table.add("methods_subdir", str(spec.get("methods_subdir") or ""))
        if spec.get("user_managed"):
            table.add("user_managed", True)
        if spec.get("files"):
            table.add("files", list(spec.get("files") or []))
        if spec.get("order"):
            table.add("order", list(spec.get("order") or []))
        if spec.get("patterns"):
            table.add("patterns", list(spec.get("patterns") or []))
        if spec.get("exclude"):
            table.add("exclude", list(spec.get("exclude") or []))
        group_array.append(table)
    doc.add("groups", group_array)
    WEB_FILE_GROUPS_FILE.parent.mkdir(parents=True, exist_ok=True)
    WEB_FILE_GROUPS_FILE.write_text(tomlkit.dumps(doc), encoding="utf-8")


def _build_config_file_group(spec: dict[str, Any]) -> dict[str, Any]:
    files: list[str] = []
    for file_path in spec["files"]:
        files.append(file_path)
    for pattern in spec["patterns"]:
        files.extend(_glob_config_files(pattern))

    unique_files: list[str] = []
    seen_files: set[str] = set()
    for file_path in files:
        normalized = file_path.replace("\\", "/")
        if normalized in spec["exclude"] or normalized in seen_files:
            continue
        if normalized in HIDDEN_CONFIG_FILES:
            continue
        path = _safe_resolve(normalized)
        if path is None or not path.exists():
            continue
        seen_files.add(normalized)
        unique_files.append(normalized)

    order = [item for item in spec.get("order", []) if item in seen_files]
    if order:
        rank = {file_path: idx for idx, file_path in enumerate(order)}
        unique_files.sort(key=lambda item: (0, rank[item]) if item in rank else (1, 0))

    return {
        "id": spec["id"],
        "label": spec["label"],
        "open": spec["open"],
        "locked": spec["locked"] or _is_user_group_locked(spec["id"]),
        "group_locked": spec["locked"],
        "user_group_locked": _is_user_group_locked(spec["id"]),
        "system_locked": spec["id"] not in USER_LOCKABLE_GROUPS and spec["locked"],
        "lockable": spec["id"] in USER_LOCKABLE_GROUPS or _is_user_managed_group(spec),
        "user_managed": _is_user_managed_group(spec),
        "renamable": _is_renamable_config_group(spec),
        "deletable": _is_deletable_config_group(spec),
        "movable": _is_move_target_group(spec),
        "trainable": spec["trainable"],
        "methods_subdir": spec["methods_subdir"],
        "files": [
            get_config_file_meta(
                file_path,
                spec["id"],
                spec["label"],
                spec["locked"],
                spec["trainable"],
                spec["methods_subdir"],
            )
            for file_path in unique_files
        ],
    }


def _glob_config_files(pattern: str) -> list[str]:
    if not pattern.startswith("configs/") or ".." in Path(pattern).parts:
        return []
    return [
        _display_path(path)
        for path in sorted(ROOT.glob(pattern))
        if path.is_file()
        and path.suffix == ".toml"
        and _safe_resolve(_display_path(path))
        and _display_path(path) not in HIDDEN_CONFIG_FILES
    ]


def _default_config_file_group_specs() -> list[dict[str, Any]]:
    return [
        {"id": "gui_methods", "label": "可训练方法变体", "open": True, "locked": False, "trainable": True, "methods_subdir": "gui-methods", "patterns": ["configs/gui-methods/*.toml"]},
        {"id": "imported", "label": "导入配置", "open": True, "locked": False, "trainable": True, "methods_subdir": "imported", "patterns": ["configs/imported/*.toml"]},
        {"id": "datasets", "label": "数据集配置", "open": False, "locked": False, "trainable": False, "patterns": ["configs/datasets/*.toml"]},
        {"id": "presets", "label": "系统预设配置（锁定只读）", "open": False, "locked": True, "trainable": False, "files": ["configs/base.toml", "configs/presets.toml"]},
    ]


def _group_defaults(
    group_id: str,
    label: str,
    locked: bool,
    trainable: bool,
    methods_subdir: str,
    open_by_default: bool,
) -> dict[str, Any]:
    return {
        "id": group_id,
        "label": label,
        "locked": locked,
        "open": open_by_default,
        "trainable": trainable,
        "methods_subdir": methods_subdir,
    }


def _find_config_group_spec(specs: list[dict[str, Any]], group_id: str) -> dict[str, Any] | None:
    normalized = _normalize_group_id(group_id)
    for spec in specs:
        if spec.get("id") == normalized:
            return spec
    return None


def _new_user_config_group_spec(group_id: str, label: str) -> dict[str, Any]:
    return {
        "id": group_id,
        "label": label,
        "open": True,
        "locked": False,
        "trainable": True,
        "methods_subdir": "imported",
        "user_managed": True,
        "files": [],
        "order": [],
        "patterns": [],
        "exclude": set(),
    }


def _move_orphaned_config_files_to_fallback_groups(specs: list[dict[str, Any]], files: list[str]) -> None:
    for file_path in files:
        normalized = _normalize_config_rel_path(file_path)
        path = _safe_resolve(normalized)
        if path is None or path.suffix.lower() != ".toml" or not path.exists():
            continue
        if _config_file_is_covered_by_specs(specs, normalized):
            continue

        fallback = _fallback_config_group_spec(normalized)
        target = _find_config_group_spec(specs, fallback["id"])
        if target is None:
            target = fallback
            specs.append(target)

        target.setdefault("files", [])
        if normalized not in target["files"]:
            target["files"].append(normalized)
        target.setdefault("order", [])
        target["order"] = [item for item in target["order"] if item != normalized]
        target["order"].append(normalized)
        target["exclude"] = set(item for item in target.get("exclude", set()) if item != normalized)


def _config_file_is_covered_by_specs(specs: list[dict[str, Any]], rel_path: str) -> bool:
    normalized = _normalize_config_rel_path(rel_path)
    for spec in specs:
        if any(item.get("path") == normalized for item in _build_config_file_group(spec).get("files", [])):
            return True
    return False


def _fallback_config_group_spec(rel_path: str) -> dict[str, Any]:
    if rel_path.startswith("configs/datasets/"):
        group_id = "unfiled_datasets"
        label = "未分组数据集配置"
        trainable = False
        methods_subdir = ""
    else:
        group_id = "unfiled_imported"
        label = "未分组导入配置"
        trainable = True
        methods_subdir = "imported"
    return {
        "id": group_id,
        "label": label,
        "open": True,
        "locked": False,
        "trainable": trainable,
        "methods_subdir": methods_subdir,
        "user_managed": True,
        "files": [],
        "order": [],
        "patterns": [],
        "exclude": set(),
    }


def _is_user_managed_group(spec: dict[str, Any]) -> bool:
    return bool(spec.get("user_managed")) and str(spec.get("id") or "") not in SYSTEM_CONFIG_GROUP_IDS


def _is_fixed_config_group(spec: dict[str, Any]) -> bool:
    group_id = str(spec.get("id") or "")
    if group_id in FIXED_SYSTEM_CONFIG_GROUP_IDS:
        return True
    if bool(spec.get("locked")) and not _is_user_managed_group(spec) and not _is_user_group_locked(group_id):
        return True
    return False


def _is_deletable_config_group(spec: dict[str, Any]) -> bool:
    group_id = str(spec.get("id") or "")
    return (
        group_id not in FIXED_SYSTEM_CONFIG_GROUP_IDS
        and not bool(spec.get("locked"))
        and not _is_user_group_locked(group_id)
    )


def _is_renamable_config_group(spec: dict[str, Any]) -> bool:
    group_id = str(spec.get("id") or "")
    return group_id not in FIXED_SYSTEM_CONFIG_GROUP_IDS and not bool(spec.get("locked"))


def _is_move_target_group(spec: dict[str, Any]) -> bool:
    group_id = str(spec.get("id") or "")
    return _is_user_managed_group(spec) or group_id in FILE_MOVE_TARGET_GROUPS


def _lockable_group_ids() -> set[str]:
    ids = set(USER_LOCKABLE_GROUPS)
    ids.update(
        spec["id"]
        for spec in _load_config_file_group_specs()
        if _is_user_managed_group(spec)
    )
    return ids


def _unique_group_id(base: str, specs: list[dict[str, Any]]) -> str:
    used = {str(spec.get("id") or "") for spec in specs}
    root = base or "custom_group"
    candidate = root
    idx = 2
    while candidate in used:
        candidate = f"{root}_{idx}"
        idx += 1
    return candidate


def _slugify_group_label(label: str) -> str:
    chars: list[str] = []
    for ch in label.strip().lower():
        if ch.isascii() and ch.isalnum():
            chars.append(ch)
        elif ch in {"-", "_"}:
            chars.append(ch)
        elif ch.isspace():
            chars.append("_")
    slug = "".join(chars).strip("_-")
    return slug or "custom_group"


def _normalize_group_label(label: str) -> str:
    return " ".join(str(label or "").strip().split())[:48]


def _group_patterns_include_file(spec: dict[str, Any], rel_path: str) -> bool:
    path = _safe_resolve(rel_path)
    if path is None:
        return False
    normalized = _normalize_config_rel_path(rel_path)
    for pattern in spec.get("patterns") or []:
        if not str(pattern).startswith("configs/") or ".." in Path(str(pattern)).parts:
            continue
        if normalized in _glob_config_files(str(pattern)):
            return True
    return False


def _normalize_config_rel_path(rel_path: str) -> str:
    return str(rel_path or "").strip().replace("\\", "/").lstrip("/")


def _normalize_dataset_preset_path(rel_path: str, *, must_exist: bool) -> str:
    raw = str(rel_path or "").replace("\\", "/").strip()
    if not raw:
        raise ValueError("缺少数据集预设路径")
    path = Path(raw)
    if path.is_absolute():
        try:
            raw = path.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError("数据集预设必须在项目目录内") from exc
        path = Path(raw)
    if ".." in path.parts:
        raise ValueError("数据集预设路径不能包含 ..")
    if path.suffix.lower() != ".toml":
        path = path.with_suffix(".toml")
    if len(path.parts) == 1:
        path = Path("configs") / "datasets" / path
    normalized = path.as_posix().lstrip("/")
    if not normalized.startswith("configs/datasets/"):
        raise ValueError("数据集预设必须保存在 configs/datasets/ 下")
    safe_path = _safe_resolve(normalized)
    if safe_path is None:
        raise ValueError("数据集预设路径不合法")
    if must_exist and not safe_path.exists():
        raise ValueError("数据集预设不存在")
    return normalized


def _normalize_group_id(group_id: str) -> str:
    return str(group_id or "").strip()


def _is_system_preset_path(rel_path: str) -> bool:
    normalized = _normalize_config_rel_path(rel_path)
    return normalized in SYSTEM_PRESET_FILES or normalized.startswith(SYSTEM_PRESET_PREFIXES)


def _is_system_locked_path(rel_path: str) -> bool:
    normalized = _normalize_config_rel_path(rel_path)
    return _is_system_preset_path(normalized) or normalized in SYSTEM_MANAGED_FILES


def _is_dataset_preset_readonly(rel_path: str) -> bool:
    normalized = _normalize_config_rel_path(rel_path)
    return normalized in SYSTEM_DATASET_PRESET_FILES or get_config_file_meta(normalized).get("locked", False)


def _is_user_locked(rel_path: str) -> bool:
    file_locks, _ = _load_user_locks()
    return _normalize_config_rel_path(rel_path) in file_locks


def _is_user_group_locked(group_id: str | None) -> bool:
    _, group_locks = _load_user_locks()
    return _normalize_group_id(group_id or "") in group_locks


def _load_user_locks() -> tuple[set[str], set[str]]:
    if not WEB_USER_LOCKS_FILE.exists():
        return set(), set()
    try:
        data = toml.loads(WEB_USER_LOCKS_FILE.read_text(encoding="utf-8"))
    except toml.TomlDecodeError:
        return set(), set()

    file_locks: set[str] = set()
    for raw in _string_list(data.get("locked")):
        normalized = _normalize_config_rel_path(raw)
        path = _safe_resolve(normalized)
        if path is None or path.suffix != ".toml":
            continue
        if _is_system_locked_path(normalized):
            continue
        file_locks.add(normalized)

    group_locks: set[str] = set()
    for raw in _string_list(data.get("locked_groups")):
        normalized = _normalize_group_id(raw)
        if normalized in _lockable_group_ids():
            group_locks.add(normalized)
    return file_locks, group_locks


def _save_user_locks(file_locks: set[str], group_locks: set[str]) -> None:
    WEB_USER_LOCKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    WEB_USER_LOCKS_FILE.write_text(
        toml.dumps({
            "locked": sorted(file_locks),
            "locked_groups": sorted(group_locks),
        }),
        encoding="utf-8",
    )


def _lock_reason_label(reason: str) -> str:
    labels = {
        "system": "系统只读",
        "user": "用户锁定",
        "user_group": "分组锁定",
        "group": "分组只读",
    }
    return labels.get(reason, "")


def _lock_reason_message(meta: dict[str, Any]) -> str:
    reason = str(meta.get("lock_reason") or "")
    if reason == "system":
        return "该配置文件是系统预设，已内置锁定"
    if reason == "user":
        return "该配置文件已被用户锁定"
    if reason == "user_group":
        return "该配置文件所在分组已被用户锁定"
    if reason == "group":
        return "该配置文件属于只读分组"
    return "该配置文件已锁定"


def _list_system_preset_files() -> list[str]:
    files: list[str] = []
    for rel_path in sorted(SYSTEM_PRESET_FILES):
        path = _safe_resolve(rel_path)
        if path is not None and path.exists():
            files.append(rel_path)
    for prefix in SYSTEM_PRESET_PREFIXES:
        folder = _safe_resolve(prefix.rstrip("/"))
        if folder is None or not folder.is_dir():
            continue
        files.extend(
            _display_path(path)
            for path in sorted(folder.glob("*.toml"))
            if path.is_file() and _display_path(path) not in HIDDEN_CONFIG_FILES
        )
    return sorted(dict.fromkeys(files))


def _read_git_head_file(rel_path: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f"HEAD:{rel_path}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def _backup_relative_path(rel_path: str) -> Path:
    path = Path(rel_path)
    try:
        return path.relative_to("configs")
    except ValueError:
        return path


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item]


def _patch_toml_top_level(content: str, values: dict[str, Any]) -> str:
    doc = tomlkit.parse(content or "")
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            continue
        if "." in key or key in {"general", "datasets"}:
            raise ValueError(f"不支持写入嵌套字段: {key}")
        doc[key] = _normalize_patch_value(key, value)
    return tomlkit.dumps(doc)


def _normalize_patch_value(key: str, value: Any) -> Any:
    if key in {"sample_every_n_epochs", "sample_every_n_steps"}:
        if value in ("", None):
            return None
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} 必须是整数") from exc
    if key == "sample_at_first":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    return value


def get_field_help() -> dict[str, dict[str, str]]:
    try:
        from gui.explanations import FIELD_HELP
        return FIELD_HELP
    except ImportError:
        return {}


def get_groups() -> dict[str, list[str]]:
    try:
        from gui import _GROUPS, _BASIC
        return {"groups": {k: sorted(v) for k, v in _GROUPS.items()}, "basic": sorted(_BASIC)}
    except ImportError:
        return {"groups": {}, "basic": []}


def _load(p: Path) -> dict:
    if not p.exists():
        return {}
    return expand_env_vars_in_obj(toml.loads(p.read_text(encoding="utf-8")))


def _safe_resolve(rel_path: str) -> Path | None:
    resolved = (ROOT / _normalize_config_rel_path(rel_path)).resolve()
    configs_root = CONFIGS_DIR.resolve()
    try:
        resolved.relative_to(configs_root)
    except ValueError:
        return None
    return resolved


def _normalize_prompt_file_path(value: str) -> str:
    clean = str(value or "").replace("\\", "/").strip()
    if not clean:
        clean = DEFAULT_SAMPLE_PROMPTS_FILE
    path = Path(clean)
    if path.is_absolute():
        try:
            clean = path.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError("提示词文件必须在项目目录内") from exc
        path = Path(clean)
    if ".." in path.parts:
        raise ValueError("提示词文件路径不能包含 ..")
    if path.suffix.lower() != ".txt":
        raise ValueError("提示词文件必须是 .txt")
    if not path.as_posix().startswith("configs/"):
        raise ValueError("提示词文件必须保存在 configs/ 下")
    return path.as_posix().lstrip("/")


def _sample_prompts_path_for_config(train_config_file: str) -> str:
    normalized_config = _normalize_config_rel_path(train_config_file)
    config_path = _safe_resolve(normalized_config)
    if config_path is None or Path(normalized_config).suffix.lower() != ".toml":
        raise ValueError("训练配置文件路径不合法")
    try:
        rel_to_configs = Path(normalized_config).relative_to("configs")
    except ValueError as exc:
        raise ValueError("训练配置文件必须保存在 configs/ 下") from exc
    prompt_path = Path("configs") / "sample-prompts" / rel_to_configs.with_suffix(".txt")
    return _normalize_prompt_file_path(prompt_path.as_posix())


def _safe_config_subdir(subdir: str) -> Path | None:
    clean = str(subdir or "").replace("\\", "/").strip("/")
    if not clean or ".." in Path(clean).parts:
        return None
    resolved = (CONFIGS_DIR / clean).resolve()
    try:
        resolved.relative_to(CONFIGS_DIR.resolve())
    except ValueError:
        return None
    return resolved


def _resolve_project_path(value: str) -> Path:
    path = Path(expand_env_vars(value))
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _auto_data_dir_for_key(value: Any, source_path: Path, suffix: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        return _derived_data_dir(source_path, suffix)
    path = _resolve_project_path(raw)
    if _is_builtin_default_data_dir(raw) or not path.exists():
        return _derived_data_dir(source_path, suffix)
    return path


def _derived_data_dir(source_path: Path, suffix: str) -> Path:
    parent = source_path.parent if source_path.name else source_path
    name = source_path.name or "dataset"
    return (parent / f"{name}_{suffix}").resolve()


def _is_builtin_default_data_dir(value: str) -> bool:
    clean = str(value or "").replace("\\", "/").strip().strip("/")
    return clean in {DEFAULT_RESIZED_IMAGE_DIR, DEFAULT_LORA_CACHE_DIR}


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _check_training_images(cfg: dict[str, Any], add) -> None:
    rows = _dataset_rows_for_estimate(cfg)
    if not rows:
        rows = [{
            "source_dir": str(cfg.get("source_image_dir") or ""),
            "image_dir": str(cfg.get("resized_image_dir") or cfg.get("source_image_dir") or ""),
        }]
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    all_missing_captions: list[str] = []
    checked_groups = 0
    for idx, row in enumerate(rows, start=1):
        image_dir = _resolve_project_path(str(row.get("image_dir") or row.get("source_dir") or ""))
        source_dir = _resolve_project_path(str(row.get("source_dir") or ""))
        settings = row.get("settings") if isinstance(row.get("settings"), dict) else {}
        caption_extension = str(settings.get("caption_extension") or cfg.get("caption_extension") or ".txt")
        if not caption_extension.startswith("."):
            caption_extension = f".{caption_extension}"
        prefer_json_caption = _bool_value(
            settings.get("prefer_json_caption", cfg.get("prefer_json_caption")),
            False,
        )
        if not image_dir.is_dir():
            continue
        checked_groups += 1
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in image_exts)
        key = "training_images" if idx == 1 else f"dataset_{idx}_training_images"
        label = "缩放图像目录" if idx == 1 else f"第 {idx} 组缩放图像目录"
        if not images:
            add("error", key, f"{label}里没有可训练图片，请先预处理生成训练图", image_dir)
            continue
        for image in images[:50]:
            source_caption = source_dir / f"{image.stem}{caption_extension}"
            resized_caption = image.with_suffix(caption_extension)
            json_exists = (
                prefer_json_caption
                and (
                    (source_dir / f"{image.stem}.json").exists()
                    or image.with_suffix(".json").exists()
                )
            )
            if not json_exists and not source_caption.exists() and not resized_caption.exists():
                all_missing_captions.append(image.name)
    if checked_groups == 0:
        return
    if all_missing_captions:
        sample = ", ".join(all_missing_captions[:3])
        add("warning", "captions", f"部分图片未找到同名标注，例如 {sample}")
    else:
        add("ok", "captions", "抽样图片均找到同名标注")


def _check_dataset_source_paths(cfg: dict[str, Any], add) -> None:
    rows = _dataset_rows_for_estimate(cfg)
    if not rows:
        return
    for idx, row in enumerate(rows, start=1):
        source = _resolve_project_path(str(row.get("source_dir") or ""))
        key = "source_image_dir" if idx == 1 else f"dataset_{idx}_source_dir"
        label = "源图像目录" if idx == 1 else f"第 {idx} 组原始数据集目录"
        if not str(row.get("source_dir") or "").strip():
            add("error", key, f"{label} 未填写")
        elif not source.exists():
            add("error", key, f"{label} 不存在", source)
        elif not source.is_dir():
            add("error", key, f"{label} 不是目录", source)
        elif not any(source.iterdir()):
            add("warning", key, f"{label} 为空", source)
        else:
            add("ok", key, f"{label} 存在", source)


def _check_dataset_paths(cfg: dict[str, Any], add, *, check_runtime_dirs: bool = True) -> None:
    rows = _dataset_rows_for_estimate(cfg)
    if not check_runtime_dirs:
        return
    for idx, row in enumerate(rows, start=1):
        image_dir = _resolve_project_path(str(row.get("image_dir") or ""))
        cache_dir = _resolve_project_path(str(row.get("cache_dir") or ""))
        prefix = f"dataset_{idx}"
        if not image_dir.exists():
            add("error", f"{prefix}_image_dir", f"第 {idx} 组缩放图路径不存在", image_dir)
        elif not image_dir.is_dir():
            add("error", f"{prefix}_image_dir", f"第 {idx} 组缩放图路径不是目录", image_dir)
        if not cache_dir.exists():
            add("error", f"{prefix}_cache_dir", f"第 {idx} 组缓存路径不存在", cache_dir)
        elif not cache_dir.is_dir():
            add("error", f"{prefix}_cache_dir", f"第 {idx} 组缓存路径不是目录", cache_dir)


def _check_cache_sidecars(cfg: dict[str, Any], add) -> None:
    cache_dirs: list[tuple[int, Path]] = []
    for idx, row in enumerate(_dataset_rows_for_estimate(cfg), start=1):
        raw = str(row.get("cache_dir") or "").strip()
        if not raw:
            continue
        cache_dirs.append((idx, _resolve_project_path(raw)))
    if not cache_dirs:
        raw = str(cfg.get("lora_cache_dir") or "").strip()
        if raw:
            cache_dirs = [(1, _resolve_project_path(raw))]

    cache_dirs = [(idx, path) for idx, path in cache_dirs if path.is_dir()]
    if not cache_dirs:
        return

    if cfg.get("use_vae_cache", cfg.get("cache_latents_to_disk", False)):
        _check_cache_sidecar_pattern(add, cache_dirs, "*.npz", "latent_cache", "VAE latent 缓存", "未找到 .npz latent 缓存，可能需要先预处理")
    if cfg.get("use_text_cache", cfg.get("cache_text_encoder_outputs_to_disk", False)):
        _check_cache_sidecar_pattern(add, cache_dirs, "*_anima_te.safetensors", "text_cache", "文本编码器缓存", "未找到文本编码器缓存，可能需要先预处理")
    if cfg.get("ip_features_cache_to_disk", False) or cfg.get("use_repa", False) or cfg.get("use_ip_adapter", False):
        _check_cache_sidecar_pattern(add, cache_dirs, "*_anima_pe.safetensors", "pe_cache", "PE 图像特征缓存", "未找到 PE 图像特征缓存，IP-Adapter/REPA 可能需要先 preprocess-pe")


def _check_cache_sidecar_pattern(
    add,
    cache_dirs: list[tuple[int, Path]],
    pattern: str,
    key: str,
    label: str,
    missing_message: str,
) -> None:
    for idx, cache_dir in cache_dirs:
        count = len(list(cache_dir.glob(pattern)))
        item_key = key if idx == 1 else f"dataset_{idx}_{key}"
        if count:
            add("ok", item_key, f"第 {idx} 组找到 {count} 个{label}", cache_dir)
        else:
            add("warning", item_key, f"第 {idx} 组{missing_message}", cache_dir)
