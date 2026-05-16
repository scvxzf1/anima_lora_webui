"""Configuration loading, merging, and saving."""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import toml
import tomlkit

from library.env import expand_env_vars, expand_env_vars_in_obj, load_dotenv

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
SYSTEM_PRESET_FILES = frozenset({
    "configs/base.toml",
    "configs/presets.toml",
})
SYSTEM_PRESET_PREFIXES = ("configs/methods/", "configs/gui-methods/")
SYSTEM_MANAGED_FILES = frozenset({
    "configs/web-file-groups.toml",
    "configs/web-user-locks.toml",
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
        "reft", "postfix", "ip_adapter", "easycontrol",
    ]


_FAMILY_VARIANTS: dict[str, list[str]] = {
    "lora": ["lora", "lora_longer", "lora-8gb", "lora_repa"],
    "lokr": ["lokr"],
    "ortholora": ["ortholora"],
    "tlora": ["tlora", "tlora_ortho"],
    "hydralora": ["hydralora_sigma", "hydralora_experimental", "hydralora_fei", "fera"],
    "reft": ["reft", "tlora_ortho_reft"],
    "postfix": ["postfix_ortho_cond"],
    "ip_adapter": ["ip_adapter"],
    "easycontrol": ["easycontrol"],
}


def list_variants(method: str) -> list[str]:
    if not GUI_METHODS_DIR.exists():
        return []
    have = {p.stem for p in GUI_METHODS_DIR.glob("*.toml")}
    want = _FAMILY_VARIANTS.get(method, [])
    return [v for v in want if v in have]


def list_all_variants() -> list[str]:
    if not GUI_METHODS_DIR.exists():
        return []
    return sorted(p.stem for p in GUI_METHODS_DIR.glob("*.toml"))


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


def preflight_training_config(variant: str, preset: str, methods_subdir: str = "gui-methods") -> dict[str, Any]:
    cfg = apply_auto_data_dirs(load_merged_config(variant, preset, methods_subdir))
    checks: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

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
    if cfg.get("dataset_config"):
        check_file("dataset_config", "数据集配置", (".toml",))

    check_dir("source_image_dir", "源图像目录", must_exist=True, warn_empty=True)
    check_dir("resized_image_dir", "缩放图像目录", must_exist=False, warn_empty=True)
    check_dir("lora_cache_dir", "LoRA 缓存目录", must_exist=False, warn_empty=True)
    check_dir("output_dir", "输出目录", must_exist=False)

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


def estimate_training_steps(variant: str, preset: str, methods_subdir: str = "gui-methods") -> dict[str, Any]:
    cfg = apply_auto_data_dirs(load_merged_config(variant, preset, methods_subdir))
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    source_dir = _resolve_project_path(str(cfg.get("source_image_dir") or ""))
    resized_dir = _resolve_project_path(str(cfg.get("resized_image_dir") or ""))
    source_images = _count_images(source_dir, image_exts)
    resized_images = _count_images(resized_dir, image_exts)
    dataset_repeats = _dataset_num_repeats(cfg)

    train_images = resized_images or source_images
    sample_ratio = _positive_float(cfg.get("sample_ratio"), 1.0)
    epochs = _positive_int(cfg.get("max_train_epochs"), 1)
    batch_size = _positive_int(cfg.get("train_batch_size"), 1)
    grad_accum = _positive_int(cfg.get("gradient_accumulation_steps"), 1)
    effective_batch = max(1, batch_size * grad_accum)
    repeated_images = int(train_images * dataset_repeats * sample_ratio)
    steps_per_epoch = (repeated_images + effective_batch - 1) // effective_batch if repeated_images else 0
    total_steps = steps_per_epoch * epochs

    return {
        "ok": True,
        "variant": variant,
        "preset": preset,
        "methods_subdir": methods_subdir,
        "source_image_count": source_images,
        "resized_image_count": resized_images,
        "train_image_count": train_images,
        "dataset_num_repeats": dataset_repeats,
        "sample_ratio": sample_ratio,
        "max_train_epochs": epochs,
        "train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size": effective_batch,
        "repeated_image_count": repeated_images,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "uses_preprocessed_images": resized_images > 0,
        "source_dir": _display_path(source_dir),
        "resized_dir": _display_path(resized_dir),
        "lora_cache_dir": _display_path(_resolve_project_path(str(cfg.get("lora_cache_dir") or ""))),
    }


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


def _positive_float(value: Any, fallback: float) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return fallback
    return n if n > 0 else fallback


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
    lines = path.read_text(encoding="utf-8").splitlines()
    prompts = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    return {
        "ok": True,
        "file": normalized,
        "content": "\n".join(prompts),
        "prompts": prompts,
    }


def save_sample_prompts_file(content: str, rel_path: str | None = None) -> dict[str, Any]:
    normalized = _normalize_prompt_file_path(rel_path or DEFAULT_SAMPLE_PROMPTS_FILE)
    lines = [line.strip() for line in str(content or "").splitlines()]
    prompts = [line for line in lines if line and not line.startswith("#")]
    path = (ROOT / normalized).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(prompts) + ("\n" if prompts else ""), encoding="utf-8")
    return {
        "ok": True,
        "file": normalized,
        "content": "\n".join(prompts),
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
    normalized = _normalize_config_rel_path(rel_path)
    path = _safe_resolve(normalized)
    if path is None:
        return False, "路径不合法", "", []
    meta = get_config_file_meta(normalized)
    if meta.get("locked"):
        return False, f"{_lock_reason_message(meta)}，请使用新名称保存新配置后编辑", "", []
    if not isinstance(values, dict):
        return False, "字段补丁格式不合法", "", []

    source = content if content is not None else load_raw_file(rel_path)
    try:
        next_content = _patch_toml_top_level(source, values)
        toml.loads(next_content)
    except Exception as e:
        return False, f"TOML 更新失败: {e}", "", []

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(next_content, encoding="utf-8")
    return True, "保存成功", next_content, sorted(values.keys())


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
    if normalized not in USER_LOCKABLE_GROUPS:
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
    return [_build_config_file_group(spec) for spec in _load_config_file_group_specs()]


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
        "label": Path(normalized).name,
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
        return _group_defaults("gui_methods", "GUI 训练变体", False, True, "gui-methods", True)
    if rel_path.startswith("configs/methods/"):
        return _group_defaults("methods", "内置方法配置（锁定只读）", True, True, "methods", False)
    if rel_path.startswith("configs/imported/"):
        return _group_defaults("imported", "导入配置", False, True, "imported", True)
    if rel_path.startswith("configs/datasets/"):
        return _group_defaults("datasets", "数据集配置", False, False, "", False)
    if rel_path in {"configs/base.toml", "configs/presets.toml"}:
        return _group_defaults("presets", "预设配置（锁定只读）", True, False, "", False)
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
            "files": _string_list(raw.get("files")),
            "patterns": _string_list(raw.get("patterns")),
            "exclude": set(_string_list(raw.get("exclude"))),
        })
    return specs


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
        path = _safe_resolve(normalized)
        if path is None or not path.exists():
            continue
        seen_files.add(normalized)
        unique_files.append(normalized)

    return {
        "id": spec["id"],
        "label": spec["label"],
        "open": spec["open"],
        "locked": spec["locked"] or _is_user_group_locked(spec["id"]),
        "group_locked": spec["locked"],
        "user_group_locked": _is_user_group_locked(spec["id"]),
        "system_locked": spec["id"] not in USER_LOCKABLE_GROUPS and spec["locked"],
        "lockable": spec["id"] in USER_LOCKABLE_GROUPS,
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
        if path.is_file() and path.suffix == ".toml" and _safe_resolve(_display_path(path))
    ]


def _default_config_file_group_specs() -> list[dict[str, Any]]:
    return [
        {"id": "presets", "label": "预设配置（锁定只读）", "open": False, "locked": True, "trainable": False, "files": ["configs/base.toml", "configs/presets.toml"]},
        {"id": "gui_methods", "label": "GUI 训练变体", "open": True, "locked": False, "trainable": True, "methods_subdir": "gui-methods", "patterns": ["configs/gui-methods/*.toml"]},
        {"id": "imported", "label": "导入配置", "open": True, "locked": False, "trainable": True, "methods_subdir": "imported", "patterns": ["configs/imported/*.toml"]},
        {"id": "datasets", "label": "数据集配置", "open": False, "locked": False, "trainable": False, "patterns": ["configs/datasets/*.toml"]},
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


def _normalize_config_rel_path(rel_path: str) -> str:
    return str(rel_path or "").strip().replace("\\", "/").lstrip("/")


def _normalize_group_id(group_id: str) -> str:
    return str(group_id or "").strip()


def _is_system_preset_path(rel_path: str) -> bool:
    normalized = _normalize_config_rel_path(rel_path)
    return normalized in SYSTEM_PRESET_FILES or normalized.startswith(SYSTEM_PRESET_PREFIXES)


def _is_system_locked_path(rel_path: str) -> bool:
    normalized = _normalize_config_rel_path(rel_path)
    return _is_system_preset_path(normalized) or normalized in SYSTEM_MANAGED_FILES


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
        if normalized in USER_LOCKABLE_GROUPS:
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
            if path.is_file()
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
    image_dir = _resolve_project_path(str(cfg.get("resized_image_dir") or cfg.get("source_image_dir") or ""))
    source_dir = _resolve_project_path(str(cfg.get("source_image_dir") or ""))
    if not image_dir.is_dir():
        return
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in image_exts)
    if not images:
        add("error", "training_images", "缩放图像目录里没有可训练图片，请先预处理生成训练图", image_dir)
        return
    missing_captions = []
    for image in images[:50]:
        source_caption = source_dir / f"{image.stem}.txt"
        resized_caption = image.with_suffix(".txt")
        if not source_caption.exists() and not resized_caption.exists():
            missing_captions.append(image.name)
    if missing_captions:
        sample = ", ".join(missing_captions[:3])
        add("warning", "captions", f"部分图片未找到同名 .txt 标注，例如 {sample}", image_dir)
    else:
        add("ok", "captions", "抽样图片均找到同名 .txt 标注", image_dir)


def _check_cache_sidecars(cfg: dict[str, Any], add) -> None:
    cache_dir = _resolve_project_path(str(cfg.get("lora_cache_dir") or ""))
    if not cache_dir.is_dir():
        return
    latent_count = len(list(cache_dir.glob("*.npz")))
    te_count = len(list(cache_dir.glob("*_anima_te.safetensors")))
    pe_count = len(list(cache_dir.glob("*_anima_pe.safetensors")))
    if cfg.get("cache_latents_to_disk", False):
        if latent_count:
            add("ok", "latent_cache", f"找到 {latent_count} 个 VAE latent 缓存", cache_dir)
        else:
            add("warning", "latent_cache", "未找到 .npz latent 缓存，可能需要先预处理", cache_dir)
    if cfg.get("cache_text_encoder_outputs_to_disk", False):
        if te_count:
            add("ok", "text_cache", f"找到 {te_count} 个文本编码器缓存", cache_dir)
        else:
            add("warning", "text_cache", "未找到文本编码器缓存，可能需要先预处理", cache_dir)
    if cfg.get("ip_features_cache_to_disk", False) or cfg.get("use_repa", False) or cfg.get("use_ip_adapter", False):
        if pe_count:
            add("ok", "pe_cache", f"找到 {pe_count} 个 PE 图像特征缓存", cache_dir)
        else:
            add("warning", "pe_cache", "未找到 PE 图像特征缓存，IP-Adapter/REPA 可能需要先 preprocess-pe", cache_dir)
