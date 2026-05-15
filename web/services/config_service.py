"""Configuration loading, merging, and saving."""

from __future__ import annotations

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

load_dotenv()


def list_methods() -> list[str]:
    return [
        "lora", "ortholora", "tlora", "hydralora",
        "reft", "postfix", "ip_adapter", "easycontrol",
    ]


_FAMILY_VARIANTS: dict[str, list[str]] = {
    "lora": ["lora", "lora_longer", "lora-8gb", "lora_repa"],
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
    return expand_env_vars_in_obj(merged)


def preflight_training_config(variant: str, preset: str, methods_subdir: str = "gui-methods") -> dict[str, Any]:
    cfg = load_merged_config(variant, preset, methods_subdir)
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
    check_dir("resized_image_dir", "缩放图像目录", must_exist=True, warn_empty=True)
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


def load_raw_file(rel_path: str) -> str:
    path = _safe_resolve(rel_path)
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def save_raw_file(rel_path: str, content: str, *, allow_locked: bool = False) -> tuple[bool, str]:
    path = _safe_resolve(rel_path)
    if path is None:
        return False, "路径不合法"
    meta = get_config_file_meta(rel_path)
    if meta.get("locked") and not allow_locked:
        return False, "该配置文件已锁定，只能另存为副本后编辑"
    try:
        toml.loads(content)
    except toml.TomlDecodeError as e:
        return False, f"TOML 语法错误: {e}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True, "保存成功"


def patch_raw_file_values(
    rel_path: str,
    values: dict[str, Any],
    *,
    content: str | None = None,
) -> tuple[bool, str, str, list[str]]:
    path = _safe_resolve(rel_path)
    if path is None:
        return False, "路径不合法", "", []
    meta = get_config_file_meta(rel_path)
    if meta.get("locked"):
        return False, "该配置文件已锁定，只能另存为副本后编辑", "", []
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


def list_config_files() -> list[str]:
    return [item["path"] for group in list_config_file_groups() for item in group["files"]]


def list_config_file_groups() -> list[dict[str, Any]]:
    return [_build_config_file_group(spec) for spec in _load_config_file_group_specs()]


def get_config_file_meta(
    rel_path: str,
    group_id: str | None = None,
    group_label: str | None = None,
    locked: bool | None = None,
    trainable: bool | None = None,
    methods_subdir: str | None = None,
) -> dict[str, Any]:
    normalized = rel_path.replace("\\", "/")
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
    return {
        "path": normalized,
        "label": Path(normalized).name,
        "group": group_id or inferred["id"],
        "group_label": group_label or inferred["label"],
        "locked": inferred["locked"] if locked is None else locked,
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
        "locked": spec["locked"],
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
        doc[key] = value
    return tomlkit.dumps(doc)


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
    resolved = (ROOT / rel_path).resolve()
    configs_root = CONFIGS_DIR.resolve()
    try:
        resolved.relative_to(configs_root)
    except ValueError:
        return None
    return resolved


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
        add("warning", "training_images", "缩放图像目录里没有可训练图片", image_dir)
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
