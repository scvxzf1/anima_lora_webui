"""Global Web UI settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import toml

from library.env import get_configs_root, get_training_history_root, get_training_queue_root
from web.services._dynamic_path import DynamicPath

ROOT = Path(__file__).resolve().parents[2]


def _default_settings_file() -> Path:
    return get_configs_root() / "web-ui-settings.toml"


SETTINGS_FILE = DynamicPath(_default_settings_file)
CONFIGS_DIR = DynamicPath(lambda: SETTINGS_FILE.parent)

DEFAULT_OUTPUT_ROOT = "output/runs"
DEFAULT_UI_SCALE = 100
DEFAULT_UI_SCALE_OVERRIDE = ""
GLOBAL_MODEL_PATH_KEYS = (
    "pretrained_model_name_or_path",
    "qwen3",
    "vae",
)
GLOBAL_CONFIG_PATH_KEYS = (
    "configs_root",
    "history_root",
    "queue_root",
)
GLOBAL_UI_OVERRIDE_KEYS = (
    "ui_scale_config",
    "ui_scale_datasets",
    "ui_scale_training",
    "ui_scale_weight_analysis",
    "ui_scale_image_test",
    "ui_scale_settings",
    "ui_scale_environment",
    "ui_scale_history_overview",
    "ui_scale_history_analysis",
    "ui_scale_history_preview",
    "ui_scale_history_logs",
    "ui_scale_history_config_files",
)
GLOBAL_UI_KEYS = (
    "ui_scale",
    *GLOBAL_UI_OVERRIDE_KEYS,
)


def get_global_settings() -> dict[str, Any]:
    settings = _load_settings()
    defaults = _default_global_settings()
    return {
        "ok": True,
        **settings,
        "defaults": defaults,
    }


def save_global_settings(data: dict[str, Any]) -> dict[str, Any]:
    current_settings_file = Path(SETTINGS_FILE)
    current = _load_settings(current_settings_file)
    output_root = _normalize_output_root(
        str(data.get("output_root", current["output_root"]) or DEFAULT_OUTPUT_ROOT),
        allow_empty=False,
    )
    current_raw = _load_raw_settings(current_settings_file)
    current_section = current_raw.get("global") if isinstance(current_raw.get("global"), dict) else {}
    target_settings_file = _settings_file_for_payload(data, current_settings_file)
    target_raw = _load_raw_settings(target_settings_file)
    target_section = target_raw.get("global") if isinstance(target_raw.get("global"), dict) else {}
    defaults = _default_global_settings(settings_file=target_settings_file)
    next_global = {
        **target_section,
        **current_section,
        "output_root": output_root,
    }
    for key in GLOBAL_MODEL_PATH_KEYS:
        if key in data:
            value = _normalize_global_model_path(data.get(key))
            next_global[key] = value or current.get(key) or defaults.get(key, "")
        elif key not in next_global:
            next_global[key] = current.get(key, "") or defaults.get(key, "")
    for key in GLOBAL_CONFIG_PATH_KEYS:
        if key in data:
            value = _normalize_config_path(data.get(key))
            next_global[key] = value or defaults.get(key, "")
        elif key not in next_global:
            next_global[key] = current.get(key, "") or defaults.get(key, "")
    if "ui_scale" in data:
        value = _normalize_ui_setting("ui_scale", data.get("ui_scale"))
        next_global["ui_scale"] = value if value is not None else current.get("ui_scale", defaults.get("ui_scale"))
    elif "ui_scale" not in next_global:
        next_global["ui_scale"] = current.get("ui_scale", defaults.get("ui_scale"))
    for key in GLOBAL_UI_OVERRIDE_KEYS:
        if key not in data:
            continue
        value = _normalize_ui_setting(key, data.get(key))
        if value is None:
            next_global.pop(key, None)
        else:
            next_global[key] = value
    raw = {
        **current_raw,
        **target_raw,
    }
    raw["global"] = {
        **next_global,
    }
    target_settings_file.parent.mkdir(parents=True, exist_ok=True)
    target_settings_file.write_text(toml.dumps(raw), encoding="utf-8")

    # 如果设置了路径覆盖，同时保存到项目根目录的专用配置文件
    path_overrides = {
        key: data[key]
        for key in ("configs_root", "history_root", "queue_root")
        if key in data
    }
    if path_overrides:
        _save_path_overrides(path_overrides)

    saved = _load_settings(target_settings_file)
    return {
        "ok": True,
        "message": "全局设置已保存",
        "requires_reload": current_settings_file.resolve() != target_settings_file.resolve(),
        **saved,
        "defaults": _default_global_settings(settings_file=target_settings_file),
    }


def resolve_output_root(value: str | None = None) -> Path:
    output_root = value if value is not None else _load_settings()["output_root"]
    return _resolve_output_root(output_root)


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _load_settings(settings_file: Path | None = None) -> dict[str, Any]:
    settings_file = settings_file or Path(SETTINGS_FILE)
    defaults = _default_global_settings(settings_file=settings_file)
    raw = _load_raw_settings(settings_file)
    section = raw.get("global", {}) if isinstance(raw, dict) else {}
    if not isinstance(section, dict):
        return defaults
    settings = {**defaults}
    try:
        settings["output_root"] = _normalize_output_root(
            str(section.get("output_root", defaults["output_root"]) or ""),
            allow_empty=False,
        )
    except ValueError:
        settings["output_root"] = defaults["output_root"]
    for key in GLOBAL_MODEL_PATH_KEYS:
        if key in section:
            settings[key] = _normalize_global_model_path(section.get(key)) or defaults.get(key, "")
    for key in GLOBAL_CONFIG_PATH_KEYS:
        if key in section:
            settings[key] = _normalize_config_path(section.get(key)) or defaults.get(key, "")
    for key in GLOBAL_UI_KEYS:
        if key in section:
            value = _normalize_ui_setting(key, section.get(key))
            settings[key] = value if value is not None else defaults.get(key)

    # 显示当前实际使用的配置根目录（包括环境变量）
    actual_configs_root = settings_file.parent.resolve()
    try:
        settings["configs_root"] = actual_configs_root.relative_to(ROOT).as_posix()
    except ValueError:
        settings["configs_root"] = actual_configs_root.as_posix()
    try:
        history_root = get_training_history_root().resolve()
        settings["history_root"] = history_root.relative_to(ROOT).as_posix()
    except Exception:
        try:
            settings["history_root"] = get_training_history_root().as_posix()
        except Exception:
            settings["history_root"] = settings.get("history_root") or ""
    try:
        queue_root = get_training_queue_root().resolve()
        settings["queue_root"] = queue_root.relative_to(ROOT).as_posix()
    except Exception:
        try:
            settings["queue_root"] = get_training_queue_root().as_posix()
        except Exception:
            settings["queue_root"] = settings.get("queue_root") or ""

    return settings


def _load_raw_settings(settings_file: Path | None = None) -> dict[str, Any]:
    settings_file = settings_file or Path(SETTINGS_FILE)
    if not settings_file.exists():
        return {}
    try:
        raw = toml.loads(settings_file.read_text(encoding="utf-8"))
    except toml.TomlDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def _default_global_settings(*, settings_file: Path | None = None) -> dict[str, Any]:
    return {
        "output_root": DEFAULT_OUTPUT_ROOT,
        "configs_root": "configs",
        "history_root": "",
        "queue_root": "",
        "ui_scale": DEFAULT_UI_SCALE,
        **{key: DEFAULT_UI_SCALE_OVERRIDE for key in GLOBAL_UI_OVERRIDE_KEYS},
        **_load_base_model_path_defaults(settings_file=settings_file),
    }


def _load_base_model_path_defaults(*, settings_file: Path | None = None) -> dict[str, str]:
    defaults = {key: "" for key in GLOBAL_MODEL_PATH_KEYS}
    settings_file = settings_file or Path(SETTINGS_FILE)
    base_file = settings_file.parent / "base.toml"
    if not base_file.exists():
        return defaults
    try:
        raw = toml.loads(base_file.read_text(encoding="utf-8"))
    except toml.TomlDecodeError:
        return defaults
    if not isinstance(raw, dict):
        return defaults
    for key in GLOBAL_MODEL_PATH_KEYS:
        defaults[key] = _normalize_global_model_path(raw.get(key))
    return defaults


def _normalize_global_model_path(value: Any) -> str:
    # 模型路径保留用户写法：相对路径、绝对路径、环境变量字符串都原样进入配置模板。
    return str(value or "").strip()


def _normalize_output_root(value: str, *, allow_empty: bool) -> str:
    clean = str(value or "").replace("\\", "/").strip()
    if not clean:
        if allow_empty:
            return ""
        raise ValueError("输出文件夹不能为空")
    path = Path(clean)
    if ".." in path.parts:
        raise ValueError("输出文件夹不能包含 ..")
    if path.is_absolute():
        return path.resolve().as_posix()
    return path.as_posix().lstrip("/").rstrip("/") or DEFAULT_OUTPUT_ROOT


def _resolve_output_root(value: str) -> Path:
    normalized = _normalize_output_root(value, allow_empty=False)
    path = Path(normalized)
    if path.is_absolute():
        return path.resolve()
    return (ROOT / normalized).resolve()


def _normalize_ui_setting(key: str, value: Any) -> int | str | None:
    """Normalize UI settings."""
    if key == "ui_scale":
        return _normalize_required_ui_scale(value)
    if key in GLOBAL_UI_OVERRIDE_KEYS:
        return _normalize_optional_ui_scale(value)
    return None


def _normalize_required_ui_scale(value: Any) -> int:
    try:
        scale = int(value) if value is not None else DEFAULT_UI_SCALE
    except (ValueError, TypeError):
        return DEFAULT_UI_SCALE
    return _clamp_ui_scale(scale)


def _normalize_optional_ui_scale(value: Any) -> int | None:
    clean = str(value or "").strip()
    if not clean:
        return None
    try:
        scale = int(clean)
    except (ValueError, TypeError):
        return DEFAULT_UI_SCALE
    return _clamp_ui_scale(scale)


def _clamp_ui_scale(value: int) -> int:
    # 限制在 25% - 400% 之间
    if value < 25:
        return 25
    if value > 400:
        return 400
    return value


def _normalize_config_path(value: Any) -> str:
    """规范化配置路径（类似 _normalize_output_root）。"""
    clean = str(value or "").replace("\\", "/").strip()
    if not clean:
        return ""
    path = Path(clean)
    if path.is_absolute():
        return path.resolve().as_posix()
    if ".." in path.parts:
        raise ValueError("配置路径不能包含 ..")
    return path.as_posix().lstrip("/").rstrip("/")


def resolve_config_path(key: str, value: str | None = None) -> Path:
    """解析配置路径（类似 resolve_output_root）。"""
    if value is None:
        value = _load_settings().get(key, "")
    normalized = _normalize_config_path(value)
    if not normalized:
        # 返回默认值
        defaults = _default_global_settings()
        normalized = defaults.get(key, "")
    path = Path(normalized)
    if path.is_absolute():
        return path.resolve()
    return (ROOT / normalized).resolve()


def _settings_file_for_payload(data: dict[str, Any], current_settings_file: Path) -> Path:
    if "configs_root" not in data:
        return current_settings_file.resolve()
    return _settings_file_for_configs_root(data.get("configs_root"))


def _settings_file_for_configs_root(value: Any) -> Path:
    normalized = _normalize_config_path(value)
    if not normalized:
        normalized = "configs"
    path = Path(normalized)
    configs_dir = path.resolve() if path.is_absolute() else (ROOT / normalized).resolve()
    return configs_dir / "web-ui-settings.toml"


def _save_path_overrides(path_values: dict[str, Any]) -> None:
    """保存 configs/history/queue root 到项目根目录的专用配置文件。"""
    webui_paths_file = ROOT / ".anima-webui-settings.toml"

    raw: dict[str, Any] = {}
    if webui_paths_file.exists():
        try:
            raw = toml.loads(webui_paths_file.read_text(encoding="utf-8"))
        except toml.TomlDecodeError:
            raw = {}

    if "paths" not in raw or not isinstance(raw["paths"], dict):
        raw["paths"] = {}

    for key, value in path_values.items():
        if key not in {"configs_root", "history_root", "queue_root"}:
            continue
        normalized = _normalize_config_path(value)
        if normalized:
            raw["paths"][key] = normalized
        else:
            # 空值表示使用默认，从配置中删除
            raw["paths"].pop(key, None)

    webui_paths_file.write_text(toml.dumps(raw), encoding="utf-8")


def _save_configs_root_override(configs_root: str) -> None:
    """兼容旧调用：只写 configs_root。"""
    _save_path_overrides({"configs_root": configs_root})
