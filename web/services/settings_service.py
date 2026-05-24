"""Global Web UI settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import toml

ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = ROOT / "configs"
SETTINGS_FILE = CONFIGS_DIR / "web-ui-settings.toml"

DEFAULT_OUTPUT_ROOT = "output/runs"
GLOBAL_MODEL_PATH_KEYS = (
    "pretrained_model_name_or_path",
    "qwen3",
    "vae",
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
    current = _load_settings()
    output_root = _normalize_output_root(
        str(data.get("output_root", current["output_root"]) or DEFAULT_OUTPUT_ROOT),
        allow_empty=False,
    )
    raw = _load_raw_settings()
    section = raw.get("global") if isinstance(raw.get("global"), dict) else {}
    defaults = _default_global_settings()
    next_global = {**section, "output_root": output_root}
    for key in GLOBAL_MODEL_PATH_KEYS:
        if key in data:
            value = _normalize_global_model_path(data.get(key))
            next_global[key] = value or current.get(key) or defaults.get(key, "")
        elif key not in next_global:
            next_global[key] = current.get(key, "") or defaults.get(key, "")
    raw["global"] = {
        **next_global,
    }
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(toml.dumps(raw), encoding="utf-8")
    saved = _load_settings()
    return {
        "ok": True,
        "message": "全局设置已保存",
        **saved,
        "defaults": _default_global_settings(),
    }


def resolve_output_root(value: str | None = None) -> Path:
    output_root = value if value is not None else _load_settings()["output_root"]
    return _resolve_output_root(output_root)


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _load_settings() -> dict[str, str]:
    defaults = _default_global_settings()
    raw = _load_raw_settings()
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
    return settings


def _load_raw_settings() -> dict[str, Any]:
    if not SETTINGS_FILE.exists():
        return {}
    try:
        raw = toml.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except toml.TomlDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def _default_global_settings() -> dict[str, str]:
    return {
        "output_root": DEFAULT_OUTPUT_ROOT,
        **_load_base_model_path_defaults(),
    }


def _load_base_model_path_defaults() -> dict[str, str]:
    defaults = {key: "" for key in GLOBAL_MODEL_PATH_KEYS}
    base_file = SETTINGS_FILE.parent / "base.toml"
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
    if path.is_absolute():
        return path.resolve().as_posix()
    if ".." in path.parts:
        raise ValueError("输出文件夹不能包含 ..")
    return path.as_posix().lstrip("/").rstrip("/") or DEFAULT_OUTPUT_ROOT


def _resolve_output_root(value: str) -> Path:
    normalized = _normalize_output_root(value, allow_empty=False)
    path = Path(normalized)
    if path.is_absolute():
        return path.resolve()
    return (ROOT / normalized).resolve()
