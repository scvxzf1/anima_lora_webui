"""Output run configuration listing and copy helpers.

This module is loaded by ``web.services.config_service`` as part of the
compatibility facade.  It keeps facade access lazy so the module can also be
imported directly without pulling in the legacy facade.
"""

from __future__ import annotations

import tomllib
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any

import tomlkit

from library.env import get_configs_root
from web.services.config import paths as _config_paths
from web.services.config.metadata import OUTPUT_RUN_CONFIG_FILES
from web.services.settings_service import (
    display_path as _display_settings_path,
)
from web.services.settings_service import resolve_output_root

def _missing_facade_dependency(*args, **kwargs):
    raise RuntimeError("output run config helper was called before facade sync")


save_raw_file = _missing_facade_dependency
get_config_file_meta = _missing_facade_dependency
list_config_file_groups = _missing_facade_dependency
move_config_file_to_group = _missing_facade_dependency

ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = get_configs_root()
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
)

_LEGACY_RAW_FILE_SHIM_NAMES = {
    "save_raw_file",
    "load_raw_file",
    "delete_raw_file",
    "patch_raw_file_values",
    "preview_raw_file_patch",
}
_LEGACY_SYNC_NAMES = tuple(
    _name for _name in _SYNC_NAMES
    if _name not in _LEGACY_RAW_FILE_SHIM_NAMES
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
        if _legacy_module is not None and _name in _LEGACY_SYNC_NAMES:
            setattr(_legacy_module, _name, _value)


def _exported(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        _sync_from_facade()
        return fn(*args, **kwargs)

    return wrapper


def _safe_resolve(rel_path: str) -> Path | None:
    return _config_paths.safe_resolve(rel_path, root=ROOT, configs_dir=CONFIGS_DIR)


def _normalize_group_id(group_id: str) -> str:
    return str(group_id or "").strip()


__all__ = ['list_output_runs', 'load_output_run_config', 'save_output_run_config_as', '_resolve_output_run_dir', '_normalize_output_run_name']

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




for _name in __all__:
    globals()[_name] = _exported(globals()[_name])
