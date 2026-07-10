"""Shared constants and path/settings helpers for WebUI preview services."""

from __future__ import annotations

from datetime import datetime

import re
from pathlib import Path
from urllib.parse import quote
from typing import Any

import toml
from PIL import Image

from web.services import path_safety, settings_service
from web.services.preview.context import call, get

ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = settings_service.CONFIGS_DIR
SETTINGS_FILE = settings_service.SETTINGS_FILE

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
WEIGHT_EXTS = {".safetensors"}
DEFAULT_TRAINING_DIR = "output/ckpt/sample"
DEFAULT_INFERENCE_DIR = "output/tests"
DEFAULT_OUTPUT_ROOT = settings_service.DEFAULT_OUTPUT_ROOT
MAX_IMAGE_LIMIT = 500
MAX_WEIGHT_LIMIT = 500
SAMPLE_NAME_RE = re.compile(
    r"^(?P<prefix>.+)_(?P<tag>e\d{6}|\d{6})_(?P<prompt_index>\d+)_(?P<timestamp>\d{14})(?:_(?P<seed>-?\d+))?$"
)


def _root() -> Path:
    """Prefer facade.ROOT so tests can monkeypatch preview_service.ROOT."""
    value = get("ROOT")
    return value if isinstance(value, Path) else Path(value)


def _settings_file() -> Path:
    """Prefer facade.SETTINGS_FILE so tests can monkeypatch the facade path."""
    value = get("SETTINGS_FILE")
    return value if isinstance(value, Path) else Path(value)


def _preview_task_label(task: dict[str, Any]) -> str:
    return str(
        task.get("name")
        or f"{task.get('methods_subdir') or '-'} / {task.get('variant') or task.get('id') or '-'}"
    )


def _load_settings() -> dict[str, str]:
    defaults = {
        "training_dir": get("DEFAULT_TRAINING_DIR"),
        "inference_dir": get("DEFAULT_INFERENCE_DIR"),
        "custom_dir": "",
    }
    raw = call("_load_raw_settings")
    preview = raw.get("preview", {}) if isinstance(raw, dict) else {}
    if not isinstance(preview, dict):
        return defaults
    out = dict(defaults)
    for key in out:
        try:
            if key == "training_dir":
                out[key] = call(
                    "_normalize_project_dir",
                    str(preview.get(key, out[key]) or ""),
                    allow_empty=(key == "custom_dir"),
                )
            else:
                out[key] = call(
                    "_normalize_preview_dir",
                    str(preview.get(key, out[key]) or ""),
                    allow_empty=(key == "custom_dir"),
                )
        except ValueError:
            out[key] = defaults[key]
    return out


def _load_raw_settings() -> dict[str, Any]:
    settings_file = _settings_file()
    if not settings_file.exists():
        return {}
    try:
        raw = toml.loads(settings_file.read_text(encoding="utf-8"))
    except toml.TomlDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def _read_safetensors_metadata(path: Path) -> dict[str, str]:
    return path_safety.read_safetensors_metadata(path)


def _normalize_optional_preview_dir(value: str | None) -> str:
    if not value:
        return ""
    try:
        return call("_normalize_preview_dir", value, allow_empty=True)
    except ValueError:
        return ""


def _normalize_preview_dir(value: str, *, allow_empty: bool) -> str:
    clean = str(value or "").replace("\\", "/").strip()
    if not clean:
        if allow_empty:
            return ""
        raise ValueError("路径不能为空")
    path = Path(clean)
    if ".." in path.parts:
        raise ValueError("路径不能包含 ..")
    if path.is_absolute():
        return path.resolve().as_posix()
    return call("_normalize_project_dir", clean, allow_empty=allow_empty)


def _normalize_project_dir(value: str, *, allow_empty: bool) -> str:
    clean = str(value or "").replace("\\", "/").strip()
    if not clean:
        if allow_empty:
            return ""
        raise ValueError("路径不能为空")
    normalized = call("_normalize_project_file", clean)
    return normalized.rstrip("/")


def _normalize_project_file(value: str) -> str:
    clean = str(value or "").replace("\\", "/").strip()
    if not clean:
        raise ValueError("路径不能为空")
    path = Path(clean)
    if ".." in path.parts:
        raise ValueError("路径不能包含 ..")
    root = _root()
    if path.is_absolute():
        resolved = path.resolve()
        try:
            return resolved.relative_to(root.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError("路径必须在项目目录内") from exc
    return path.as_posix().lstrip("/")


def _resolve_project_path(value: str) -> Path | None:
    try:
        rel = call("_normalize_project_dir", value, allow_empty=False)
    except ValueError:
        return None
    root = _root()
    resolved = (root / rel).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return None
    return resolved


def _resolve_preview_dir(value: str, *, current_task_sample_dir: str | None = None) -> Path | None:
    path = Path(str(value or "").replace("\\", "/").strip())
    if path.is_absolute():
        resolved = path.resolve()
        allowed = call("_resolve_allowed_sample_dir", current_task_sample_dir)
        if allowed is not None:
            try:
                resolved.relative_to(allowed)
            except ValueError:
                return None
        return resolved
    return call("_resolve_project_path", value)


def _resolve_preview_file(value: str, *, allowed_sample_dir: str | None = None) -> Path:
    clean = str(value or "").replace("\\", "/").strip()
    if not clean:
        raise ValueError("路径不能为空")
    path = Path(clean)
    root = _root()
    if path.is_absolute():
        resolved = path.resolve()
        for allowed in call("_allowed_external_preview_dirs", allowed_sample_dir):
            try:
                resolved.relative_to(allowed)
                return resolved
            except ValueError:
                continue
        raise ValueError("项目外图片只允许读取当前任务样张目录或已保存的预览目录")

    normalized = call("_normalize_project_file", clean)
    resolved = (root / normalized).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("图片路径必须在项目目录内") from exc
    return resolved


def _resolve_weight_file(value: str, *, task: dict[str, Any] | None = None) -> Path:
    clean = str(value or "").replace("\\", "/").strip()
    if not clean:
        raise ValueError("路径不能为空")
    path = Path(clean)
    root = _root()
    if path.is_absolute():
        resolved = path.resolve()
    else:
        normalized = call("_normalize_project_file", clean)
        resolved = (root / normalized).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError("权重路径必须在项目目录内") from exc

    for allowed in call("_allowed_weight_dirs", task):
        try:
            resolved.relative_to(allowed)
            return resolved
        except ValueError:
            continue
    raise ValueError("权重文件只允许从训练输出目录或全局输出目录下载")


def _resolve_allowed_sample_dir(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(str(value).replace("\\", "/").strip())
    if not path.is_absolute():
        path = _root() / path
    return path.resolve()


def _allowed_external_preview_dirs(allowed_sample_dir: str | None) -> list[Path]:
    dirs: list[Path] = []
    sample_dir = call("_resolve_allowed_sample_dir", allowed_sample_dir)
    if sample_dir is not None:
        dirs.append(sample_dir)
    dirs.append(call("_resolve_global_output_root"))
    settings = call("_load_settings")
    for key in ("inference_dir", "custom_dir"):
        resolved = call("_resolve_display_path", settings.get(key, ""))
        if resolved is not None:
            dirs.append(resolved)
    return dirs


def _allowed_weight_dirs(task: dict[str, Any] | None = None) -> list[Path]:
    # Prefer effective settings so allowlist matches analysis service.
    try:
        settings = call("get_preview_settings")
    except Exception:
        settings = call("_load_settings")
    return path_safety.allowed_weight_dirs(
        root=_root(),
        output_root=call("_resolve_global_output_root"),
        task=task,
        training_dirs=path_safety.training_dirs_from_preview_settings(settings),
    )


def _resolve_training_output_dir(value: str) -> Path | None:
    raw = str(value or "").replace("\\", "/").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = _root() / raw
    return path.resolve()


def _resolve_display_path(value: str) -> Path | None:
    return path_safety.resolve_display_path(value, root=_root())


def _int_or_none(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _display_path(path: Path) -> str:
    return path_safety.display_path(path, root=_root())


def _latest_runtime_sample_dir() -> dict[str, str] | None:
    output_root = call("_resolve_global_output_root")
    if not output_root.exists() or not output_root.is_dir():
        return None

    candidates: list[tuple[float, str, Path, Path]] = []
    try:
        children = list(output_root.iterdir())
    except OSError:
        return None
    for run_dir in children:
        if not run_dir.is_dir():
            continue
        sample_dir = run_dir / "training_output" / "sample"
        if not sample_dir.is_dir():
            continue
        candidates.append(
            (call("_runtime_sample_sort_ts", run_dir, sample_dir), run_dir.name, run_dir, sample_dir)
        )
    if not candidates:
        return None
    _, _, run_dir, sample_dir = max(candidates, key=lambda item: (item[0], item[1]))
    return {
        "run_dir": call("_display_path", run_dir),
        "sample_dir": call("_display_path", sample_dir),
    }


def _runtime_sample_sort_ts(run_dir: Path, sample_dir: Path) -> float:
    timestamps: list[float] = []
    for path in (run_dir, run_dir / "training_output", sample_dir):
        try:
            timestamps.append(path.stat().st_mtime)
        except OSError:
            continue
    image_exts = get("IMAGE_EXTS")
    try:
        latest_image = max(
            (
                path.stat().st_mtime
                for path in sample_dir.iterdir()
                if path.is_file() and path.suffix.lower() in image_exts
            ),
            default=None,
        )
        if latest_image is not None:
            timestamps.append(latest_image)
    except OSError:
        pass
    return max(timestamps, default=0.0)


def _resolve_global_output_root() -> Path:
    return settings_service.resolve_output_root()
