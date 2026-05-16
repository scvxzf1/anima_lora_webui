"""Preview image discovery and Web UI preview settings."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import toml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = ROOT / "configs"
SETTINGS_FILE = CONFIGS_DIR / "web-ui-settings.toml"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_TRAINING_DIR = "output/ckpt/sample"
DEFAULT_INFERENCE_DIR = "output/tests"
MAX_IMAGE_LIMIT = 500


def get_preview_settings(current_task_sample_dir: str | None = None) -> dict[str, Any]:
    settings = _load_settings()
    task_dir = _normalize_optional_preview_dir(current_task_sample_dir)
    training_dir = task_dir or settings["training_dir"]
    return {
        "ok": True,
        "training_dir": settings["training_dir"],
        "inference_dir": settings["inference_dir"],
        "custom_dir": settings["custom_dir"],
        "current_task_sample_dir": task_dir,
        "effective_training_dir": training_dir,
        "defaults": {
            "training_dir": DEFAULT_TRAINING_DIR,
            "inference_dir": DEFAULT_INFERENCE_DIR,
            "custom_dir": "",
        },
    }


def save_preview_settings(data: dict[str, Any]) -> dict[str, Any]:
    current = _load_settings()
    next_settings = {
        "training_dir": _normalize_project_dir(
            data.get("training_dir", current["training_dir"]) or DEFAULT_TRAINING_DIR,
            allow_empty=False,
        ),
        "inference_dir": _normalize_project_dir(
            data.get("inference_dir", current["inference_dir"]) or DEFAULT_INFERENCE_DIR,
            allow_empty=False,
        ),
        "custom_dir": _normalize_project_dir(data.get("custom_dir", current["custom_dir"]) or "", allow_empty=True),
    }
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(toml.dumps({"preview": next_settings}), encoding="utf-8")
    return {"ok": True, "message": "预览图路径设置已保存", **next_settings}


def list_preview_images(
    source: str,
    *,
    current_task_sample_dir: str | None = None,
    sample_config: dict[str, Any] | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    source = (source or "training").strip().lower()
    if source not in {"training", "inference", "custom"}:
        raise ValueError("source 只能是 training、inference 或 custom")

    settings = get_preview_settings(current_task_sample_dir)
    if source == "training":
        rel_dir = settings["effective_training_dir"]
        label = "当前任务样张"
    elif source == "inference":
        rel_dir = settings["inference_dir"]
        label = "推理预览"
    else:
        rel_dir = settings["custom_dir"]
        label = "自定义路径"

    if not rel_dir:
        return _empty_listing(source, label, "", exists=False, message="尚未设置自定义预览图路径")

    resolved = _resolve_preview_dir(rel_dir, current_task_sample_dir=current_task_sample_dir if source == "training" else None)
    if resolved is None:
        raise ValueError("预览图路径不合法")

    display_dir = _display_path(resolved)
    if not resolved.exists():
        listing = _empty_listing(
            source,
            label,
            display_dir,
            exists=False,
            message=_preview_empty_message(source, "目录不存在", sample_config),
        )
        listing["sample_config"] = sample_config or {}
        return listing
    if not resolved.is_dir():
        return _empty_listing(source, label, display_dir, exists=False, message="路径不是目录")

    limit = max(1, min(int(limit or 200), MAX_IMAGE_LIMIT))
    candidates = [
        p
        for p in resolved.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return {
        "ok": True,
        "source": source,
        "label": label,
        "directory": display_dir,
        "directory_exists": True,
        "count": len(candidates[:limit]),
        "total": len(candidates),
        "images": [_image_meta(path) for path in candidates[:limit]],
        "message": "" if candidates else _preview_empty_message(source, "暂无预览图", sample_config),
        "sample_config": sample_config or {},
    }


def resolve_preview_image(rel_path: str, allowed_sample_dir: str | None = None) -> Path:
    resolved = _resolve_preview_file(rel_path, allowed_sample_dir=allowed_sample_dir)
    if resolved.suffix.lower() not in IMAGE_EXTS:
        raise ValueError("只允许读取预览图片文件")
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError("图片不存在")
    return resolved


def _load_settings() -> dict[str, str]:
    defaults = {
        "training_dir": DEFAULT_TRAINING_DIR,
        "inference_dir": DEFAULT_INFERENCE_DIR,
        "custom_dir": "",
    }
    if not SETTINGS_FILE.exists():
        return defaults
    try:
        raw = toml.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except toml.TomlDecodeError:
        return defaults
    preview = raw.get("preview", {}) if isinstance(raw, dict) else {}
    if not isinstance(preview, dict):
        return defaults
    out = dict(defaults)
    for key in out:
        try:
            out[key] = _normalize_project_dir(str(preview.get(key, out[key]) or ""), allow_empty=(key == "custom_dir"))
        except ValueError:
            out[key] = defaults[key]
    return out


def _empty_listing(source: str, label: str, directory: str, *, exists: bool, message: str) -> dict[str, Any]:
    return {
        "ok": True,
        "source": source,
        "label": label,
        "directory": directory,
        "directory_exists": exists,
        "count": 0,
        "total": 0,
        "images": [],
        "message": message,
        "sample_config": {},
    }


def _image_meta(path: Path) -> dict[str, Any]:
    stat = path.stat()
    width = None
    height = None
    try:
        with Image.open(path) as img:
            width, height = img.size
    except Exception:
        pass
    rel_path = _display_path(path)
    return {
        "file": rel_path,
        "name": path.name,
        "url": f"/api/preview/image?file={quote(rel_path)}",
        "mtime": stat.st_mtime,
        "mtime_text": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "size_bytes": stat.st_size,
        "width": width,
        "height": height,
    }


def _normalize_optional_preview_dir(value: str | None) -> str:
    if not value:
        return ""
    try:
        return _normalize_preview_dir(value, allow_empty=True)
    except ValueError:
        return ""


def _normalize_preview_dir(value: str, *, allow_empty: bool) -> str:
    clean = str(value or "").replace("\\", "/").strip()
    if not clean:
        if allow_empty:
            return ""
        raise ValueError("路径不能为空")
    path = Path(clean)
    if path.is_absolute():
        return path.resolve().as_posix()
    return _normalize_project_dir(clean, allow_empty=allow_empty)


def _normalize_project_dir(value: str, *, allow_empty: bool) -> str:
    clean = str(value or "").replace("\\", "/").strip()
    if not clean:
        if allow_empty:
            return ""
        raise ValueError("路径不能为空")
    normalized = _normalize_project_file(clean)
    return normalized.rstrip("/")


def _normalize_project_file(value: str) -> str:
    clean = str(value or "").replace("\\", "/").strip()
    if not clean:
        raise ValueError("路径不能为空")
    path = Path(clean)
    if path.is_absolute():
        resolved = path.resolve()
        try:
            return resolved.relative_to(ROOT.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError("路径必须在项目目录内") from exc
    if ".." in path.parts:
        raise ValueError("路径不能包含 ..")
    return path.as_posix().lstrip("/")


def _resolve_project_path(value: str) -> Path | None:
    try:
        rel = _normalize_project_dir(value, allow_empty=False)
    except ValueError:
        return None
    resolved = (ROOT / rel).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        return None
    return resolved


def _resolve_preview_dir(value: str, *, current_task_sample_dir: str | None = None) -> Path | None:
    path = Path(str(value or "").replace("\\", "/").strip())
    if path.is_absolute():
        resolved = path.resolve()
        allowed = _resolve_allowed_sample_dir(current_task_sample_dir)
        if allowed is None:
            return None
        try:
            resolved.relative_to(allowed)
        except ValueError:
            return None
        return resolved
    return _resolve_project_path(value)


def _resolve_preview_file(value: str, *, allowed_sample_dir: str | None = None) -> Path:
    clean = str(value or "").replace("\\", "/").strip()
    if not clean:
        raise ValueError("路径不能为空")
    path = Path(clean)
    if path.is_absolute():
        resolved = path.resolve()
        allowed = _resolve_allowed_sample_dir(allowed_sample_dir)
        if allowed is None:
            raise ValueError("项目外图片只允许读取当前任务样张目录")
        try:
            resolved.relative_to(allowed)
        except ValueError as exc:
            raise ValueError("项目外图片只允许读取当前任务样张目录") from exc
        return resolved

    normalized = _normalize_project_file(clean)
    resolved = (ROOT / normalized).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError("图片路径必须在项目目录内") from exc
    return resolved


def _resolve_allowed_sample_dir(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(str(value).replace("\\", "/").strip())
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _preview_empty_message(source: str, fallback: str, sample_config: dict[str, Any] | None) -> str:
    if source != "training":
        return fallback
    cfg = sample_config or {}
    message = str(cfg.get("message") or "")
    if message and message != "训练中采样已配置":
        return f"{fallback}。{message}。"
    if cfg.get("enabled"):
        return f"{fallback}。如果训练刚开始，可能还没到达采样频率。"
    return f"{fallback}。未启用训练中采样时不会自动生成样张。"


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)
