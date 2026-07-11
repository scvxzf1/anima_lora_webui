"""Listing, path resolution and allowed weight directories."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping

from web.services import path_safety, preview_service, settings_service
from web.services.weight_analysis.constants import MAX_ANALYSIS_WEIGHT_LIMIT, WEIGHT_EXTS
from web.services.weight_analysis.context import get


def _root() -> Path:
    """Prefer facade.ROOT so tests can monkeypatch weight_analysis_service.ROOT."""
    value = get("ROOT")
    return value if isinstance(value, Path) else Path(value)


@dataclass(frozen=True)
class WeightListingContext:
    task: dict[str, Any] | None = None


def list_analysis_weights(
    *,
    task: dict[str, Any] | None = None,
    allow_latest_fallback: bool = True,
    training_service: Any | None = None,
    include_archived: bool = False,
    limit: int = MAX_ANALYSIS_WEIGHT_LIMIT,
) -> dict[str, Any]:
    """Return training weight candidates usable by the analysis page."""

    limit = max(1, min(int(limit or MAX_ANALYSIS_WEIGHT_LIMIT), MAX_ANALYSIS_WEIGHT_LIMIT))
    if task:
        payload = preview_service.list_training_weights(
            task,
            allow_latest_fallback=allow_latest_fallback,
        )
        weights = [
            _analysis_weight_meta(item)
            for item in payload.get("weights", [])[:limit]
            if isinstance(item, dict)
        ]
        return {
            "ok": True,
            "directory": payload.get("directory", ""),
            "directory_exists": bool(payload.get("directory_exists")),
            "count": len(weights),
            "total": payload.get("total", len(weights)),
            "task_count": payload.get("task_count", 0),
            "weights": weights,
            "message": payload.get("message", ""),
            "analysis_note": "只读取 .safetensors 静态权重，不加载模型、不跑图、不占用 GPU。",
        }

    weights_by_path: dict[str, dict[str, Any]] = {}
    directories: list[str] = []
    errors: list[str] = []

    def add_listing(payload: Mapping[str, Any], *, source_task: dict[str, Any] | None = None) -> None:
        directory = str(payload.get("directory") or "")
        if directory and directory not in directories:
            directories.append(directory)
        source_meta = _source_task_meta(source_task) if source_task else {}
        for item in payload.get("weights") or []:
            if not isinstance(item, dict):
                continue
            meta = _analysis_weight_meta(item)
            if source_meta:
                meta["source_task"] = source_meta
                meta["scope_label"] = meta.get("scope_label") or source_meta.get("label", "")
            key = str(meta.get("abs_path") or meta.get("file") or meta.get("name") or "")
            if not key:
                continue
            prev = weights_by_path.get(key)
            if prev is None or float(meta.get("mtime") or 0) > float(prev.get("mtime") or 0):
                weights_by_path[key] = meta

    try:
        add_listing(preview_service.list_training_weights(None, allow_latest_fallback=allow_latest_fallback))
    except Exception as exc:  # 仅影响下拉候选，不影响手填路径分析。
        errors.append(str(exc))

    for source_task in _analysis_source_tasks(training_service, include_archived=include_archived):
        try:
            add_listing(
                preview_service.list_training_weights(source_task, allow_latest_fallback=False),
                source_task=source_task,
            )
        except Exception as exc:
            label = _source_task_meta(source_task).get("label") or source_task.get("id") or "历史任务"
            errors.append(f"{label}: {exc}")

    weights = sorted(
        weights_by_path.values(),
        key=lambda item: (float(item.get("mtime") or 0), str(item.get("name") or "")),
        reverse=True,
    )[:limit]
    return {
        "ok": True,
        "directory": " · ".join(directories[:2]) + (" · ..." if len(directories) > 2 else ""),
        "directories": directories,
        "directory_exists": bool(directories),
        "count": len(weights),
        "total": len(weights_by_path),
        "task_count": sum(1 for item in weights if item.get("source_task")),
        "weights": weights,
        "message": "" if weights else "未找到可分析权重文件",
        "errors": errors[:8],
        "analysis_note": "只读取 .safetensors 静态权重，不加载模型、不跑图、不占用 GPU。",
    }


def resolve_analysis_weight(value: str, *, task: dict[str, Any] | None = None) -> Path:
    """Resolve analysis weight path via shared path_safety allowlist policy."""
    allowed = _allowed_weight_dirs(task=task)
    try:
        resolved = path_safety.resolve_allowed_file(
            value,
            root=_root(),
            allowed_dirs=allowed,
            require_suffix=".safetensors",
        )
    except ValueError as exc:
        msg = str(exc)
        if "路径为空" in msg:
            raise ValueError("请填写权重路径") from exc
        if ".." in msg:
            raise ValueError("权重路径不能包含 ..") from exc
        if "只支持" in msg:
            raise ValueError("只支持 .safetensors 权重文件") from exc
        raise ValueError("权重文件只允许从训练输出目录或全局输出目录读取") from exc
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError("权重文件不存在")
    if not os.access(resolved, os.R_OK):
        raise ValueError("权重文件不可读取")
    return resolved


def _normalize_user_path_value(value: str) -> str:
    return path_safety.normalize_user_path_value(value)


def _analysis_source_tasks(training_service: Any | None, *, include_archived: bool) -> list[dict[str, Any]]:
    if training_service is None:
        return []
    tasks: list[dict[str, Any]] = []
    current_output_dir = str(getattr(training_service, "current_output_dir", "") or "").strip()
    if current_output_dir:
        tasks.append(
            {
                "id": str(getattr(training_service, "current_task_id", "") or "current"),
                "job": "training",
                "name": "当前训练",
                "output_dir": current_output_dir,
                "variant": str(getattr(training_service, "current_variant", "") or ""),
                "state": str(getattr(training_service, "status", "") or ""),
            }
        )
    try:
        history = training_service.list_history_tasks(include_archived=include_archived, limit=160)
    except Exception:
        history = []
    for item in history or []:
        if not isinstance(item, dict) or item.get("job") != "training":
            continue
        if not str(item.get("output_dir") or "").strip():
            continue
        tasks.append(item)
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in tasks:
        key = str(item.get("id") or item.get("output_dir") or "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _source_task_meta(task: dict[str, Any] | None) -> dict[str, Any]:
    task = task or {}
    return {
        "id": str(task.get("id") or ""),
        "label": str(task.get("name") or task.get("history_run_label") or task.get("variant") or task.get("id") or "训练任务"),
        "state": str(task.get("state") or ""),
        "started_at": task.get("started_at"),
        "started_at_text": str(task.get("started_at_text") or ""),
        "output_dir": str(task.get("output_dir") or ""),
    }


def _analysis_weight_meta(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "file": item.get("file", ""),
        "abs_path": item.get("abs_path", ""),
        "name": item.get("name", ""),
        "kind": item.get("kind", ""),
        "scope": item.get("scope", ""),
        "scope_label": item.get("scope_label", ""),
        "epoch": item.get("epoch"),
        "steps": item.get("steps"),
        "mtime": item.get("mtime"),
        "mtime_text": item.get("mtime_text", ""),
        "size_bytes": item.get("size_bytes", 0),
        "output_name": item.get("output_name", ""),
    }


def _allowed_weight_dirs(task: dict[str, Any] | None = None) -> list[Path]:
    try:
        settings = preview_service.get_preview_settings()
    except Exception:
        settings = {}
    return path_safety.allowed_weight_dirs(
        root=_root(),
        output_root=settings_service.resolve_output_root(),
        task=task,
        training_dirs=path_safety.training_dirs_from_preview_settings(settings),
    )


def _is_under_allowed_weight_dir(path: Path, *, task: dict[str, Any] | None = None) -> bool:
    return path_safety.is_under_allowed_dirs(path, _allowed_weight_dirs(task))


def _resolve_display_path(value: str) -> Path:
    resolved = path_safety.resolve_display_path(value, root=_root())
    if resolved is None:
        raise ValueError("路径不能为空")
    return resolved


def _display_path(path: Path) -> str:
    return path_safety.display_path(path, root=_root())

