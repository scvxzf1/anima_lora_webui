"""Training weight listing helpers for WebUI preview service."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote
import re

from web.services.preview.context import call, get


def list_training_weights(
    task: dict[str, Any] | None = None,
    *,
    allow_latest_fallback: bool = True,
) -> dict[str, Any]:
    task = task or {}
    output_dir = str(task.get("output_dir") or "")
    if not output_dir:
        if not allow_latest_fallback:
            return _empty_weights_listing("", "这个历史训练任务没有记录输出目录")
        latest = call("_latest_runtime_sample_dir")
        if not latest:
            return _empty_weights_listing("", "训练任务没有记录输出目录")
        output_dir = str(Path(latest["sample_dir"]).parent)
        task = {**task, "output_dir": output_dir}

    resolved = call("_resolve_training_output_dir", output_dir)
    if resolved is None:
        raise ValueError("训练输出目录不合法")
    display_dir = call("_display_path", resolved)
    if not resolved.exists():
        return _empty_weights_listing(display_dir, "输出目录不存在")
    if not resolved.is_dir():
        return _empty_weights_listing(display_dir, "输出路径不是目录")

    output_name = str(task.get("variant") or "")
    candidates = [
        p
        for p in resolved.iterdir()
        if p.is_file()
        and p.suffix.lower() in get("WEIGHT_EXTS")
        and not p.name.endswith("_moe.safetensors")
    ]
    if output_name:
        named = [p for p in candidates if p.name.startswith(output_name)]
        if named:
            candidates = named

    items = [_weight_meta(path, task=task) for path in candidates[:get("MAX_WEIGHT_LIMIT")]]
    items.sort(key=_weight_sort_key)
    task_count = sum(1 for item in items if item.get("scope") == "task")
    return {
        "ok": True,
        "directory": display_dir,
        "directory_exists": True,
        "count": len(items),
        "total": len(candidates),
        "task_count": task_count,
        "weights": items,
        "message": "" if items else "未找到权重文件",
    }


def list_config_group_training_weights(
    tasks: list[dict[str, Any]],
    *,
    methods_subdir: str,
    variant: str,
    preset: str,
) -> dict[str, Any]:
    group_label = f"{methods_subdir} / {variant} / {preset or 'default'}"
    weights_by_path: dict[str, dict[str, Any]] = {}
    directories: list[str] = []

    for task in tasks:
        listing = list_training_weights(task, allow_latest_fallback=False)
        directory = str(listing.get("directory") or "")
        if directory and directory not in directories:
            directories.append(directory)
        task_source = {
            "id": str(task.get("id") or ""),
            "label": call("_preview_task_label", task),
            "state": task.get("state", ""),
            "started_at": task.get("started_at"),
            "started_at_text": task.get("started_at_text", ""),
            "finished_at": task.get("finished_at"),
            "finished_at_text": task.get("finished_at_text", ""),
            "output_dir": str(task.get("output_dir") or ""),
        }
        for item in listing.get("weights") or []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("file") or "")
            if not key:
                continue
            merged = dict(item)
            merged["source_task"] = task_source
            merged["scope_label"] = _group_weight_scope_label(merged, task_source)
            previous = weights_by_path.get(key)
            if previous is None or _group_weight_match_score(merged) > _group_weight_match_score(previous):
                weights_by_path[key] = merged

    weights = list(weights_by_path.values())
    weights.sort(key=_weight_sort_key)
    task_weight_count = sum(1 for item in weights if item.get("scope") == "task")
    return {
        "ok": True,
        "mode": "config_group",
        "label": f"训练分组合并权重 · {group_label} · {len(tasks)} 次训练",
        "directory": " · ".join(directories[:2]) + (" · ..." if len(directories) > 2 else ""),
        "directories": directories,
        "directory_exists": bool(directories),
        "count": len(weights),
        "total": len(weights),
        "task_count": task_weight_count,
        "weights": weights,
        "message": "" if weights else "这个训练分组还没有可显示的权重文件",
        "group": {
            "methods_subdir": methods_subdir,
            "variant": variant,
            "preset": preset or "default",
        },
        "group_task_count": len(tasks),
    }


def resolve_training_weight(rel_path: str, task: dict[str, Any] | None = None) -> Path:
    resolved = call("_resolve_weight_file", rel_path, task=task)
    if resolved.suffix.lower() not in get("WEIGHT_EXTS") or resolved.name.endswith("_moe.safetensors"):
        raise ValueError("只允许下载训练权重文件")
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError("权重文件不存在")
    return resolved


def _group_weight_match_score(weight: dict[str, Any]) -> tuple[int, float]:
    source_task = weight.get("source_task") or {}
    scope = str(weight.get("scope") or "")
    score = 0
    mtime = call("_float_or_none", weight.get("mtime"))
    started_at = call("_float_or_none", source_task.get("started_at"))
    finished_at = call("_float_or_none", source_task.get("finished_at"))
    if mtime is not None and started_at is not None:
        if mtime < started_at - 180:
            return (score, started_at)
        if finished_at is not None:
            score += 6 if mtime <= finished_at + 180 else 2
        else:
            score += 3
    if scope == "task":
        score += 2
    if source_task.get("id"):
        score += 1
    return (score, started_at or 0)


def _group_weight_scope_label(weight: dict[str, Any], source_task: dict[str, Any]) -> str:
    base = str(weight.get("scope_label") or "")
    if source_task.get("id"):
        return f"{base} · {source_task.get('label') or source_task.get('id')}"
    return base


def _weight_meta(path: Path, *, task: dict[str, Any] | None = None) -> dict[str, Any]:
    stat = path.stat()
    metadata = call("_read_safetensors_metadata", path)
    epoch = call("_int_or_none", metadata.get("ss_epoch"))
    steps = call("_int_or_none", metadata.get("ss_steps"))
    num_epochs = call("_int_or_none", metadata.get("ss_num_epochs"))
    max_steps = call("_int_or_none", metadata.get("ss_max_train_steps"))
    output_name = str(metadata.get("ss_output_name") or "")
    kind = _weight_kind(path.name, output_name)
    scope = _weight_scope(stat.st_mtime, metadata, task)
    rel_path = call("_display_path", path)
    download_url = f"/api/preview/weight?file={quote(rel_path)}"
    task_id = str((task or {}).get("id") or "")
    if task_id:
        download_url += f"&task_id={quote(task_id)}"
    return {
        "file": rel_path,
        "abs_path": str(path.resolve()),
        "name": path.name,
        "download_url": download_url,
        "kind": kind,
        "scope": scope,
        "scope_label": "本任务" if scope == "task" else "同目录其他运行",
        "epoch": epoch,
        "steps": steps,
        "num_epochs": num_epochs,
        "max_steps": max_steps,
        "mtime": stat.st_mtime,
        "mtime_text": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "size_bytes": stat.st_size,
        "output_name": output_name,
    }


def _weight_kind(name: str, output_name: str) -> str:
    if output_name and name == f"{output_name}.safetensors":
        return "final"
    if output_name and name == f"{output_name}-checkpoint.safetensors":
        return "resume"
    if re.search(r"-step\d+\.safetensors$", name):
        return "step"
    if re.search(r"-\d{6}\.safetensors$", name):
        return "epoch"
    return "weight"


def _weight_sort_key(item: dict[str, Any]) -> tuple[int, int, float, str]:
    scope_rank = {"task": 0, "other": 1}
    kind_rank = {"epoch": 0, "step": 1, "resume": 2, "final": 3, "weight": 4}
    primary = item.get("steps") if item.get("steps") is not None else -1
    return (
        int(scope_rank.get(str(item.get("scope")), 9)),
        int(kind_rank.get(str(item.get("kind")), 9)),
        int(primary),
        float(item.get("mtime") or 0),
        str(item.get("name") or ""),
    )


def _weight_scope(mtime: float, metadata: dict[str, str], task: dict[str, Any] | None) -> str:
    if not task:
        return "other"
    started = call("_float_or_none", task.get("started_at"))
    finished = call("_float_or_none", task.get("finished_at"))
    if started is None:
        return "other"
    lower = started - 180
    upper = (finished + 180) if finished is not None else (datetime.now().timestamp() + 180)
    meta_started = call("_float_or_none", metadata.get("ss_training_started_at"))
    if meta_started is not None and lower <= meta_started <= upper:
        return "task"
    if lower <= float(mtime) <= upper:
        return "task"
    return "other"


def _empty_weights_listing(directory: str, message: str) -> dict[str, Any]:
    return {
        "ok": True,
        "directory": directory,
        "directory_exists": False,
        "count": 0,
        "total": 0,
        "weights": [],
        "message": message,
    }


