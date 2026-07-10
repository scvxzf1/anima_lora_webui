"""Batch history task mutations (archive / group binding)."""

from __future__ import annotations

from typing import Any

from web.services.training.history_meta import _update_history_task
from web.services.training.history_store import _history_meta_records


def batch_archive_history_tasks(task_ids: list[str], *, archived: bool) -> dict[str, Any]:
    tasks = []
    for task_id in task_ids:
        tasks.append(_update_history_task(task_id, {"archived": archived})["task"])
    return {
        "ok": True,
        "message": "已归档所选历史任务" if archived else "已取消归档所选历史任务",
        "updated": len(tasks),
        "tasks": tasks,
    }


def batch_set_history_group(task_ids: list[str], group: Any) -> dict[str, Any]:
    expanded_task_ids = bound_history_task_ids(task_ids)
    tasks = []
    for task_id in expanded_task_ids:
        tasks.append(_update_history_task(task_id, {"group": group}, bind_group=False)["task"])
    return {
        "ok": True,
        "message": "已更新同配置文件自动分组内的历史任务集合",
        "updated": len(tasks),
        "requested": len(task_ids),
        "tasks": tasks,
    }


def bound_history_task_ids(task_ids: list[str]) -> list[str]:
    requested = [str(task_id or "").strip() for task_id in task_ids if str(task_id or "").strip()]
    if not requested:
        return []
    requested_set = set(requested)
    records = _history_meta_records()
    selected_keys = {
        str(record["group_key"])
        for record in records
        if record["id"] in requested_set and str(record["group_key"])
    }
    if not selected_keys:
        return requested
    expanded = [
        str(record["id"])
        for record in records
        if record["id"] in requested_set or str(record["group_key"]) in selected_keys
    ]
    ordered: list[str] = []
    for task_id in requested + expanded:
        if task_id not in ordered:
            ordered.append(task_id)
    return ordered
