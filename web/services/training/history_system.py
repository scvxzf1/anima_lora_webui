"""Bounded read-only paging of GPU/system history; detail keeps its tail limit."""
from __future__ import annotations

from web.services.training.history_meta import _history_task_dir
from web.services.training.log_index import fingerprint, index_for, records


def get_history_system_page(self, task_id: str, *, offset: int = 0, limit: int = 2000) -> dict:
    start, size = int(offset), int(limit)
    if start < 0 or not 1 <= size <= 2000:
        raise ValueError("system offset 必须非负，limit 必须在 1 到 2000 之间")
    directory = _history_task_dir(task_id)
    if not directory.is_dir():
        raise FileNotFoundError("任务不存在")
    path = directory / "system.jsonl"
    total, selected = 0, []
    if path.is_file():
        key = fingerprint(path)
        total = index_for(key)[1]
        selected = list(records(key, start, start + size))
    end = min(total, start + size)
    return {"ok": True, "system": [value for _, value in selected],
            "indices": [index for index, _ in selected], "offset": min(start, total),
            "limit": size, "returned": len(selected), "total": total,
            "next_offset": end, "has_more_after": end < total}
