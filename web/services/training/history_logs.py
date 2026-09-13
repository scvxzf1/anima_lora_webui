"""Paged access to complete history log files."""

from __future__ import annotations

from typing import Any
from web.services.training.log_index import fingerprint, index_for, records, search

from web.services.training.constants import (
    DEFAULT_HISTORY_LOG_PAGE_RECORDS,
    MAX_HISTORY_LOG_PAGE_RECORDS,
)
from web.services.training.history_meta import _history_log_path, _history_task_dir


def get_history_log_page(
    self,
    task_id: str,
    *,
    offset: int | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    safe_limit = max(1, min(int(limit or DEFAULT_HISTORY_LOG_PAGE_RECORDS), MAX_HISTORY_LOG_PAGE_RECORDS))
    path = _resolve_log_path(task_id)
    if path is None:
        return _empty_log_page(safe_limit)
    requested_offset = max(0, int(offset)) if offset is not None else None
    key = fingerprint(path)
    total = index_for(key)[1]
    start = max(0, total - safe_limit) if requested_offset is None else requested_offset
    selected = list(records(key, start, start + safe_limit))
    # Preserve the legacy tail contract: return the last N valid records.
    while requested_offset is None and len(selected) < safe_limit and start > 0:
        previous = max(0, start - safe_limit)
        selected = (list(records(key, previous, start)) + selected)[-safe_limit:]
        start = previous
    resolved_offset = selected[0][0] if selected else min(requested_offset or total, total)
    next_offset = min(total, (requested_offset if requested_offset is not None else resolved_offset) + safe_limit)
    return {
        "ok": True,
        "logs": [value for _index, value in selected],
        "indices": [index for index, _value in selected],
        "offset": resolved_offset,
        "limit": safe_limit,
        "returned": len(selected),
        "total": total,
        "next_offset": next_offset,
        "has_more_before": resolved_offset > 0,
        "has_more_after": next_offset < total,
    }


def find_history_log_match(
    self,
    task_id: str,
    *,
    query: str,
    cursor: int = 0,
    direction: str = "forward",
) -> dict[str, Any]:
    safe_query = str(query or "").strip().casefold()
    if not safe_query:
        raise ValueError("日志搜索关键词不能为空")
    if len(safe_query) > 512:
        raise ValueError("日志搜索关键词过长")
    safe_direction = str(direction or "forward").strip().lower()
    if safe_direction not in {"forward", "backward"}:
        raise ValueError("日志搜索方向不合法")
    path = _resolve_log_path(task_id)
    if path is None:
        return _empty_log_search()

    key = fingerprint(path)
    total = index_for(key)[1]
    selected = search(key, safe_query, int(cursor), safe_direction)
    if selected is None:
        return {**_empty_log_search(), "total": total}
    index, record, ordinal, matches_total = selected
    return {
        "ok": True,
        "match": record,
        "match_index": index,
        "match_ordinal": ordinal,
        "matches_total": matches_total,
        "total": total,
    }


def _resolve_log_path(task_id: str):
    task_dir = _history_task_dir(task_id)
    if not task_dir.exists() or not task_dir.is_dir():
        raise FileNotFoundError("任务不存在")
    try:
        return _history_log_path(task_id)
    except FileNotFoundError:
        return None


def _log_search_text(value: dict[str, Any]) -> str:
    raw = value.get("line", value.get("message", value.get("text", value)))
    return str(raw).casefold()


def _empty_log_page(limit: int) -> dict[str, Any]:
    return {
        "ok": True,
        "logs": [],
        "offset": 0,
        "limit": limit,
        "returned": 0,
        "total": 0,
        "next_offset": 0,
        "has_more_before": False,
        "has_more_after": False,
    }


def _empty_log_search() -> dict[str, Any]:
    return {
        "ok": True,
        "match": None,
        "match_index": None,
        "match_ordinal": 0,
        "matches_total": 0,
        "total": 0,
    }
