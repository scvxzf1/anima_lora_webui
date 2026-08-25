"""Paged access to complete history log files."""

from __future__ import annotations

import json
from collections import deque
from typing import Any

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
    page_end = requested_offset + safe_limit if requested_offset is not None else None
    tail: deque[tuple[int, dict[str, Any]]] = deque(maxlen=safe_limit)
    page: list[tuple[int, dict[str, Any]]] = []
    total = 0

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            index = total
            total += 1
            try:
                value = json.loads(line)
            except Exception:
                continue
            if not isinstance(value, dict):
                continue
            if requested_offset is None:
                tail.append((index, value))
            elif requested_offset <= index < page_end:
                page.append((index, value))

    selected = list(tail) if requested_offset is None else page
    resolved_offset = selected[0][0] if selected else min(requested_offset or total, total)
    next_offset = min(total, (requested_offset if requested_offset is not None else resolved_offset) + safe_limit)
    return {
        "ok": True,
        "logs": [value for _index, value in selected],
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

    target = int(cursor)
    first_match = None
    last_match = None
    selected = None
    matches_total = 0
    total = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            index = total
            total += 1
            try:
                value = json.loads(line)
            except Exception:
                continue
            if not isinstance(value, dict) or safe_query not in _log_search_text(value):
                continue
            matches_total += 1
            match = (index, value, matches_total)
            if first_match is None:
                first_match = match
            last_match = match
            if safe_direction == "forward" and selected is None and index >= target:
                selected = match
            elif safe_direction == "backward" and index <= target:
                selected = match

    selected = selected or (first_match if safe_direction == "forward" else last_match)
    if selected is None:
        return {**_empty_log_search(), "total": total}
    index, record, ordinal = selected
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
