"""History listing with stable cursors; the storage limit=0 contract means all records."""

from __future__ import annotations

import base64
import binascii
import json
import math

from aiohttp import web


def history_order_key(task: dict) -> tuple[float, str]:
    try:
        timestamp = float(task.get("started_at") or 0)
    except (TypeError, ValueError):
        timestamp = 0.0
    return (timestamp if math.isfinite(timestamp) else 0.0, str(task.get("id") or ""))


def _encode_cursor(task: dict) -> str:
    raw = json.dumps(history_order_key(task), separators=(",", ":")).encode()
    return "v1." + base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(raw: str) -> tuple[float, str]:
    try:
        if not raw.startswith("v1.") or len(raw) > 4096:
            raise ValueError
        value = json.loads(base64.b64decode(raw[3:] + "=" * (-len(raw[3:]) % 4), altchars=b"-_", validate=True))
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError
        timestamp, task_id = value
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise ValueError
        if not isinstance(task_id, str):
            raise ValueError
        return timestamp, task_id
    except (ValueError, TypeError, binascii.Error, UnicodeError) as exc:
        raise ValueError("无效的历史分页游标") from exc


def history_page(tasks: list[dict], *, limit: int, cursor: str = "") -> dict:
    ordered = sorted(tasks, key=history_order_key, reverse=True)
    total = len(ordered)
    if cursor:
        if cursor.isascii() and cursor.isdecimal():
            ordered = ordered[int(cursor):]  # Compatibility with the initial offset API.
        else:
            boundary = _decode_cursor(cursor)
            ordered = [task for task in ordered if history_order_key(task) < boundary]
    page = ordered[:limit]
    return {
        "tasks": page,
        "total": total,
        "next_cursor": _encode_cursor(page[-1]) if len(ordered) > len(page) else None,
    }


async def history_list_response(request: web.Request) -> web.Response:
    try:
        limit = int(request.query.get("limit") or 100)
        if limit <= 0 or limit > 100000:
            raise ValueError("limit 必须在 1 到 100000 之间")
        search = str(request.query.get("q") or "").strip()
        include_archived = str(request.query.get("include_archived") or "0").lower() in {"1", "true", "yes"}
        options = {"include_archived": include_archived, "limit": 0}
        if search:
            options["search"] = search
        tasks = request.app["training_service"].list_history_tasks(**options)
        page = history_page(tasks, limit=limit, cursor=str(request.query.get("cursor") or ""))
        return web.json_response({"ok": True, **page, "search": search})
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
