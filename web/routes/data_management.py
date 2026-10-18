"""Safe data-management inspection and cleanup endpoints."""

from __future__ import annotations

import asyncio

from aiohttp import web

from web.services.data_management_retention import DataManagementRetentionService
from web.services.data_management_service import DataManagementError, DataManagementService


def setup_data_management_routes(app: web.Application) -> None:
    app.router.add_get("/api/data-management/roots", handle_roots)
    app.router.add_get("/api/data-management/entries", handle_entries)
    app.router.add_post("/api/data-management/cleanup/preview", handle_cleanup_preview)
    app.router.add_post("/api/data-management/cleanup", handle_cleanup)
    app.router.add_post("/api/data-management/retention/preview", handle_retention_preview)
    app.router.add_post("/api/data-management/retention/cleanup", handle_retention_cleanup)


def _error_response(exc: DataManagementError) -> web.Response:
    return web.json_response({"ok": False, "error": exc.message, **exc.extra}, status=exc.status)


_META_MODES = frozenset({"meta", "skeleton"})


async def handle_roots(request: web.Request) -> web.Response:
    mode = str(request.query.get("mode", "") or "").strip().lower()
    try:
        result = await asyncio.to_thread(
            DataManagementService(request.app).list_roots,
            request.query.get("refresh", ""),
            mode not in _META_MODES,
        )
        return web.json_response(result)
    except DataManagementError as exc:
        return _error_response(exc)
    except (OSError, RuntimeError, ValueError):
        return web.json_response({"ok": False, "error": "目录扫描暂时不可用"}, status=503)


async def handle_entries(request: web.Request) -> web.Response:
    root_id = request.query.get("root_id")
    relative = request.query.get("path", "")
    refresh = request.query.get("refresh", "")
    if not root_id:
        return web.json_response({"ok": False, "error": "缺少 root_id"}, status=400)
    try:
        result = await asyncio.to_thread(
            DataManagementService(request.app).list_entries, root_id, relative, refresh,
        )
        return web.json_response(result)
    except DataManagementError as exc:
        return _error_response(exc)
    except (OSError, RuntimeError, ValueError):
        return web.json_response({"ok": False, "error": "目录扫描暂时不可用"}, status=503)


async def _json_object(request: web.Request) -> dict:
    try:
        payload = await request.json()
    except (TypeError, ValueError, web.HTTPException):
        raise DataManagementError("请求正文必须是有效 JSON 对象", 400)
    if not isinstance(payload, dict):
        raise DataManagementError("请求正文必须是 JSON 对象", 400)
    return payload


async def handle_cleanup_preview(request: web.Request) -> web.Response:
    try:
        payload = await _json_object(request)
        result = await asyncio.to_thread(
            DataManagementService(request.app).preview_cleanup,
            payload.get("root_id"), payload.get("items"),
        )
        return web.json_response(result)
    except DataManagementError as exc:
        return _error_response(exc)
    except (OSError, RuntimeError, ValueError):
        return web.json_response({"ok": False, "error": "清理计划暂时不可用"}, status=503)


async def handle_cleanup(request: web.Request) -> web.Response:
    try:
        payload = await _json_object(request)
        result = await asyncio.to_thread(
            DataManagementService(request.app).cleanup,
            payload.get("root_id"), payload.get("items"), payload.get("expected_items"),
            payload.get("plan_revision"), payload.get("confirmed"),
        )
        return web.json_response(result)
    except DataManagementError as exc:
        return _error_response(exc)
    except (OSError, RuntimeError, ValueError):
        return web.json_response({"ok": False, "error": "永久删除暂时不可用"}, status=503)


async def handle_retention_preview(request: web.Request) -> web.Response:
    try:
        payload = await _json_object(request)
        result = await asyncio.to_thread(
            DataManagementRetentionService(request.app).preview,
            payload.get("root_id"), payload.get("dirs"), payload.get("policy"),
            payload.get("exclude_units"),
        )
        return web.json_response(result)
    except DataManagementError as exc:
        return _error_response(exc)
    except (OSError, RuntimeError, ValueError):
        return web.json_response({"ok": False, "error": "检查点策略预览暂时不可用"}, status=503)


async def handle_retention_cleanup(request: web.Request) -> web.Response:
    try:
        payload = await _json_object(request)
        result = await asyncio.to_thread(
            DataManagementRetentionService(request.app).cleanup,
            payload.get("root_id"), payload.get("dirs"), payload.get("policy"),
            payload.get("expected_items"), payload.get("plan_revision"), payload.get("confirmed"),
            payload.get("exclude_units"),
        )
        return web.json_response(result)
    except DataManagementError as exc:
        return _error_response(exc)
    except (OSError, RuntimeError, ValueError):
        return web.json_response({"ok": False, "error": "检查点策略清理暂时不可用"}, status=503)


__all__ = ["setup_data_management_routes"]
