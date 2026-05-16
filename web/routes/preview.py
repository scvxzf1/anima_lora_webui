"""Preview image browsing routes."""

from __future__ import annotations

from aiohttp import web

from web.services.preview_service import (
    get_preview_settings,
    list_preview_images,
    resolve_preview_image,
    save_preview_settings,
)


def setup_preview_routes(app: web.Application) -> None:
    app.router.add_get("/api/preview/settings", handle_preview_settings_get)
    app.router.add_put("/api/preview/settings", handle_preview_settings_put)
    app.router.add_get("/api/preview/images", handle_preview_images)
    app.router.add_get("/api/preview/image", handle_preview_image)


async def handle_preview_settings_get(request: web.Request) -> web.Response:
    sample_dir = _current_sample_dir(request)
    return web.json_response(get_preview_settings(sample_dir))


async def handle_preview_settings_put(request: web.Request) -> web.Response:
    data = await request.json()
    try:
        return web.json_response(save_preview_settings(data))
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_preview_images(request: web.Request) -> web.Response:
    source = request.query.get("source", "training")
    try:
        limit = int(request.query.get("limit", "200") or 200)
        payload = list_preview_images(
            source,
            current_task_sample_dir=_current_sample_dir(request),
            sample_config=_current_sample_config(request),
            limit=limit,
        )
        return web.json_response(payload)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_preview_image(request: web.Request) -> web.StreamResponse:
    file_path = request.query.get("file", "")
    if not file_path:
        return web.json_response({"ok": False, "error": "缺少 file 参数"}, status=400)
    try:
        path = resolve_preview_image(file_path, allowed_sample_dir=_current_sample_dir(request))
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=403)
    return web.FileResponse(path)


def _current_sample_dir(request: web.Request) -> str:
    svc = request.app.get("training_service")
    if not svc:
        return ""
    return str(getattr(svc, "current_sample_dir", "") or "")


def _current_sample_config(request: web.Request) -> dict:
    svc = request.app.get("training_service")
    if not svc:
        return {}
    value = getattr(svc, "current_sample_config", {}) or {}
    return value if isinstance(value, dict) else {}
