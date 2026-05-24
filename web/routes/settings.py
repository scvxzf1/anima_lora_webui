"""Global settings routes."""

from __future__ import annotations

from aiohttp import web

from web.services.settings_service import get_global_settings, save_global_settings


def setup_settings_routes(app: web.Application) -> None:
    app.router.add_get("/api/settings/global", handle_global_settings_get)
    app.router.add_put("/api/settings/global", handle_global_settings_put)


async def handle_global_settings_get(request: web.Request) -> web.Response:
    return web.json_response(get_global_settings())


async def handle_global_settings_put(request: web.Request) -> web.Response:
    data = await request.json()
    try:
        return web.json_response(save_global_settings(data))
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)
