"""Image test generation routes."""

from __future__ import annotations

from aiohttp import web


def setup_image_test_routes(app: web.Application) -> None:
    app.router.add_get("/api/image-test/status", handle_image_test_status)
    app.router.add_post("/api/image-test/start", handle_image_test_start)
    app.router.add_post("/api/image-test/stop", handle_image_test_stop)


async def handle_image_test_status(request: web.Request) -> web.Response:
    svc = request.app.get("image_test_service")
    if svc is None:
        return web.json_response({"ok": False, "error": "生图测试服务未初始化"}, status=503)
    return web.json_response(svc.get_status_snapshot())


async def handle_image_test_start(request: web.Request) -> web.Response:
    svc = request.app.get("image_test_service")
    if svc is None:
        return web.json_response({"ok": False, "error": "生图测试服务未初始化"}, status=503)
    data = await request.json()
    try:
        payload = await svc.start(data)
        return web.json_response(payload)
    except RuntimeError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=409)
    except (ValueError, FileNotFoundError) as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)


async def handle_image_test_stop(request: web.Request) -> web.Response:
    svc = request.app.get("image_test_service")
    if svc is None:
        return web.json_response({"ok": False, "error": "生图测试服务未初始化"}, status=503)
    try:
        payload = await svc.stop()
        return web.json_response(payload)
    except RuntimeError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=409)
