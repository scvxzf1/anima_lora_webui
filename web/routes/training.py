"""Training control and WebSocket routes."""

from __future__ import annotations

from aiohttp import web

from web.services.config_service import preflight_training_config


def setup_training_routes(app: web.Application) -> None:
    app.router.add_post("/api/training/preflight", handle_preflight)
    app.router.add_post("/api/training/start", handle_start)
    app.router.add_post("/api/training/stop", handle_stop)
    app.router.add_get("/api/training/status", handle_status)
    app.router.add_get("/api/training/metrics", handle_metrics)
    app.router.add_get("/api/training/logs", handle_logs)
    app.router.add_get("/ws/training", handle_ws)


async def handle_preflight(request: web.Request) -> web.Response:
    data = await request.json()
    variant = data.get("variant", "lora")
    preset = data.get("preset", "default")
    methods_subdir = data.get("methods_subdir", "gui-methods")
    try:
        result = preflight_training_config(variant, preset, methods_subdir)
        return web.json_response(result)
    except Exception as e:
        return web.json_response({
            "ok": False,
            "variant": variant,
            "preset": preset,
            "methods_subdir": methods_subdir,
            "summary": {"errors": 1, "warnings": 0, "checks": 1},
            "checks": [{
                "level": "error",
                "key": "preflight",
                "message": f"预检测失败: {e}",
            }],
            "errors": [{
                "level": "error",
                "key": "preflight",
                "message": f"预检测失败: {e}",
            }],
            "warnings": [],
        }, status=400)


async def handle_start(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    data = await request.json()
    variant = data.get("variant", "lora")
    preset = data.get("preset", "default")
    methods_subdir = data.get("methods_subdir", "gui-methods")
    extra_args = data.get("extra_args", [])
    try:
        await svc.start(variant, preset, extra_args, methods_subdir)
        return web.json_response({"ok": True, "message": "训练已启动"})
    except RuntimeError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=409)


async def handle_stop(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    await svc.stop()
    return web.json_response({"ok": True, "message": "训练已停止"})


async def handle_status(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    return web.json_response(svc.get_status_snapshot())


async def handle_metrics(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    return web.json_response(svc.get_metrics_history())


async def handle_logs(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    after = int(request.query.get("after", "0") or 0)
    limit = int(request.query.get("limit", "1000") or 1000)
    return web.json_response({
        "records": svc.get_log_records(after=after, limit=limit),
    })


async def handle_ws(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    svc = request.app["training_service"]
    svc.subscribe(ws)

    try:
        async for msg in ws:
            pass
    finally:
        svc.unsubscribe(ws)

    return ws
