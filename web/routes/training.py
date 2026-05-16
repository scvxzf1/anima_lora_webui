"""Training control and WebSocket routes."""

from __future__ import annotations

from aiohttp import web

from web.services.config_service import preflight_training_config


def setup_training_routes(app: web.Application) -> None:
    app.router.add_post("/api/training/preflight", handle_preflight)
    app.router.add_post("/api/training/start", handle_start)
    app.router.add_post("/api/training/preprocess", handle_preprocess)
    app.router.add_post("/api/training/stop", handle_stop)
    app.router.add_get("/api/training/status", handle_status)
    app.router.add_get("/api/training/metrics", handle_metrics)
    app.router.add_get("/api/training/logs", handle_logs)
    app.router.add_get("/api/training/history", handle_history_list)
    app.router.add_get("/api/training/history/{task_id}", handle_history_detail)
    app.router.add_patch("/api/training/history/{task_id}", handle_history_update)
    app.router.add_delete("/api/training/history/{task_id}", handle_history_delete)
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
    preflight = preflight_training_config(variant, preset, methods_subdir)
    if not preflight.get("ok", False):
        return web.json_response({
            "ok": False,
            "error": "预检测发现错误，已阻止训练启动",
            "preflight": preflight,
        }, status=400)
    try:
        await svc.start(variant, preset, extra_args, methods_subdir)
        return web.json_response({"ok": True, "message": "训练已启动"})
    except RuntimeError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=409)


async def handle_preprocess(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    data = await request.json()
    variant = data.get("variant", "lora")
    preset = data.get("preset", "default")
    methods_subdir = data.get("methods_subdir", "gui-methods")
    extra_args = data.get("extra_args", [])
    train_after = bool(data.get("train_after", False))
    try:
        preflight = preflight_training_config(variant, preset, methods_subdir)
    except Exception as e:
        return web.json_response({"ok": False, "error": f"预处理预检测失败: {e}"}, status=400)
    if not _preflight_allows_preprocess(preflight):
        return web.json_response({
            "ok": False,
            "error": "当前配置还有预处理无法自动解决的问题，请先修正预检测错误",
            "preflight": preflight,
        }, status=400)
    try:
        await svc.start_preprocess(variant, preset, methods_subdir, extra_args, train_after)
        message = "预处理已启动，完成后会自动开始训练" if train_after else "预处理已启动"
        return web.json_response({"ok": True, "message": message})
    except RuntimeError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=409)


def _preflight_allows_preprocess(result: dict) -> bool:
    checks = result.get("checks") or []
    errors = result.get("errors") or []
    allowed_error_keys = {"training_images", "resized_image_dir"}
    if any(item.get("key") not in allowed_error_keys for item in errors):
        return False
    return any(
        item.get("key") == "source_image_dir" and item.get("level") == "ok"
        for item in checks
    )


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


async def handle_history_list(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    return web.json_response({"ok": True, "tasks": svc.list_history_tasks()})


async def handle_history_detail(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    task_id = request.match_info["task_id"]
    try:
        return web.json_response(svc.get_history_task(task_id))
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_history_update(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    task_id = request.match_info["task_id"]
    try:
        data = await request.json()
        return web.json_response(svc.update_history_task(task_id, data))
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_history_delete(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    task_id = request.match_info["task_id"]
    try:
        return web.json_response(svc.delete_history_task(task_id))
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except RuntimeError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=409)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


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
