"""Training control and WebSocket routes."""

from __future__ import annotations

from aiohttp import web

from web.services.config_service import is_web_runtime_config, preflight_training_config
from web.services.training_service import inspect_continue_lora_weight


def setup_training_routes(app: web.Application) -> None:
    app.router.add_post("/api/training/preflight", handle_preflight)
    app.router.add_post("/api/training/continue-lora/inspect", handle_continue_lora_inspect)
    app.router.add_post("/api/training/start", handle_start)
    app.router.add_post("/api/training/resume", handle_resume)
    app.router.add_post("/api/training/preprocess", handle_preprocess)
    app.router.add_post("/api/training/stop", handle_stop)
    app.router.add_get("/api/training/status", handle_status)
    app.router.add_get("/api/training/metrics", handle_metrics)
    app.router.add_get("/api/training/logs", handle_logs)
    app.router.add_get("/api/training/gpus", handle_gpus)
    app.router.add_get("/api/training/queue", handle_queue_status)
    app.router.add_post("/api/training/queue/start", handle_queue_start)
    app.router.add_post("/api/training/queue/resume", handle_queue_resume)
    app.router.add_post("/api/training/queue/settings", handle_queue_settings)
    app.router.add_post("/api/training/queue/cancel-waiting", handle_queue_cancel_waiting)
    app.router.add_post("/api/training/queue/clear", handle_queue_clear)
    app.router.add_post("/api/training/queue/{item_id}/move", handle_queue_move)
    app.router.add_post("/api/training/queue/{item_id}/retry", handle_queue_retry)
    app.router.add_delete("/api/training/queue/{item_id}", handle_queue_cancel)
    app.router.add_post("/api/training/queue/pause", handle_queue_pause)
    app.router.add_get("/api/training/history", handle_history_list)
    app.router.add_post("/api/training/history/batch", handle_history_batch)
    app.router.add_get("/api/training/history/collections/settings", handle_history_collection_settings_get)
    app.router.add_put("/api/training/history/collections/settings", handle_history_collection_settings_put)
    app.router.add_get("/api/training/history/config-group/timeline", handle_config_group_timeline)
    app.router.add_get("/api/training/history/{task_id}", handle_history_detail)
    app.router.add_get("/api/training/history/{task_id}/resume-options", handle_history_resume_options)
    app.router.add_patch("/api/training/history/{task_id}", handle_history_update)
    app.router.add_delete("/api/training/history/{task_id}", handle_history_delete)
    app.router.add_get("/ws/training", handle_ws)


async def handle_preflight(request: web.Request) -> web.Response:
    data = await request.json()
    variant = data.get("variant", "lora")
    preset = data.get("preset", "default")
    methods_subdir = data.get("methods_subdir", "gui-methods")
    config_file = str(data.get("config_file") or "").strip() or None
    if _is_cli_only_spd(variant, methods_subdir):
        return web.json_response(_cli_only_spd_payload(variant, preset, methods_subdir), status=400)
    try:
        result = preflight_training_config(variant, preset, methods_subdir, config_file=config_file)
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


async def handle_continue_lora_inspect(request: web.Request) -> web.Response:
    data = await request.json()
    path = str(data.get("path") or "").strip()
    variant = data.get("variant", "lora")
    preset = data.get("preset", "default")
    methods_subdir = data.get("methods_subdir", "gui-methods")
    config_file = str(data.get("config_file") or "").strip() or None
    try:
        return web.json_response(inspect_continue_lora_weight(
            path,
            variant=variant,
            preset=preset,
            methods_subdir=methods_subdir,
            config_file=config_file,
        ))
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_start(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    data = await request.json()
    variant = data.get("variant", "lora")
    preset = data.get("preset", "default")
    methods_subdir = data.get("methods_subdir", "gui-methods")
    extra_args = data.get("extra_args", [])
    gpu_whitelist = data.get("gpu_whitelist")
    config_file = str(data.get("config_file") or "").strip() or None
    confirmed = bool(data.get("confirmed", False))
    confirm_preprocess = bool(data.get("confirm_preprocess", False))
    continue_info = _continue_lora_info_from_request(data)
    if _is_cli_only_spd(variant, methods_subdir):
        return web.json_response({"ok": False, "error": _cli_only_spd_message()}, status=400)
    try:
        preflight = preflight_training_config(variant, preset, methods_subdir, config_file=config_file)
        if not preflight.get("ok", False):
            return web.json_response({
                "ok": False,
                "error": "预检测发现错误，已阻止训练启动",
                "preflight": preflight,
            }, status=400)
        needs_preprocess = not config_file or not is_web_runtime_config(config_file)
        if not confirmed or (needs_preprocess and not confirm_preprocess):
            return web.json_response({
                "ok": False,
                "error": "请先确认训练前预检测结果",
                "preflight": preflight,
                "requires_confirmation": True,
                "requires_preprocess_confirmation": needs_preprocess,
            }, status=409)
        if needs_preprocess:
            await svc.start_preprocess(
                variant,
                preset,
                methods_subdir,
                extra_args,
                True,
                gpu_whitelist=gpu_whitelist,
                config_file=config_file,
                continue_info=continue_info,
            )
            return web.json_response({
                "ok": True,
                "job": "preprocess",
                "train_after": True,
                "message": "预处理已启动，完成后会自动开始训练",
            })
        await svc.start(
            variant,
            preset,
            extra_args,
            methods_subdir,
            gpu_whitelist=gpu_whitelist,
            config_file=config_file,
            continue_info=continue_info,
            use_runtime_dir=False,
        )
        return web.json_response({"ok": True, "job": "training", "train_after": False, "message": "训练已启动"})
    except RuntimeError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=409)
    except (FileNotFoundError, ValueError) as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_resume(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    data = await request.json()
    task_id = str(data.get("task_id") or "").strip()
    checkpoint = str(data.get("checkpoint") or "").strip()
    gpu_whitelist = data.get("gpu_whitelist")
    if not task_id:
        return web.json_response({"ok": False, "error": "缺少 task_id"}, status=400)
    try:
        payload = await svc.resume_from_history_task(task_id, checkpoint or None, gpu_whitelist=gpu_whitelist)
        return web.json_response(payload)
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)
    except OSError as e:
        return web.json_response({"ok": False, "error": f"删除运行缓存失败: {e}"}, status=500)
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
    confirmed = bool(data.get("confirmed", False))
    confirm_train_after = bool(
        data.get("confirm_train_after", False)
        or data.get("confirm_preprocess", False)
    )
    gpu_whitelist = data.get("gpu_whitelist")
    config_file = str(data.get("config_file") or "").strip() or None
    continue_info = _continue_lora_info_from_request(data)
    if _is_cli_only_spd(variant, methods_subdir):
        return web.json_response({"ok": False, "error": _cli_only_spd_message()}, status=400)
    try:
        preflight = preflight_training_config(variant, preset, methods_subdir, config_file=config_file)
    except Exception as e:
        return web.json_response({"ok": False, "error": f"预处理预检测失败: {e}"}, status=400)
    if not _preflight_allows_preprocess(preflight):
        return web.json_response({
            "ok": False,
            "error": "当前配置还有预处理无法自动解决的问题，请先修正预检测错误",
            "preflight": preflight,
        }, status=400)
    if train_after and (not confirmed or not confirm_train_after):
        return web.json_response({
            "ok": False,
            "error": "请先确认预处理完成后自动开始训练",
            "preflight": preflight,
            "requires_confirmation": True,
            "requires_preprocess_confirmation": True,
            "requires_train_after_confirmation": True,
        }, status=409)
    try:
        await svc.start_preprocess(
            variant,
            preset,
            methods_subdir,
            extra_args,
            train_after,
            gpu_whitelist=gpu_whitelist,
            config_file=config_file,
            continue_info=continue_info,
        )
        message = "预处理已启动，完成后会自动开始训练" if train_after else "预处理已启动"
        return web.json_response({"ok": True, "message": message})
    except RuntimeError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=409)
    except (FileNotFoundError, ValueError) as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


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


def _continue_lora_info_from_request(data: dict) -> dict | None:
    path = str(data.get("continue_from_weight_abs_path") or "").strip()
    if not path:
        return None
    return {
        "continue_from_weight_abs_path": path,
        "continue_from_weight_name": str(data.get("continue_from_weight_name") or "").strip(),
        "continue_from_weight_kind": str(data.get("continue_from_weight_kind") or "").strip(),
    }


def _is_cli_only_spd(variant: str, methods_subdir: str) -> bool:
    return str(methods_subdir or "") == "methods" and str(variant or "") == "spd"


def _cli_only_spd_message() -> str:
    return "SPD 是 scripts/distill_spd.py 使用的 CLI 实验配置，当前 Web 训练入口不会用 train.py 启动它。请通过 tasks.py exp-spd 或对应 CLI 流程运行。"


def _cli_only_spd_payload(variant: str, preset: str, methods_subdir: str) -> dict:
    message = _cli_only_spd_message()
    item = {"level": "error", "key": "spd", "message": message}
    return {
        "ok": False,
        "variant": variant,
        "preset": preset,
        "methods_subdir": methods_subdir,
        "summary": {"errors": 1, "warnings": 0, "checks": 1},
        "checks": [item],
        "errors": [item],
        "warnings": [],
    }


def _preflight_requiring_preprocess(result: dict) -> dict:
    checks = list(result.get("checks") or [])
    item = {
        "level": "error",
        "key": "runtime_preprocess",
        "message": "Web 训练会写入独立运行目录；请先点击“开始预处理”，完成后会自动训练。",
    }
    checks.append(item)
    errors = list(result.get("errors") or [])
    errors.append(item)
    summary = dict(result.get("summary") or {})
    summary["errors"] = len(errors)
    summary["checks"] = len(checks)
    return {
        **result,
        "ok": False,
        "summary": summary,
        "checks": checks,
        "errors": errors,
    }


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


async def handle_gpus(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    return web.json_response({"ok": True, "gpus": await svc.list_gpus()})


async def handle_queue_status(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    return web.json_response(svc.get_queue_snapshot())


async def handle_queue_start(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    data = await request.json()
    variant = data.get("variant", "lora")
    preset = data.get("preset", "default")
    methods_subdir = data.get("methods_subdir", "gui-methods")
    extra_args = data.get("extra_args", [])
    gpu_whitelist = data.get("gpu_whitelist")
    config_file = str(data.get("config_file") or "").strip() or None
    confirmed = bool(data.get("confirmed", False))
    confirm_preprocess = bool(
        data.get("confirm_preprocess", False)
        or data.get("confirm_train_after", False)
    )
    continue_info = _continue_lora_info_from_request(data)
    if _is_cli_only_spd(variant, methods_subdir):
        return web.json_response({"ok": False, "error": _cli_only_spd_message()}, status=400)
    try:
        preflight = preflight_training_config(variant, preset, methods_subdir, config_file=config_file)
        needs_preprocess = not config_file or not is_web_runtime_config(config_file)
        if not preflight.get("ok", False):
            if not (needs_preprocess and confirm_preprocess and _preflight_allows_preprocess(preflight)):
                return web.json_response({
                    "ok": False,
                    "error": "预检测发现错误，已阻止加入队列",
                    "preflight": preflight,
                }, status=400)
        if not confirmed or (needs_preprocess and not confirm_preprocess):
            return web.json_response({
                "ok": False,
                "error": "请先确认训练前预检测结果",
                "preflight": preflight,
                "requires_confirmation": True,
                "requires_preprocess_confirmation": needs_preprocess,
            }, status=409)
        payload = await svc.enqueue_training(
            variant,
            preset,
            methods_subdir,
            extra_args=extra_args,
            config_file=config_file,
            gpu_whitelist=gpu_whitelist,
            continue_info=continue_info,
            requires_preprocess=needs_preprocess,
        )
        return web.json_response(payload)
    except (FileNotFoundError, ValueError) as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_queue_resume(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    data = await request.json()
    task_id = str(data.get("task_id") or "").strip()
    checkpoint = str(data.get("checkpoint") or "").strip()
    gpu_whitelist = data.get("gpu_whitelist")
    if not task_id:
        return web.json_response({"ok": False, "error": "缺少 task_id"}, status=400)
    try:
        return web.json_response(await svc.enqueue_resume_from_history_task(
            task_id,
            checkpoint or None,
            gpu_whitelist=gpu_whitelist,
        ))
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_queue_move(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    data = await request.json()
    direction = str(data.get("direction") or "").strip()
    if direction not in {"up", "down", "top", "bottom"}:
        return web.json_response({"ok": False, "error": "direction 必须是 up、down、top 或 bottom"}, status=400)
    try:
        return web.json_response(await svc.move_queue_item(request.match_info["item_id"], direction))
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_queue_retry(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    try:
        return web.json_response(await svc.retry_queue_item(request.match_info["item_id"]))
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_queue_cancel_waiting(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    return web.json_response(await svc.cancel_waiting_queue_items())


async def handle_queue_clear(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    return web.json_response(await svc.clear_finished_queue_items())


async def handle_queue_settings(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    data = await request.json()
    paused = data.get("paused") if "paused" in data else None
    failure_policy = data.get("failure_policy") if "failure_policy" in data else None
    try:
        return web.json_response(await svc.set_queue_settings(
            paused=bool(paused) if paused is not None else None,
            failure_policy=str(failure_policy) if failure_policy is not None else None,
        ))
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_queue_cancel(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    try:
        data = await request.json()
    except Exception:
        data = {}
    try:
        return web.json_response(await svc.cancel_queue_item(
            request.match_info["item_id"],
            delete_runtime=bool(data.get("delete_runtime")),
        ))
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)
    except OSError as e:
        return web.json_response({"ok": False, "error": f"删除运行缓存失败: {e}"}, status=500)


async def handle_queue_pause(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    data = await request.json()
    paused = bool(data.get("paused", False))
    return web.json_response(await svc.set_queue_paused(paused))


async def handle_history_list(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    include_archived = str(request.query.get("include_archived") or "0").lower() in {"1", "true", "yes"}
    limit = _positive_query_int(request.query.get("limit"))
    return web.json_response({
        "ok": True,
        "tasks": svc.list_history_tasks(include_archived=include_archived, limit=limit),
    })


async def handle_history_batch(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    try:
        data = await request.json()
        return web.json_response(svc.batch_update_history_tasks(data))
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except RuntimeError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=409)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_history_collection_settings_get(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    return web.json_response(svc.get_history_collection_settings())


async def handle_history_collection_settings_put(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    try:
        data = await request.json()
        return web.json_response(svc.save_history_collection_settings(data))
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)
    except OSError as e:
        return web.json_response({"ok": False, "error": f"保存集合设置失败: {e}"}, status=500)


def _positive_query_int(value) -> int | None:
    try:
        number = int(str(value or "").strip())
    except ValueError:
        return None
    return number if number > 0 else None


async def handle_history_detail(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    task_id = request.match_info["task_id"]
    try:
        return web.json_response(svc.get_history_task(task_id))
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_config_group_timeline(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    methods_subdir = str(request.query.get("methods_subdir") or "").strip()
    variant = str(request.query.get("variant") or "").strip()
    preset = str(request.query.get("preset") or "default").strip() or "default"
    group_key = str(request.query.get("group_key") or "").strip()
    include_archived = str(request.query.get("include_archived") or "0").lower() in {"1", "true", "yes"}
    task_ids = _timeline_task_ids_from_query(request)
    if not task_ids and not group_key and (not methods_subdir or not variant):
        return web.json_response({"ok": False, "error": "缺少 group_key 或 methods_subdir/variant"}, status=400)
    try:
        return web.json_response(svc.get_config_group_timeline(
            methods_subdir,
            variant,
            preset,
            group_key=group_key,
            include_archived=include_archived,
            task_ids=task_ids,
        ))
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


def _timeline_task_ids_from_query(request: web.Request) -> list[str]:
    values: list[str] = []
    for key in ("task_id", "task_ids"):
        values.extend(request.query.getall(key, []))
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        for part in str(value or "").split(","):
            task_id = part.strip()
            if not task_id or task_id in seen:
                continue
            out.append(task_id)
            seen.add(task_id)
    return out


async def handle_history_resume_options(request: web.Request) -> web.Response:
    svc = request.app["training_service"]
    task_id = request.match_info["task_id"]
    try:
        return web.json_response(svc.get_resume_options(task_id))
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
