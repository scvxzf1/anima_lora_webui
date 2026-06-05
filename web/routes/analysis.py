"""Static LoRA weight analysis routes."""

from __future__ import annotations

from aiohttp import web

from web.services.weight_analysis_service import (
    MAX_UPLOAD_WEIGHT_BYTES,
    inspect_weight,
    inspect_weight_bytes,
    list_analysis_weights,
)


def setup_analysis_routes(app: web.Application) -> None:
    app.router.add_get("/api/analysis/weights", handle_analysis_weights)
    app.router.add_post("/api/analysis/inspect", handle_analysis_inspect)
    app.router.add_post("/api/analysis/inspect-upload", handle_analysis_inspect_upload)


async def handle_analysis_weights(request: web.Request) -> web.Response:
    try:
        task = _selected_history_task(request)
        return web.json_response(list_analysis_weights(
            task=task,
            allow_latest_fallback=not _has_task_selection(request),
            training_service=request.app.get("training_service"),
            include_archived=str(request.query.get("include_archived") or "0").lower() in {"1", "true", "yes"},
        ))
    except FileNotFoundError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=404)
    except (ValueError, OSError) as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)


async def handle_analysis_inspect(request: web.Request) -> web.Response:
    data = await request.json()
    path = str(data.get("path") or "").strip()
    try:
        task = _selected_history_task(request)
        return web.json_response(inspect_weight(path, task=task))
    except FileNotFoundError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=404)
    except (ValueError, OSError) as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)


async def handle_analysis_inspect_upload(request: web.Request) -> web.Response:
    try:
        reader = await request.multipart()
        part = await reader.next()
        while part is not None and part.name != "file":
            part = await reader.next()
        if part is None:
            raise ValueError("没有收到权重文件")
        filename = str(part.filename or "uploaded.safetensors")
        data = bytearray()
        while True:
            chunk = await part.read_chunk()
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > MAX_UPLOAD_WEIGHT_BYTES:
                raise ValueError(f"拖入文件过大，最大支持 {MAX_UPLOAD_WEIGHT_BYTES // (1024 * 1024)} MiB")
        if not data:
            raise ValueError("拖入文件为空")
        return web.json_response(inspect_weight_bytes(bytes(data), filename=filename))
    except FileNotFoundError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=404)
    except (ValueError, OSError) as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)


def _has_task_selection(request: web.Request) -> bool:
    return bool((request.query.get("task_id") or "").strip())


def _selected_history_task(request: web.Request) -> dict:
    task_id = (request.query.get("task_id") or "").strip()
    if not task_id:
        return {}
    svc = request.app.get("training_service")
    if not svc:
        raise ValueError("训练服务未初始化")
    payload = svc.get_history_task(task_id)
    task = payload.get("task") if isinstance(payload, dict) else {}
    if not isinstance(task, dict):
        return {}
    if task.get("job") != "training":
        raise ValueError("只能选择训练任务读取权重")
    return task
