"""Preview image browsing routes."""

from __future__ import annotations

from aiohttp import web

from web.services.preview_service import (
    get_preview_settings,
    list_preview_images,
    list_training_weights,
    resolve_preview_image,
    save_preview_settings,
)


def setup_preview_routes(app: web.Application) -> None:
    app.router.add_get("/api/preview/settings", handle_preview_settings_get)
    app.router.add_put("/api/preview/settings", handle_preview_settings_put)
    app.router.add_get("/api/preview/images", handle_preview_images)
    app.router.add_get("/api/preview/image", handle_preview_image)
    app.router.add_get("/api/preview/weights", handle_preview_weights)


async def handle_preview_settings_get(request: web.Request) -> web.Response:
    try:
        sample_dir = _selected_sample_dir(request)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
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
        task = _selected_history_task(request)
        limit = int(request.query.get("limit", "200") or 200)
        payload = list_preview_images(
            source,
            current_task_sample_dir=_selected_sample_dir(request, task=task),
            sample_config=_selected_sample_config(request, task=task),
            task=task,
            task_id=task.get("id") if task else "",
            task_label=_history_task_label(task) if task else "",
            limit=limit,
        )
        return web.json_response(payload)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)


async def handle_preview_image(request: web.Request) -> web.StreamResponse:
    file_path = request.query.get("file", "")
    if not file_path:
        return web.json_response({"ok": False, "error": "缺少 file 参数"}, status=400)
    try:
        task = _selected_history_task(request)
        path = resolve_preview_image(
            file_path,
            allowed_sample_dir=_selected_sample_dir(request, task=task),
        )
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=403)
    return web.FileResponse(path)


async def handle_preview_weights(request: web.Request) -> web.Response:
    try:
        task = _selected_history_task(request)
        return web.json_response(list_training_weights(task))
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)


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


def _selected_history_task(request: web.Request) -> dict:
    task_id = (request.query.get("task_id") or "").strip()
    if not task_id:
        return {}
    svc = request.app.get("training_service")
    if not svc:
        raise ValueError("训练服务未初始化")
    try:
        payload = svc.get_history_task(task_id)
    except FileNotFoundError as exc:
        raise FileNotFoundError(str(exc)) from exc
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    task = payload.get("task") if isinstance(payload, dict) else {}
    if not isinstance(task, dict):
        return {}
    if task.get("job") != "training":
        raise ValueError("只能选择训练任务读取样张")
    return task


def _selected_sample_dir(request: web.Request, *, task: dict | None = None) -> str:
    task = task if task is not None else _selected_history_task(request)
    if task:
        return str(task.get("sample_dir") or "")
    return _current_sample_dir(request)


def _selected_sample_config(request: web.Request, *, task: dict | None = None) -> dict:
    task = task if task is not None else _selected_history_task(request)
    if task:
        value = task.get("sample_config") or {}
        return value if isinstance(value, dict) else {}
    return _current_sample_config(request)


def _history_task_label(task: dict) -> str:
    return str(
        task.get("name")
        or f"{task.get('methods_subdir') or '-'} / {task.get('variant') or task.get('id') or '-'}"
    )
