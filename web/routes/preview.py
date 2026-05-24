"""Preview image browsing routes."""

from __future__ import annotations

from urllib.parse import quote

from aiohttp import web

from web.services.preview_service import (
    get_preview_settings,
    list_config_group_preview_images,
    list_config_group_training_weights,
    list_preview_images,
    list_training_weights,
    resolve_preview_image,
    resolve_training_weight,
    save_preview_settings,
)


def setup_preview_routes(app: web.Application) -> None:
    app.router.add_get("/api/preview/settings", handle_preview_settings_get)
    app.router.add_put("/api/preview/settings", handle_preview_settings_put)
    app.router.add_get("/api/preview/images", handle_preview_images)
    app.router.add_get("/api/preview/image", handle_preview_image)
    app.router.add_get("/api/preview/weights", handle_preview_weights)
    app.router.add_get("/api/preview/weight", handle_preview_weight_download)


async def handle_preview_settings_get(request: web.Request) -> web.Response:
    try:
        sample_dir = _selected_sample_dir(request)
        task_selected = _has_task_selection(request)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    return web.json_response(get_preview_settings(
        sample_dir,
        allow_latest_fallback=not task_selected,
    ))


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
        if source == "training" and request.query.get("mode") == "config_group":
            tasks = _selected_config_group_tasks(request)
            payload = list_config_group_preview_images(
                tasks,
                methods_subdir=str(request.query.get("methods_subdir") or ""),
                variant=str(request.query.get("variant") or ""),
                preset=str(request.query.get("preset") or "default"),
                limit=limit,
            )
            return web.json_response(payload)

        task = _selected_history_task(request)
        task_selected = _has_task_selection(request)
        payload = list_preview_images(
            source,
            current_task_sample_dir=_selected_sample_dir(request, task=task),
            sample_config=_selected_sample_config(request, task=task),
            task=task,
            task_id=task.get("id") if task else "",
            task_label=_history_task_label(task) if task else "",
            allow_latest_fallback=not task_selected,
            limit=limit,
        )
        return web.json_response(payload)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except OSError as e:
        return web.json_response({"ok": False, "error": f"读取预览资源失败: {e}"}, status=400)


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
        if request.query.get("mode") == "config_group":
            tasks = _selected_config_group_tasks(request)
            return web.json_response(list_config_group_training_weights(
                tasks,
                methods_subdir=str(request.query.get("methods_subdir") or ""),
                variant=str(request.query.get("variant") or ""),
                preset=str(request.query.get("preset") or "default"),
            ))
        task = _selected_history_task(request)
        return web.json_response(list_training_weights(
            task,
            allow_latest_fallback=not _has_task_selection(request),
        ))
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except OSError as e:
        return web.json_response({"ok": False, "error": f"读取权重资源失败: {e}"}, status=400)


async def handle_preview_weight_download(request: web.Request) -> web.StreamResponse:
    file_path = request.query.get("file", "")
    if not file_path:
        return web.json_response({"ok": False, "error": "缺少 file 参数"}, status=400)
    try:
        task = _selected_history_task(request)
        path = resolve_training_weight(file_path, task=task)
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=403)
    headers = {
        "Content-Disposition": f"attachment; filename*=UTF-8''{quote(path.name)}",
    }
    return web.FileResponse(path, headers=headers)


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


def _has_task_selection(request: web.Request) -> bool:
    return bool((request.query.get("task_id") or "").strip())


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


def _selected_config_group_tasks(request: web.Request) -> list[dict]:
    methods_subdir = str(request.query.get("methods_subdir") or "").strip()
    variant = str(request.query.get("variant") or "").strip()
    preset = str(request.query.get("preset") or "default").strip() or "default"
    include_archived = str(request.query.get("include_archived") or "0").lower() in {"1", "true", "yes"}
    if not methods_subdir or not variant:
        raise ValueError("缺少 methods_subdir 或 variant")

    svc = request.app.get("training_service")
    if not svc:
        raise ValueError("训练服务未初始化")

    group = {
        "methods_subdir": methods_subdir,
        "variant": variant,
        "preset": preset,
    }
    summaries = [
        task for task in svc.list_history_tasks()
        if task.get("job") == "training"
        and _task_config_group_matches(task, group)
        and (include_archived or not task.get("archived"))
    ]
    summaries.sort(key=lambda item: (float(item.get("started_at") or 0), str(item.get("id") or "")))
    tasks = []
    for summary in summaries:
        task_id = str(summary.get("id") or "")
        if not task_id:
            continue
        try:
            payload = svc.get_history_task(task_id)
        except (FileNotFoundError, ValueError):
            continue
        task = payload.get("task") if isinstance(payload, dict) else {}
        if isinstance(task, dict) and task.get("job") == "training":
            tasks.append(task)
    if not tasks:
        raise FileNotFoundError("这个训练分组没有可读取的训练任务")
    return tasks


def _task_config_group_matches(task: dict, group: dict[str, str]) -> bool:
    return (
        str(task.get("methods_subdir") or "").strip() == group["methods_subdir"]
        and str(task.get("variant") or "").strip() == group["variant"]
        and (str(task.get("preset") or "default").strip() or "default") == group["preset"]
    )


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
