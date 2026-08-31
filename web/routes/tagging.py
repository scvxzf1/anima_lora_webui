"""HTTP routes for the Dragon external-API tagging workbench."""

from __future__ import annotations

from aiohttp import web

from web.services.tagging.client import TaggingApiError
from web.services.tagging import TaggingService

_PREFIXES = ("/api/captioning", "/api/tagging")


def setup_tagging_routes(app: web.Application) -> None:
    """Register the canonical ``captioning`` API and a short ``tagging`` alias."""

    for prefix in _PREFIXES:
        app.router.add_get(f"{prefix}/settings", handle_settings_get)
        app.router.add_put(f"{prefix}/settings", handle_settings_put)
        app.router.add_post(f"{prefix}/test", handle_provider_test)
        app.router.add_get(f"{prefix}/prompt-presets", handle_prompt_presets_get)
        app.router.add_post(f"{prefix}/prompt-presets", handle_prompt_preset_create)
        app.router.add_put(f"{prefix}/prompt-presets/{{preset_id}}", handle_prompt_preset_update)
        app.router.add_delete(f"{prefix}/prompt-presets/{{preset_id}}", handle_prompt_preset_delete)
        app.router.add_get(f"{prefix}/logs", handle_logs_get)
        app.router.add_delete(f"{prefix}/logs", handle_logs_delete)
        app.router.add_get(f"{prefix}/jobs", handle_jobs_get)
        app.router.add_post(f"{prefix}/jobs", handle_job_create)
        app.router.add_get(f"{prefix}/jobs/{{job_id}}", handle_job_get)
        app.router.add_post(f"{prefix}/jobs/{{job_id}}/cancel", handle_job_cancel)
        app.router.add_patch(f"{prefix}/jobs/{{job_id}}/items/{{item_id}}", handle_item_update)
        app.router.add_post(f"{prefix}/jobs/{{job_id}}/commit", handle_job_commit)


def _service(request: web.Request) -> TaggingService | None:
    service = request.app.get("tagging_service")
    return service if isinstance(service, TaggingService) else service


async def handle_settings_get(request: web.Request) -> web.Response:
    service = _service(request)
    if service is None:
        return _unavailable()
    return web.json_response(service.get_settings())


async def handle_settings_put(request: web.Request) -> web.Response:
    service = _service(request)
    if service is None:
        return _unavailable()
    try:
        data = await _json_object(request)
        return web.json_response(service.save_settings(data))
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)


async def handle_provider_test(request: web.Request) -> web.Response:
    service = _service(request)
    if service is None:
        return _unavailable()
    try:
        data = await _json_object(request, allow_empty=True)
        mode = str(data.get("mode") or request.query.get("mode") or "ping").lower()
        if mode not in {"ping", "actual"}:
            raise ValueError("测试模式只能是 ping 或 actual")
        return web.json_response(await service.test_provider(mode))
    except TaggingApiError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=502)
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)


async def handle_prompt_presets_get(request: web.Request) -> web.Response:
    service = _service(request)
    if service is None:
        return _unavailable()
    return web.json_response(service.list_prompt_presets())


async def handle_prompt_preset_create(request: web.Request) -> web.Response:
    service = _service(request)
    if service is None:
        return _unavailable()
    try:
        return web.json_response(service.create_prompt_preset(await _json_object(request)), status=201)
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)


async def handle_prompt_preset_update(request: web.Request) -> web.Response:
    service = _service(request)
    if service is None:
        return _unavailable()
    try:
        return web.json_response(
            service.update_prompt_preset(request.match_info["preset_id"], await _json_object(request))
        )
    except KeyError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=404)
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)


async def handle_prompt_preset_delete(request: web.Request) -> web.Response:
    service = _service(request)
    if service is None:
        return _unavailable()
    try:
        return web.json_response(service.delete_prompt_preset(request.match_info["preset_id"]))
    except KeyError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=404)


async def handle_logs_get(request: web.Request) -> web.Response:
    service = _service(request)
    if service is None:
        return _unavailable()
    return web.json_response(
        service.get_logs(
            after=request.query.get("after", 0),
            limit=request.query.get("limit"),
            job_id=request.query.get("job_id", ""),
        )
    )


async def handle_logs_delete(request: web.Request) -> web.Response:
    service = _service(request)
    if service is None:
        return _unavailable()
    return web.json_response(service.clear_logs())


async def handle_jobs_get(request: web.Request) -> web.Response:
    service = _service(request)
    if service is None:
        return _unavailable()
    return web.json_response(service.list_jobs())


async def handle_job_create(request: web.Request) -> web.Response:
    service = _service(request)
    if service is None:
        return _unavailable()
    try:
        payload = await _json_object(request)
        return web.json_response(await service.create_job(payload), status=202)
    except (ValueError, OSError) as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    except RuntimeError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=409)


async def handle_job_get(request: web.Request) -> web.Response:
    service = _service(request)
    if service is None:
        return _unavailable()
    try:
        return web.json_response(service.get_job(request.match_info["job_id"]))
    except KeyError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=404)


async def handle_job_cancel(request: web.Request) -> web.Response:
    service = _service(request)
    if service is None:
        return _unavailable()
    try:
        return web.json_response(await service.cancel_job(request.match_info["job_id"]))
    except KeyError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=404)
    except RuntimeError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=409)


async def handle_item_update(request: web.Request) -> web.Response:
    service = _service(request)
    if service is None:
        return _unavailable()
    try:
        data = await _json_object(request)
        text = data.get("proposed_caption")
        if not isinstance(text, str):
            raise ValueError("proposed_caption 必须是字符串")
        return web.json_response(
            service.update_item(request.match_info["job_id"], request.match_info["item_id"], text)
        )
    except KeyError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=404)
    except RuntimeError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=409)
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)


async def handle_job_commit(request: web.Request) -> web.Response:
    service = _service(request)
    if service is None:
        return _unavailable()
    try:
        data = await _json_object(request, allow_empty=True)
        item_ids = data.get("item_ids")
        if item_ids is not None and not isinstance(item_ids, list):
            raise ValueError("item_ids 必须是数组")
        return web.json_response(
            await service.commit_job(
                request.match_info["job_id"],
                all_items=bool(data.get("all")),
                item_ids=[str(value) for value in (item_ids or [])],
            )
        )
    except KeyError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=404)
    except RuntimeError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=409)
    except (ValueError, OSError) as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)


async def _json_object(request: web.Request, *, allow_empty: bool = False) -> dict:
    try:
        value = await request.json()
    except (ValueError, TypeError) as exc:
        raise ValueError("请求体必须是 JSON object") from exc
    if value is None and allow_empty:
        return {}
    if not isinstance(value, dict):
        raise ValueError("请求体必须是 JSON object")
    return value


def _unavailable() -> web.Response:
    return web.json_response({"ok": False, "error": "打标服务尚未初始化"}, status=503)
