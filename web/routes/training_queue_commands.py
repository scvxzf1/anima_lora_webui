"""Optional snapshot guards preserve the bodyless legacy queue commands."""

from aiohttp import web

from web.services.training.queue_revision import QueueRevisionConflict


async def queue_command_response(request: web.Request, method: str) -> web.Response:
    kwargs = {}
    if getattr(request, "can_read_body", False):
        try:
            body = await request.json()
        except (ValueError, TypeError):
            return web.json_response({"ok": False, "error": "请求正文必须是有效 JSON"}, status=400)
        revision = body.get("expected_revision") if isinstance(body, dict) else None
        if not isinstance(revision, str) or not revision.strip():
            return web.json_response({"ok": False, "error": "缺少有效的队列快照版本"}, status=400)
        kwargs["expected_revision"] = revision
    service = request.app["training_service"]
    try:
        return web.json_response(await getattr(service, method)(**kwargs))
    except QueueRevisionConflict as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=409)
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
