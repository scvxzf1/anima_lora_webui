"""Read-only GPU history endpoint."""
import asyncio
from aiohttp import web


async def history_system_response(request: web.Request) -> web.Response:
    try:
        offset = int(request.query.get("offset", "0"))
        limit = int(request.query.get("limit", "2000"))
        payload = await asyncio.to_thread(request.app["training_service"].get_history_system_page,
                                         request.match_info["task_id"], offset=offset, limit=limit)
        return web.json_response(payload)
    except FileNotFoundError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=404)
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
