"""Read-only model structure lookup for unsaved training drafts."""

import asyncio
from aiohttp import web

from web.services.model_block_inventory import inspect_model_blocks


def setup_model_block_routes(app: web.Application) -> None:
    app.router.add_get("/api/config/model-blocks", handle_model_blocks)


async def handle_model_blocks(request: web.Request) -> web.Response:
    try:
        result = await asyncio.to_thread(
            inspect_model_blocks,
            request.query.get("model_family", "anima"),
            request.query.get("pretrained_model_name_or_path", ""),
        )
        return web.json_response(result)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
