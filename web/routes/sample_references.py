"""Sample reference upload/path import endpoints."""

import asyncio
import logging

from aiohttp import web

from library.training.preview_spec import MAX_REFERENCE_BYTES
from web.services.sample_references import import_reference, resolve_reference_asset


logger = logging.getLogger(__name__)


def setup_sample_reference_routes(app):
    app.router.add_post("/api/config/sample-references", handle_import)
    app.router.add_get("/api/config/sample-references/{key}", handle_image)


async def handle_import(request):
    try:
        if request.content_type == "multipart/form-data":
            reader = await request.multipart()
            part = await reader.next()
            if part is None or part.name != "file":
                raise ValueError("缺少图片文件")
            data = bytearray()
            while chunk := await part.read_chunk():
                data.extend(chunk)
                if len(data) > MAX_REFERENCE_BYTES:
                    raise ValueError("参考图超过 20 MiB")
            result = await asyncio.to_thread(import_reference, data=bytes(data))
        else:
            body = await request.json()
            if not isinstance(body, dict) or not isinstance(body.get("path"), str):
                raise ValueError("请提供参考图 path 字符串")
            result = await asyncio.to_thread(import_reference, path=str(body.get("path") or ""))
        return web.json_response(result)
    except (ValueError, OSError) as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    except Exception:
        logger.exception("failed to import sample reference")
        return web.json_response({"ok": False, "error": "参考图载入失败，请检查图片格式后重试"}, status=500)


async def handle_image(request):
    try:
        path = resolve_reference_asset(request.match_info["key"])
        return web.FileResponse(path, headers={"Cache-Control": "private, max-age=86400"})
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=404)
