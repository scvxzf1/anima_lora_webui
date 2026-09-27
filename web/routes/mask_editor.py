"""Bounded HTTP entry points for the manual dataset mask editor."""

import asyncio

from aiohttp import web
from PIL import Image

from web.services.config import mask_editor as service


def setup_mask_editor_routes(app: web.Application) -> None:
    prefix = "/api/config/dataset-masks"
    app.router.add_get(prefix, handle_list)
    app.router.add_get(prefix + "/image", handle_read)
    app.router.add_put(prefix + "/image", handle_save)
    app.router.add_post(prefix + "/apply", handle_apply)


def _args(request):
    return request.query.get("file", ""), int(request.query.get("dataset_index", "0"))


def _bad_request(message: str, status: int = 400):
    return web.json_response({"ok": False, "error": message}, status=status)


async def _respond(fn, *args):
    try:
        return web.json_response(await asyncio.to_thread(fn, *args), headers={"Cache-Control": "no-store"})
    except service.MaskConflict as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=409)
    except PermissionError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=403)
    except (ValueError, OSError, Image.DecompressionBombError) as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)


async def handle_list(request):
    try:
        return await _respond(service.list_masks, *_args(request), int(request.query.get("offset", "0")))
    except ValueError:
        return _bad_request("Invalid dataset index or offset")


async def handle_read(request):
    try:
        return await _respond(service.read_mask, *_args(request), request.query.get("image", ""))
    except ValueError:
        return _bad_request("Invalid dataset index")


async def handle_save(request):
    if request.content_type != "image/png":
        return _bad_request("Expected image/png", 415)
    data = bytearray()
    async for chunk in request.content.iter_chunked(65536):
        data.extend(chunk)
        if len(data) > service.MAX_PNG_BYTES:
            return _bad_request("PNG exceeds size limit", 413)
    try:
        return await _respond(service.save_mask, *_args(request), request.query.get("image", ""),
                              request.headers.get("If-Match", ""), bytes(data))
    except ValueError:
        return _bad_request("Invalid dataset index")


async def handle_apply(request):
    if not request.can_read_body:
        data = {}
    else:
        try:
            data = await request.json()
        except (ValueError, web.HTTPBadRequest):
            return _bad_request("请求体必须是有效 JSON 对象")
    if not isinstance(data, dict):
        return _bad_request("请求体必须是 JSON 对象")
    target_indices = data.get("indices")
    if target_indices is not None and (
        not isinstance(target_indices, list)
        or any(type(index) is not int for index in target_indices)
    ):
        return _bad_request("indices 必须是整数数组")
    try:
        return await _respond(
            service.apply_masks,
            *_args(request),
            request.headers.get("If-Match", ""),
            target_indices,
        )
    except ValueError:
        return _bad_request("Invalid dataset index")
