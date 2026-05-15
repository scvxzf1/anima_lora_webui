"""Config REST API routes."""

from __future__ import annotations

from aiohttp import web

from web.services.config_service import (
    get_config_file_meta,
    get_field_help,
    get_groups,
    list_all_variants,
    list_config_file_groups,
    list_config_files,
    list_methods,
    list_presets,
    list_variants,
    load_merged_config,
    load_raw_file,
    patch_raw_file_values,
    save_raw_file,
)


def setup_config_routes(app: web.Application) -> None:
    app.router.add_get("/api/methods", handle_methods)
    app.router.add_get("/api/methods/{method}/variants", handle_variants)
    app.router.add_get("/api/presets", handle_presets)
    app.router.add_get("/api/config/merged", handle_merged)
    app.router.add_get("/api/config/raw", handle_raw_get)
    app.router.add_put("/api/config/raw", handle_raw_put)
    app.router.add_patch("/api/config/raw", handle_raw_patch)
    app.router.add_post("/api/config/raw/save-as", handle_raw_save_as)
    app.router.add_get("/api/config/files", handle_files)
    app.router.add_get("/api/config/file-groups", handle_file_groups)
    app.router.add_get("/api/config/field-help", handle_field_help)
    app.router.add_get("/api/config/groups", handle_groups)


async def handle_methods(request: web.Request) -> web.Response:
    return web.json_response(list_methods())


async def handle_variants(request: web.Request) -> web.Response:
    method = request.match_info["method"]
    return web.json_response(list_variants(method))


async def handle_presets(request: web.Request) -> web.Response:
    return web.json_response(list_presets())


async def handle_merged(request: web.Request) -> web.Response:
    variant = request.query.get("variant", "lora")
    preset = request.query.get("preset", "default")
    methods_subdir = request.query.get("methods_subdir", "gui-methods")
    try:
        config = load_merged_config(variant, preset, methods_subdir)
        return web.json_response(config)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)


async def handle_raw_get(request: web.Request) -> web.Response:
    file_path = request.query.get("file", "")
    if not file_path:
        return web.json_response({"error": "缺少 file 参数"}, status=400)
    content = load_raw_file(file_path)
    return web.json_response({"file": file_path, "content": content, "meta": get_config_file_meta(file_path)})


async def handle_raw_put(request: web.Request) -> web.Response:
    data = await request.json()
    file_path = data.get("file", "")
    content = data.get("content", "")
    if not file_path:
        return web.json_response({"error": "缺少 file 参数"}, status=400)
    ok, msg = save_raw_file(file_path, content)
    if ok:
        return web.json_response({"ok": True, "message": msg})
    return web.json_response({"ok": False, "error": msg}, status=400)


async def handle_raw_patch(request: web.Request) -> web.Response:
    data = await request.json()
    file_path = data.get("file", "")
    values = data.get("values", {})
    content = data.get("content")
    if not file_path:
        return web.json_response({"error": "缺少 file 参数"}, status=400)
    ok, msg, next_content, changed = patch_raw_file_values(file_path, values, content=content)
    if ok:
        return web.json_response({
            "ok": True,
            "file": file_path,
            "message": msg,
            "content": next_content,
            "changed": changed,
        })
    return web.json_response({"ok": False, "error": msg}, status=400)


async def handle_raw_save_as(request: web.Request) -> web.Response:
    data = await request.json()
    file_path = data.get("file", "")
    content = data.get("content", "")
    if not file_path:
        return web.json_response({"error": "缺少 file 参数"}, status=400)
    ok, msg = save_raw_file(file_path, content)
    if ok:
        return web.json_response({"ok": True, "file": file_path, "message": msg})
    return web.json_response({"ok": False, "error": msg}, status=400)


async def handle_files(request: web.Request) -> web.Response:
    return web.json_response(list_config_files())


async def handle_file_groups(request: web.Request) -> web.Response:
    return web.json_response(list_config_file_groups())


async def handle_field_help(request: web.Request) -> web.Response:
    return web.json_response(get_field_help())


async def handle_groups(request: web.Request) -> web.Response:
    return web.json_response(get_groups())
