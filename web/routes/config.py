"""Config REST API routes."""

from __future__ import annotations

from aiohttp import web

from web.services.config_service import (
    apply_dataset_preset_to_training_config,
    create_config_file_group,
    delete_dataset_preset,
    delete_raw_file,
    delete_config_file_group,
    estimate_training_steps,
    get_config_file_meta,
    get_field_help,
    get_groups,
    load_dataset_editor,
    load_sample_prompts_file,
    list_all_variants,
    list_config_file_groups,
    list_config_files,
    list_dataset_presets,
    list_dataset_preset_images,
    list_methods,
    list_presets,
    list_variants,
    load_merged_config,
    load_dataset_preset,
    load_raw_file,
    patch_raw_file_values,
    restore_system_presets,
    resolve_dataset_preview_image,
    save_raw_file,
    save_dataset_editor,
    save_dataset_preset,
    save_dataset_preset_as,
    save_sample_prompts_file,
    set_user_file_lock,
    set_user_group_lock,
    move_config_file_to_group,
    rename_config_file_group,
    reorder_config_file_group,
    reorder_config_file_in_group,
    suggest_data_dirs,
    suggest_dataset_dirs,
)


def setup_config_routes(app: web.Application) -> None:
    app.router.add_get("/api/methods", handle_methods)
    app.router.add_get("/api/methods/{method}/variants", handle_variants)
    app.router.add_get("/api/presets", handle_presets)
    app.router.add_get("/api/config/merged", handle_merged)
    app.router.add_get("/api/config/steps", handle_steps)
    app.router.add_get("/api/config/data-dirs/suggest", handle_data_dirs_suggest)
    app.router.add_get("/api/config/datasets", handle_datasets_get)
    app.router.add_put("/api/config/datasets", handle_datasets_put)
    app.router.add_post("/api/config/datasets/suggest", handle_datasets_suggest)
    app.router.add_get("/api/config/dataset-presets", handle_dataset_presets_list)
    app.router.add_get("/api/config/dataset-presets/read", handle_dataset_preset_read)
    app.router.add_put("/api/config/dataset-presets", handle_dataset_preset_put)
    app.router.add_post("/api/config/dataset-presets/save-as", handle_dataset_preset_save_as)
    app.router.add_delete("/api/config/dataset-presets", handle_dataset_preset_delete)
    app.router.add_post("/api/config/dataset-presets/apply", handle_dataset_preset_apply)
    app.router.add_get("/api/config/dataset-presets/images", handle_dataset_preset_images)
    app.router.add_get("/api/config/dataset-presets/image", handle_dataset_preset_image)
    app.router.add_get("/api/config/raw", handle_raw_get)
    app.router.add_put("/api/config/raw", handle_raw_put)
    app.router.add_patch("/api/config/raw", handle_raw_patch)
    app.router.add_delete("/api/config/raw", handle_raw_delete)
    app.router.add_post("/api/config/raw/save-as", handle_raw_save_as)
    app.router.add_get("/api/config/sample-prompts", handle_sample_prompts_get)
    app.router.add_put("/api/config/sample-prompts", handle_sample_prompts_put)
    app.router.add_post("/api/config/lock", handle_config_lock)
    app.router.add_post("/api/config/group-lock", handle_config_group_lock)
    app.router.add_post("/api/config/file-groups", handle_file_group_create)
    app.router.add_patch("/api/config/file-groups/{group_id}", handle_file_group_update)
    app.router.add_delete("/api/config/file-groups/{group_id}", handle_file_group_delete)
    app.router.add_post("/api/config/file-groups/move-file", handle_file_group_move_file)
    app.router.add_post("/api/config/file-groups/reorder-file", handle_file_group_reorder_file)
    app.router.add_post("/api/config/file-groups/reorder-group", handle_file_group_reorder_group)
    app.router.add_post("/api/config/restore-system", handle_restore_system)
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


async def handle_steps(request: web.Request) -> web.Response:
    variant = request.query.get("variant", "lora")
    preset = request.query.get("preset", "default")
    methods_subdir = request.query.get("methods_subdir", "gui-methods")
    dataset_config = request.query.get("dataset_config")
    try:
        return web.json_response(estimate_training_steps(variant, preset, methods_subdir, dataset_config=dataset_config))
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_data_dirs_suggest(request: web.Request) -> web.Response:
    source_image_dir = request.query.get("source_image_dir", "")
    try:
        result = suggest_data_dirs(source_image_dir)
        status = 200 if result.get("ok") else 400
        return web.json_response(result, status=status)
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_datasets_get(request: web.Request) -> web.Response:
    variant = request.query.get("variant", "lora")
    preset = request.query.get("preset", "default")
    methods_subdir = request.query.get("methods_subdir", "gui-methods")
    try:
        return web.json_response(load_dataset_editor(variant, preset, methods_subdir))
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_datasets_put(request: web.Request) -> web.Response:
    data = await request.json()
    variant = data.get("variant", "lora")
    preset = data.get("preset", "default")
    methods_subdir = data.get("methods_subdir", "gui-methods")
    datasets = data.get("datasets", [])
    defaults = data.get("defaults", {})
    train_file = data.get("train_file")
    train_content = data.get("train_content")
    prefer_existing_dataset_config = data.get("prefer_existing_dataset_config", True)
    if not isinstance(datasets, list):
        return web.json_response({"ok": False, "error": "datasets 必须是数组"}, status=400)
    try:
        return web.json_response(save_dataset_editor(
            variant,
            preset,
            methods_subdir,
            datasets,
            defaults=defaults if isinstance(defaults, dict) else {},
            train_file=train_file,
            train_content=train_content,
            prefer_existing_dataset_config=bool(prefer_existing_dataset_config),
        ))
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_datasets_suggest(request: web.Request) -> web.Response:
    data = await request.json()
    source_dirs = data.get("source_dirs", [])
    if not isinstance(source_dirs, list):
        return web.json_response({"ok": False, "error": "source_dirs 必须是数组"}, status=400)
    try:
        result = suggest_dataset_dirs([str(item or "") for item in source_dirs])
        status = 200 if result.get("ok") else 400
        return web.json_response(result, status=status)
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_dataset_presets_list(request: web.Request) -> web.Response:
    try:
        return web.json_response(list_dataset_presets())
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_dataset_preset_read(request: web.Request) -> web.Response:
    file = request.query.get("file", "")
    try:
        return web.json_response(load_dataset_preset(file))
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_dataset_preset_put(request: web.Request) -> web.Response:
    data = await request.json()
    datasets = data.get("datasets", [])
    defaults = data.get("defaults", {})
    if not isinstance(datasets, list):
        return web.json_response({"ok": False, "error": "datasets 必须是数组"}, status=400)
    try:
        return web.json_response(save_dataset_preset(
            str(data.get("file") or ""),
            datasets,
            defaults if isinstance(defaults, dict) else {},
        ))
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_dataset_preset_save_as(request: web.Request) -> web.Response:
    data = await request.json()
    datasets = data.get("datasets", [])
    defaults = data.get("defaults", {})
    if not isinstance(datasets, list):
        return web.json_response({"ok": False, "error": "datasets 必须是数组"}, status=400)
    try:
        return web.json_response(save_dataset_preset_as(
            str(data.get("name") or data.get("file") or ""),
            datasets,
            defaults if isinstance(defaults, dict) else {},
        ))
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_dataset_preset_delete(request: web.Request) -> web.Response:
    file = request.query.get("file", "")
    try:
        return web.json_response(delete_dataset_preset(file))
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_dataset_preset_apply(request: web.Request) -> web.Response:
    data = await request.json()
    try:
        return web.json_response(apply_dataset_preset_to_training_config(
            str(data.get("dataset_file") or data.get("file") or ""),
            str(data.get("train_file") or ""),
            data.get("train_content"),
        ))
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_dataset_preset_images(request: web.Request) -> web.Response:
    file = request.query.get("file", "")
    source = request.query.get("source", "training")
    try:
        dataset_index = int(request.query.get("dataset_index", "0") or 0)
        limit = int(request.query.get("limit", "120") or 120)
        return web.json_response(list_dataset_preset_images(
            file,
            dataset_index,
            source=source,
            limit=limit,
        ))
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_dataset_preset_image(request: web.Request) -> web.StreamResponse:
    file = request.query.get("file", "")
    image = request.query.get("image", "")
    source = request.query.get("source", "training")
    try:
        dataset_index = int(request.query.get("dataset_index", "0") or 0)
        path = resolve_dataset_preview_image(file, dataset_index, image, source=source)
    except FileNotFoundError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=404)
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=403)
    return web.FileResponse(path)


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


async def handle_raw_delete(request: web.Request) -> web.Response:
    file_path = request.query.get("file", "")
    if not file_path:
        try:
            data = await request.json()
        except Exception:
            data = {}
        file_path = data.get("file", "")
    if not file_path:
        return web.json_response({"ok": False, "error": "缺少 file 参数"}, status=400)

    ok, msg = delete_raw_file(file_path)
    if ok:
        return web.json_response({"ok": True, "file": file_path, "message": msg})
    return web.json_response({"ok": False, "error": msg}, status=400)


async def handle_raw_save_as(request: web.Request) -> web.Response:
    data = await request.json()
    file_path = data.get("file", "")
    content = data.get("content", "")
    if not file_path:
        return web.json_response({"error": "缺少 file 参数"}, status=400)
    ok, msg = save_raw_file(file_path, content, overwrite=False)
    if ok:
        return web.json_response({"ok": True, "file": file_path, "message": msg})
    return web.json_response({"ok": False, "error": msg}, status=400)


async def handle_sample_prompts_get(request: web.Request) -> web.Response:
    file_path = request.query.get("file", "")
    try:
        return web.json_response(load_sample_prompts_file(file_path or None))
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_sample_prompts_put(request: web.Request) -> web.Response:
    data = await request.json()
    file_path = data.get("file", "")
    content = data.get("content", "")
    try:
        return web.json_response(save_sample_prompts_file(content, file_path or None))
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_config_lock(request: web.Request) -> web.Response:
    data = await request.json()
    file_path = data.get("file", "")
    locked = bool(data.get("locked", False))
    if not file_path:
        return web.json_response({"error": "缺少 file 参数"}, status=400)
    ok, msg, meta = set_user_file_lock(file_path, locked)
    if ok:
        return web.json_response({"ok": True, "file": file_path, "locked": locked, "message": msg, "meta": meta})
    return web.json_response({"ok": False, "error": msg, "meta": meta}, status=400)


async def handle_config_group_lock(request: web.Request) -> web.Response:
    data = await request.json()
    group_id = data.get("group", "")
    locked = bool(data.get("locked", False))
    if not group_id:
        return web.json_response({"error": "缺少 group 参数"}, status=400)
    ok, msg, group = set_user_group_lock(group_id, locked)
    if ok:
        return web.json_response({"ok": True, "group": group_id, "locked": locked, "message": msg, "meta": group})
    return web.json_response({"ok": False, "error": msg, "meta": group}, status=400)


async def handle_file_group_create(request: web.Request) -> web.Response:
    data = await request.json()
    ok, msg, group = create_config_file_group(data.get("label", ""))
    if ok:
        return web.json_response({"ok": True, "message": msg, "group": group})
    return web.json_response({"ok": False, "error": msg, "group": group}, status=400)


async def handle_file_group_update(request: web.Request) -> web.Response:
    group_id = request.match_info["group_id"]
    data = await request.json()
    ok, msg, group = rename_config_file_group(group_id, data.get("label", ""))
    if ok:
        return web.json_response({"ok": True, "message": msg, "group": group})
    return web.json_response({"ok": False, "error": msg, "group": group}, status=400)


async def handle_file_group_delete(request: web.Request) -> web.Response:
    group_id = request.match_info["group_id"]
    ok, msg = delete_config_file_group(group_id)
    if ok:
        return web.json_response({"ok": True, "message": msg})
    return web.json_response({"ok": False, "error": msg}, status=400)


async def handle_file_group_move_file(request: web.Request) -> web.Response:
    data = await request.json()
    ok, msg, group = move_config_file_to_group(data.get("file", ""), data.get("group", ""))
    if ok:
        return web.json_response({"ok": True, "message": msg, "group": group})
    return web.json_response({"ok": False, "error": msg, "group": group}, status=400)


async def handle_file_group_reorder_file(request: web.Request) -> web.Response:
    data = await request.json()
    ok, msg, group = reorder_config_file_in_group(
        data.get("file", ""),
        data.get("group", ""),
        data.get("direction", ""),
    )
    if ok:
        return web.json_response({"ok": True, "message": msg, "group": group})
    return web.json_response({"ok": False, "error": msg, "group": group}, status=400)


async def handle_file_group_reorder_group(request: web.Request) -> web.Response:
    data = await request.json()
    ok, msg, group = reorder_config_file_group(
        data.get("group", ""),
        data.get("direction", ""),
    )
    if ok:
        return web.json_response({"ok": True, "message": msg, "group": group})
    return web.json_response({"ok": False, "error": msg, "group": group}, status=400)


async def handle_restore_system(request: web.Request) -> web.Response:
    data = await request.json()
    files = data.get("files")
    single_file = data.get("file")
    if isinstance(single_file, str) and single_file:
        files = [single_file]
    elif files is not None and not isinstance(files, list):
        return web.json_response({"ok": False, "error": "files 必须是数组"}, status=400)

    result = restore_system_presets(files)
    status = 200 if result.get("ok") else 400
    return web.json_response(result, status=status)


async def handle_files(request: web.Request) -> web.Response:
    return web.json_response(list_config_files())


async def handle_file_groups(request: web.Request) -> web.Response:
    return web.json_response(list_config_file_groups())


async def handle_field_help(request: web.Request) -> web.Response:
    return web.json_response(get_field_help())


async def handle_groups(request: web.Request) -> web.Response:
    return web.json_response(get_groups())
