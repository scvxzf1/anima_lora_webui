"""Global settings routes."""

from __future__ import annotations

from aiohttp import web

from web.services import config_service, settings_service, training_service
from web.services.model_config_service import (
    ModelConfigConflictError,
    ModelConfigFileError,
    get_model_configs,
    save_model_configs,
)
from web.services.settings_service import get_global_settings, save_global_settings


def setup_settings_routes(app: web.Application) -> None:
    app.router.add_get("/api/settings/global", handle_global_settings_get)
    app.router.add_put("/api/settings/global", handle_global_settings_put)
    app.router.add_get("/api/settings/model-configs", handle_model_configs_get)
    app.router.add_put("/api/settings/model-configs", handle_model_configs_put)


async def handle_global_settings_get(request: web.Request) -> web.Response:
    try:
        return web.json_response(get_global_settings())
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=409)


async def handle_global_settings_put(request: web.Request) -> web.Response:
    data = await request.json()
    try:
        payload = save_global_settings(data)
        tagging_service = request.app.get("tagging_service")
        set_job_retention = getattr(tagging_service, "set_job_retention", None)
        if callable(set_job_retention):
            set_job_retention(payload.get("tagging_max_retained_jobs"))
        if "configs_root" in data:
            config_service.set_configs_root(settings_service.SETTINGS_FILE.parent)
            training_service.reload_runtime_storage_state(request.app.get("training_service"))
        return web.json_response(payload)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)


async def handle_model_configs_get(request: web.Request) -> web.Response:
    try:
        return web.json_response(get_model_configs())
    except (ModelConfigFileError, ValueError) as e:
        return web.json_response({"ok": False, "error": str(e)}, status=409)


async def handle_model_configs_put(request: web.Request) -> web.Response:
    try:
        data = await request.json()
        return web.json_response(save_model_configs(data))
    except (ModelConfigConflictError, ModelConfigFileError) as e:
        return web.json_response({"ok": False, "error": str(e)}, status=409)
    except ValueError as e:
        return web.json_response({"ok": False, "error": str(e)}, status=400)
