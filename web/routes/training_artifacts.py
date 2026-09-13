import asyncio

from aiohttp import web

from web.services.training.history_artifacts import history_artifact_manifest


async def handle_history_artifact_manifest(request: web.Request) -> web.Response:
    try:
        manifest = await asyncio.to_thread(
            history_artifact_manifest, request.app["training_service"], request.match_info["task_id"],
        )
        return web.json_response(manifest)
    except FileNotFoundError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=404)
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
