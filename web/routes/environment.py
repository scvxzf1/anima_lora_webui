"""Environment integrity routes."""

from __future__ import annotations

import asyncio

from aiohttp import web

from web.services.environment_check_service import run_environment_check


def setup_environment_routes(app: web.Application) -> None:
    app.router.add_get("/api/environment/check", handle_environment_check)


async def handle_environment_check(request: web.Request) -> web.Response:
    del request
    return web.json_response(await asyncio.to_thread(run_environment_check))
