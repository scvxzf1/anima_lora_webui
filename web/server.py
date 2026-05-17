"""aiohttp application factory and startup."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aiohttp import web

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"


async def index_handler(request: web.Request) -> web.FileResponse:
    response = web.FileResponse(STATIC_DIR / "index.html")
    response.headers["Cache-Control"] = "no-cache"
    return response


async def static_handler(request: web.Request) -> web.FileResponse:
    rel_path = request.match_info["path"]
    path = (STATIC_DIR / rel_path).resolve()
    if STATIC_DIR not in path.parents and path != STATIC_DIR:
        raise web.HTTPForbidden()
    if not path.is_file():
        raise web.HTTPNotFound()
    response = web.FileResponse(path)
    if path.suffix in {".js", ".css", ".html"}:
        response.headers["Cache-Control"] = "no-cache"
    return response


def create_app() -> web.Application:
    app = web.Application()

    app["root"] = ROOT
    app["training_service"] = None  # lazy init on first import

    from web.routes import setup_routes
    setup_routes(app)

    app.router.add_get("/", index_handler)
    app.router.add_get("/static/{path:.*}", static_handler)

    app.on_startup.append(_on_startup)
    app.on_shutdown.append(_on_shutdown)
    return app


async def _on_startup(app: web.Application) -> None:
    from web.services.training_service import TrainingService
    svc = TrainingService(app)
    app["training_service"] = svc


async def _on_shutdown(app: web.Application) -> None:
    svc = app["training_service"]
    if svc and svc.status == "running":
        await svc.stop()


def main():
    parser = argparse.ArgumentParser(description="Anima LoRA Web UI")
    parser.add_argument("--port", type=int, default=20102)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    app = create_app()
    print(f"Anima LoRA Web UI: http://{args.host}:{args.port}")
    web.run_app(app, host=args.host, port=args.port, print=None)
