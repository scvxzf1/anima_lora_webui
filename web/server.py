"""aiohttp application factory and startup."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from aiohttp import web

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 512 * 1024 * 1024


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
    app = web.Application(client_max_size=MAX_UPLOAD_BYTES)

    app["root"] = ROOT
    app["training_service"] = None  # lazy init on first import
    app["image_test_service"] = None

    from web.routes import setup_routes
    setup_routes(app)

    app.router.add_get("/", index_handler)
    app.router.add_get("/static/{path:.*}", static_handler)

    app.on_startup.append(_on_startup)
    app.on_shutdown.append(_on_shutdown)
    return app


async def _on_startup(app: web.Application) -> None:
    from web.services.image_test_service import ImageTestService
    from web.services.training_service import TrainingService

    svc = TrainingService(app)
    app["image_test_service"] = ImageTestService(app)
    app["training_service"] = svc
    await svc.start_queue_on_startup()


async def _on_shutdown(app: web.Application) -> None:
    image_test_svc = app["image_test_service"]
    if image_test_svc:
        await image_test_svc.shutdown()
    svc = app["training_service"]
    if svc and svc.status == "running":
        await svc.stop()


def main():
    # 最早加载环境变量，确保在任何模块导入前生效
    from library.env import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Anima LoRA Web UI")
    parser.add_argument("--port", type=int, default=20102)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.path.insert(0, str(ROOT))
    app = create_app()
    print(f"Anima LoRA Web UI: http://{args.host}:{args.port}")
    web.run_app(app, host=args.host, port=args.port, print=None)
