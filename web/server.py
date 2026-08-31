"""aiohttp application factory and startup."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from aiohttp import web
from aiohttp.web_log import AccessLogger

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
AUTH_COOKIE = "anima_webui_token"
AUTH_HEADER = "X-Anima-Token"
TOKEN_ENV = "ANIMA_WEBUI_TOKEN"


class _WebAccessLogger(AccessLogger):
    """Log failed HTTP requests without printing successful request traffic."""

    def log(self, request, response, time) -> None:
        if response is not None and response.status < 400:
            return
        super().log(request, response, time)


async def index_handler(request: web.Request) -> web.FileResponse:
    response = web.FileResponse(STATIC_DIR / "index.html")
    response.headers["Cache-Control"] = "no-cache"
    return response


async def next_index_handler(request: web.Request) -> web.FileResponse:
    """Serve the isolated next-generation frontend during migration."""
    path = STATIC_DIR / "dragon-next" / "index.html"
    if not path.is_file():
        raise web.HTTPNotFound(
            text=(
                "Dragon next frontend has not been built. Run "
                "`pnpm --dir web/frontend-next build`."
            )
        )
    response = web.FileResponse(path)
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


def _is_loopback_bind(host: str) -> bool:
    """True when the bind host can only be reached from this machine."""
    import ipaddress

    value = (host or "").strip().lower()
    if value in {"", "127.0.0.1", "::1", "localhost"}:
        return True
    # Strip IPv6 brackets used by some CLIs ("[::1]").
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        # Hostnames and wildcard binds (0.0.0.0 / ::) are externally reachable.
        return False


def _extract_request_token(request: web.Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
        if token:
            return token
    header = request.headers.get(AUTH_HEADER, "").strip()
    if header:
        return header
    cookie = request.cookies.get(AUTH_COOKIE, "").strip()
    if cookie:
        return cookie
    query = request.rel_url.query.get("token", "").strip()
    if query:
        return query
    return None


@web.middleware
async def auth_middleware(request: web.Request, handler):
    """Require a shared token when the server is bound beyond loopback.

    Loopback binds stay open for local-only use. External binds must pass a
    token via ``Authorization: Bearer``, ``X-Anima-Token``, cookie, or
    ``?token=`` (the query form sets a session cookie for subsequent browser
    navigations).
    """
    expected = request.app.get("auth_token")
    if not expected:
        return await handler(request)

    provided = _extract_request_token(request)
    if provided != expected:
        raise web.HTTPUnauthorized(
            text=(
                "Unauthorized: provide ANIMA_WEBUI_TOKEN via Authorization Bearer, "
                f"{AUTH_HEADER}, cookie {AUTH_COOKIE}, or ?token="
            ),
            headers={"WWW-Authenticate": 'Bearer realm="Dragon trainer WebUI"'},
        )

    response = await handler(request)
    # Bootstrap browser sessions from ?token= without forcing every static
    # asset request to carry the query string.
    if request.rel_url.query.get("token", "").strip() == expected:
        response.set_cookie(
            AUTH_COOKIE,
            expected,
            httponly=True,
            samesite="Lax",
            path="/",
        )
    return response


def create_app_with_options(*, auth_token: str | None = None) -> web.Application:
    middlewares = []
    token = (auth_token or "").strip() or None
    if token:
        middlewares.append(auth_middleware)

    app = web.Application(client_max_size=MAX_UPLOAD_BYTES, middlewares=middlewares)
    app["auth_token"] = token

    app["root"] = ROOT
    app["training_service"] = None  # lazy init on first import
    app["image_test_service"] = None
    app["tagging_service"] = None

    from web.routes import setup_routes

    setup_routes(app)

    app.router.add_get("/", index_handler)
    app.router.add_get("/next", next_index_handler)
    app.router.add_get("/next/{path:.*}", next_index_handler)
    app.router.add_get("/static/{path:.*}", static_handler)

    app.on_startup.append(_on_startup)
    app.on_shutdown.append(_on_shutdown)
    return app


def create_app() -> web.Application:
    """Backward-compatible factory used by tests and local loopback starts."""
    return create_app_with_options()


async def _on_startup(app: web.Application) -> None:
    from web.services.image_test_service import ImageTestService
    from web.services.tagging import TaggingService
    from web.services.training_service import TrainingService

    svc = TrainingService(app)
    app["image_test_service"] = ImageTestService(app)
    app["tagging_service"] = TaggingService(app)
    app["training_service"] = svc
    await svc.start_queue_on_startup()


async def _on_shutdown(app: web.Application) -> None:
    tagging_svc = app.get("tagging_service")
    if tagging_svc:
        await tagging_svc.shutdown()
    image_test_svc = app["image_test_service"]
    if image_test_svc:
        await image_test_svc.shutdown()
    svc = app["training_service"]
    if svc:
        await svc.shutdown()


def main():
    # 最早加载环境变量，确保在任何模块导入前生效
    from library.env import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Dragon trainer")
    parser.add_argument("--port", type=int, default=20102)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "Shared access token required when binding beyond loopback. "
            f"Falls back to ${TOKEN_ENV}. Ignored for 127.0.0.1/::1."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.path.insert(0, str(ROOT))

    token = (args.token or "").strip() or None
    if token is None:
        token = (os.environ.get(TOKEN_ENV) or "").strip() or None

    external = not _is_loopback_bind(args.host)
    if external and not token:
        print(
            f"Refusing to bind WebUI on non-loopback host {args.host!r} without auth.\n"
            f"Pass --token <secret> or set {TOKEN_ENV}, "
            "or bind to 127.0.0.1 for local-only access.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Loopback binds do not enforce auth even if a token is present — keeps the
    # default local workflow unchanged. External binds always enforce.
    app = create_app_with_options(auth_token=token if external else None)
    print(f"Dragon trainer: http://{args.host}:{args.port}")
    if external:
        print(
            "Auth required for external bind "
            f"(Authorization: Bearer / {AUTH_HEADER} / ?token=)."
        )
    # Successful browser traffic is noisy; HTTP failures remain visible while
    # application, training, warning, and error loggers are unaffected.
    web.run_app(
        app,
        host=args.host,
        port=args.port,
        print=None,
        access_log_class=_WebAccessLogger,
    )
