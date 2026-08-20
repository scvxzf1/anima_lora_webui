from __future__ import annotations

import asyncio
import logging
import sys
from types import SimpleNamespace

import pytest
from aiohttp import web

from web.server import index_handler, next_index_handler, static_handler
from web import server as web_server


class _StaticRequest:
    def __init__(self, path: str = "") -> None:
        self.match_info = {"path": path}


def _run(coro):
    return asyncio.run(coro)


def test_web_index_serves_versioned_frontend_entrypoint() -> None:
    response = _run(index_handler(_StaticRequest()))

    assert response.status == 200
    assert response.headers["Cache-Control"] == "no-cache"
    assert response._path.name == "index.html"


def test_next_frontend_reports_missing_build(monkeypatch) -> None:
    monkeypatch.setattr(web_server, "STATIC_DIR", web_server.STATIC_DIR / "missing-next")

    with pytest.raises(web.HTTPNotFound) as exc_info:
        _run(next_index_handler(_StaticRequest()))

    assert "frontend has not been built" in exc_info.value.text


def test_next_frontend_handler_serves_spa_nested_paths() -> None:
    response = _run(next_index_handler(_StaticRequest("datasets")))

    assert response.status == 200
    assert response.headers["Cache-Control"] == "no-cache"
    assert response._path.name == "index.html"


@pytest.mark.parametrize(
    "path",
    [
        "style.css",
        "css/00-tokens.css",
        "css/40-weight-analysis.css",
        "css/42-image-test.css",
        "css/dragon-style.css",
        "app.js",
        "js/ui-bootstrap.js",
        "js/dragon-ui/index.js",
        "js/features/app-shell/tabs.js",
        "js/features/sample-prompts/model.js",
        "js/features/toml-manager/group-state.js",
    ],
)
def test_web_static_serves_split_frontend_assets(path: str) -> None:
    response = _run(static_handler(_StaticRequest(path)))

    assert response.status == 200
    assert response.headers["Cache-Control"] == "no-cache"
    assert response._path.is_file()


@pytest.mark.parametrize("path", ["missing.css", "../server.py", "css/../../server.py"])
def test_web_static_rejects_missing_or_escaping_paths(path: str) -> None:
    with pytest.raises((web.HTTPForbidden, web.HTTPNotFound)):
        _run(static_handler(_StaticRequest(path)))


def test_web_main_defaults_to_loopback(monkeypatch) -> None:
    captured = {}
    app = object()
    monkeypatch.setattr(sys, "argv", ["web"])
    monkeypatch.setattr(
        web_server,
        "create_app_with_options",
        lambda *, auth_token=None: app,
    )
    monkeypatch.setattr(
        web_server.web,
        "run_app",
        lambda actual_app, **kwargs: captured.update(app=actual_app, **kwargs),
    )

    web_server.main()

    assert captured["app"] is app
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 20102
    assert captured["access_log_class"] is web_server._WebAccessLogger


def test_web_access_logger_suppresses_successful_requests_only(monkeypatch) -> None:
    logged = []
    monkeypatch.setattr(
        web_server.AccessLogger,
        "log",
        lambda _self, request, response, _time: logged.append(
            (request.path, response.status)
        ),
    )
    access_logger = web_server._WebAccessLogger(logging.getLogger("test.web.access"))

    access_logger.log(
        SimpleNamespace(path="/ws/training"), SimpleNamespace(status=101), 0.0
    )
    access_logger.log(
        SimpleNamespace(path="/static/app.js"), SimpleNamespace(status=304), 0.0
    )
    access_logger.log(
        SimpleNamespace(path="/api/training/history"),
        SimpleNamespace(status=200),
        0.0,
    )
    access_logger.log(
        SimpleNamespace(path="/api/missing"), SimpleNamespace(status=404), 0.0
    )
    access_logger.log(
        SimpleNamespace(path="/api/training/start"),
        SimpleNamespace(status=500),
        0.0,
    )

    assert logged == [("/api/missing", 404), ("/api/training/start", 500)]


def test_web_main_allows_explicit_network_bind(monkeypatch) -> None:
    captured = {}
    created = {}
    monkeypatch.setattr(
        sys,
        "argv",
        ["web", "--host", "0.0.0.0", "--port", "20103", "--token", "s3cret"],
    )
    monkeypatch.delenv(web_server.TOKEN_ENV, raising=False)

    def _create_app_with_options(*, auth_token=None):
        created["auth_token"] = auth_token
        return object()

    monkeypatch.setattr(web_server, "create_app_with_options", _create_app_with_options)
    monkeypatch.setattr(
        web_server.web,
        "run_app",
        lambda _app, **kwargs: captured.update(**kwargs),
    )

    web_server.main()

    assert created["auth_token"] == "s3cret"
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 20103


def test_web_main_refuses_external_bind_without_token(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["web", "--host", "0.0.0.0", "--port", "20103"])
    monkeypatch.delenv(web_server.TOKEN_ENV, raising=False)
    ran = {"called": False}
    monkeypatch.setattr(
        web_server.web,
        "run_app",
        lambda *_a, **_k: ran.__setitem__("called", True),
    )

    with pytest.raises(SystemExit) as exc:
        web_server.main()

    assert exc.value.code == 2
    assert ran["called"] is False


def test_web_main_external_bind_requires_token_from_cli(monkeypatch) -> None:
    captured = {}
    created = {}
    monkeypatch.setattr(
        sys,
        "argv",
        ["web", "--host", "0.0.0.0", "--port", "20103", "--token", "s3cret"],
    )
    monkeypatch.delenv(web_server.TOKEN_ENV, raising=False)

    def _create_app_with_options(*, auth_token=None):
        created["auth_token"] = auth_token
        return object()

    monkeypatch.setattr(web_server, "create_app_with_options", _create_app_with_options)
    monkeypatch.setattr(
        web_server.web,
        "run_app",
        lambda _app, **kwargs: captured.update(**kwargs),
    )

    web_server.main()

    assert created["auth_token"] == "s3cret"
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 20103


def test_web_main_loopback_ignores_token_enforcement(monkeypatch) -> None:
    created = {}
    monkeypatch.setattr(sys, "argv", ["web", "--host", "127.0.0.1", "--token", "s3cret"])
    monkeypatch.setattr(
        web_server,
        "create_app_with_options",
        lambda *, auth_token=None: created.__setitem__("auth_token", auth_token) or object(),
    )
    monkeypatch.setattr(web_server.web, "run_app", lambda *_a, **_k: None)

    web_server.main()

    assert created["auth_token"] is None


def test_auth_middleware_accepts_bearer_and_rejects_missing() -> None:
    app = web_server.create_app_with_options(auth_token="s3cret")

    async def _ok(_request):
        return web.Response(text="ok")

    app.router.add_get("/ping", _ok)

    async def _exercise():
        from aiohttp.test_utils import TestClient, TestServer

        async with TestServer(app) as server:
            async with TestClient(server) as client:
                denied = await client.get("/ping")
                assert denied.status == 401
                allowed = await client.get(
                    "/ping", headers={"Authorization": "Bearer s3cret"}
                )
                assert allowed.status == 200
                assert await allowed.text() == "ok"
                cookie = await client.get("/ping?token=s3cret")
                assert cookie.status == 200
                assert web_server.AUTH_COOKIE in cookie.cookies

    _run(_exercise())


def test_is_loopback_bind_helpers() -> None:
    assert web_server._is_loopback_bind("127.0.0.1")
    assert web_server._is_loopback_bind("::1")
    assert web_server._is_loopback_bind("localhost")
    assert not web_server._is_loopback_bind("0.0.0.0")
    assert not web_server._is_loopback_bind("192.168.1.10")
