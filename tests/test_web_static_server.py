from __future__ import annotations

import asyncio
import sys

import pytest
from aiohttp import web

from web.server import index_handler, static_handler
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


@pytest.mark.parametrize(
    "path",
    [
        "style.css",
        "css/00-tokens.css",
        "css/40-weight-analysis.css",
        "css/42-image-test.css",
        "app.js",
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
    monkeypatch.setattr(web_server, "create_app", lambda: app)
    monkeypatch.setattr(
        web_server.web,
        "run_app",
        lambda actual_app, **kwargs: captured.update(app=actual_app, **kwargs),
    )

    web_server.main()

    assert captured["app"] is app
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 20102


def test_web_main_allows_explicit_network_bind(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        sys,
        "argv",
        ["web", "--host", "0.0.0.0", "--port", "20103"],
    )
    monkeypatch.setattr(web_server, "create_app", object)
    monkeypatch.setattr(
        web_server.web,
        "run_app",
        lambda _app, **kwargs: captured.update(**kwargs),
    )

    web_server.main()

    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 20103
