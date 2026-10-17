from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import toml
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from web.routes.settings import setup_settings_routes
from web.services import settings_paths, settings_service


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ANIMA_HOME", str(tmp_path))
    monkeypatch.setattr(settings_service, "SETTINGS_FILE", tmp_path / "settings.toml")
    return tmp_path


def _model_root(home: Path) -> Path:
    root = home / "models"
    root.mkdir()
    return root


def test_model_path_status_kinds(home: Path) -> None:
    root = _model_root(home)
    weight = root / "dit.safetensors"
    weight.write_bytes(b"abc")
    assert settings_paths.check_model_path(str(weight)) == {
        "path": str(weight), "kind": "file", "size_bytes": 3
    }
    assert settings_paths.check_model_path(str(root))["kind"] == "directory"
    assert settings_paths.check_model_path(str(root / "missing.pt"))["kind"] == "missing"
    assert settings_paths.check_model_path("") == {"path": "", "kind": "empty"}


def test_model_path_status_rejects_traversal_and_outside(home: Path) -> None:
    root = _model_root(home)
    outside = home / "outside"
    outside.mkdir()
    with pytest.raises(PermissionError):
        settings_paths.check_model_path(str(outside))
    with pytest.raises(ValueError):
        settings_paths.check_model_path(str(root / ".." / "outside"))


def test_directory_listing_and_pagination(home: Path) -> None:
    (home / "configs").mkdir()
    root = home / "output" / "runs"
    root.mkdir(parents=True)
    for name in ("z", "a", ".hidden"):
        (root / name).mkdir()
    payload = settings_paths.list_directories(str(root), limit=1)
    assert payload["entries"] == [{"name": "a", "path": str(root / "a"), "kind": "directory"}]
    assert payload["next_offset"] == 1
    assert payload["total"] == 2


def test_routes_status_and_directory_errors(home: Path) -> None:
    root = _model_root(home)
    (root / "x.pt").write_bytes(b"12")

    async def run() -> None:
        app = web.Application()
        setup_settings_routes(app)
        async with TestServer(app) as server, TestClient(server) as client:
            response = await client.get("/api/settings/model-path-status", params={"path": str(root / "x.pt")})
            assert response.status == 200
            assert (await response.json())["size_bytes"] == 2
            response = await client.get("/api/settings/model-path-status", params={"path": str(home)})
            assert response.status == 403
            response = await client.get("/api/settings/directories", params={"directory": str(home / "..")})
            assert response.status == 400

    asyncio.run(run())
