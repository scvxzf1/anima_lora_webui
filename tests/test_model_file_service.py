from __future__ import annotations

import asyncio
import errno
from pathlib import Path
import threading

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest
import toml

from web.routes.settings import setup_settings_routes
from web.services import model_file_service, settings_service


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ANIMA_HOME", str(tmp_path))
    monkeypatch.setattr(settings_service, "SETTINGS_FILE", tmp_path / "settings.toml")
    return tmp_path


def _configure(home: Path, **paths: str) -> None:
    (home / "settings.toml").write_text(toml.dumps({"global": paths}), encoding="utf-8")


def test_filter_sort_page_and_parent_contract(home: Path) -> None:
    models = home / "models"
    models.mkdir()
    for name in ("Z-folder", "a-folder", ".private"):
        (models / name).mkdir()
    for name in ("z.safetensors", "A.BIN", "b.pt", "c.pth", "d.ckpt", "e.gguf", "readme.txt", ".secret.pt"):
        (models / name).touch()
    payload = model_file_service.list_model_files(limit=3)
    assert payload == {
        "ok": True,
        "roots": [{"name": "models", "path": str(models)}],
        "directory": str(models),
        "parent": None,
        "entries": [
            {"name": "a-folder", "path": str(models / "a-folder"), "kind": "directory"},
            {"name": "Z-folder", "path": str(models / "Z-folder"), "kind": "directory"},
            {"name": "A.BIN", "path": str(models / "A.BIN"), "kind": "file"},
        ],
        "offset": 0, "next_offset": 3, "total": 8, "truncated": False,
    }
    next_page = model_file_service.list_model_files(str(models), offset=3, limit=200)
    assert [entry["name"] for entry in next_page["entries"]] == ["b.pt", "c.pth", "d.ckpt", "e.gguf", "z.safetensors"]
    assert next_page["next_offset"] is None
    assert model_file_service.list_model_files(str(models / "a-folder"))["parent"] == str(models)


def test_roots_expand_server_settings_and_deduplicate(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    weights = home / "external"
    weights.mkdir()
    (weights / "dit.pt").touch()
    monkeypatch.setenv("MODEL_BROWSER_WEIGHTS", str(weights))
    monkeypatch.setenv("HOME", str(home))
    _configure(home, pretrained_model_name_or_path="$MODEL_BROWSER_WEIGHTS/dit.pt", qwen3="~/external", vae="external/missing.safetensors")
    before = (home / "settings.toml").read_bytes()
    payload = model_file_service.list_model_files()
    assert payload["roots"] == [
        {"name": "models", "path": str(home / "models")},
        {"name": "external", "path": str(weights)},
    ]
    assert payload["directory"] == str(weights)
    assert (home / "settings.toml").read_bytes() == before


def test_missing_roots_return_empty_default_but_explicit_missing_is_404(home: Path) -> None:
    payload = model_file_service.list_model_files()
    assert payload["directory"] == ""
    assert payload["entries"] == []
    with pytest.raises(FileNotFoundError):
        model_file_service.list_model_files(str(home / "models"))


def test_scope_traversal_hidden_and_symlink_boundaries(home: Path) -> None:
    models = home / "models"
    models.mkdir()
    outside = home / "outside"
    outside.mkdir()
    (outside / "secret.pt").touch()
    (models / "escape").symlink_to(outside, target_is_directory=True)
    (models / "escape.pt").symlink_to(outside / "secret.pt")
    (models / ".hidden").mkdir()
    (models / "visible-hidden").symlink_to(models / ".hidden", target_is_directory=True)
    (models / "visible").mkdir()
    (models / ".hidden-link").symlink_to(models / "visible", target_is_directory=True)
    (models / "real.pt").touch()
    (models / "alias.pt").symlink_to(models / "real.pt")
    assert [entry["name"] for entry in model_file_service.list_model_files()["entries"]] == ["visible", "alias.pt", "real.pt"]
    for value in (str(outside), str(models / "escape"), str(models / ".hidden"), str(models / ".hidden-link"), "/"):
        with pytest.raises(PermissionError):
            model_file_service.list_model_files(value)
    with pytest.raises(ValueError, match=r"\.\."):
        model_file_service.list_model_files(str(models / ".." / "outside"))


def test_registered_hidden_root_does_not_return_a_forbidden_parent(home: Path) -> None:
    registered = home / "models" / ".hidden" / "registered"
    registered.mkdir(parents=True)
    _configure(home, qwen3=str(registered))
    assert model_file_service.list_model_files(str(registered))["parent"] is None
    with pytest.raises(PermissionError):
        model_file_service.list_model_files(str(registered.parent))


def test_filesystem_roots_cannot_be_granted_by_settings(home: Path) -> None:
    _configure(home, pretrained_model_name_or_path="/", qwen3="/missing.pt", vae="C:\\")
    assert model_file_service.list_model_files()["roots"] == [{"name": "models", "path": str(home / "models")}]


@pytest.mark.parametrize("kwargs", [{"offset": -1}, {"limit": 0}, {"limit": 201}, {"directory": ""}])
def test_invalid_arguments(home: Path, kwargs: dict) -> None:
    with pytest.raises(ValueError):
        model_file_service.list_model_files(**kwargs)


def test_large_directory_scan_is_bounded(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    models = home / "models"
    models.mkdir()
    for index in range(6):
        (models / f"{index}.pt").touch()
    monkeypatch.setattr(model_file_service, "MAX_DIRECTORY_ENTRIES", 3)
    payload = model_file_service.list_model_files()
    assert payload["truncated"] is True
    assert payload["total"] == len(payload["entries"]) == 3


def test_route_status_and_read_only_contract(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    models = home / "models"
    models.mkdir()
    (models / "dit.pt").touch()

    async def run() -> None:
        app = web.Application()
        setup_settings_routes(app)
        async with TestServer(app) as server, TestClient(server) as client:
            for query, status in (
                ({}, 200), ({"directory": str(home)}, 403),
                ({"directory": str(models / "missing")}, 404),
                ({"directory": str(models / "..")}, 400),
                ({"limit": "no"}, 400), ({"offset": "-1"}, 400),
                ({"directory": str(models / ("x" * 300))}, 400),
            ):
                response = await client.get("/api/settings/model-files", params=query)
                assert response.status == status
                assert (await response.json())["ok"] is (status == 200)
            (home / "settings.toml").write_text("[broken", encoding="utf-8")
            response = await client.get("/api/settings/model-files")
            assert response.status == 409
            (home / "settings.toml").unlink()

            def forbidden_scan(_path):
                raise PermissionError("denied")

            monkeypatch.setattr(model_file_service.os, "scandir", forbidden_scan)
            response = await client.get("/api/settings/model-files")
            assert response.status == 403

            def io_failure(_path):
                raise OSError(errno.EIO, "fixture error")

            monkeypatch.setattr(model_file_service.os, "scandir", io_failure)
            response = await client.get("/api/settings/model-files")
            assert response.status == 503
            assert (await response.json())["ok"] is False
    asyncio.run(run())


def test_route_performs_filesystem_work_off_event_loop(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import web.routes.settings as routes
    caller_thread = threading.get_ident()
    seen = []

    def list_files(directory, **kwargs):
        seen.append((threading.get_ident(), directory, kwargs))
        return {"ok": True}

    monkeypatch.setattr(routes, "list_model_files", list_files)

    async def run() -> None:
        app = web.Application()
        setup_settings_routes(app)
        async with TestServer(app) as server, TestClient(server) as client:
            response = await client.get("/api/settings/model-files?directory=models&offset=2&limit=10")
            assert response.status == 200
    asyncio.run(run())
    assert seen[0][0] != caller_thread
    assert seen[0][1:] == ("models", {"offset": 2, "limit": 10})
