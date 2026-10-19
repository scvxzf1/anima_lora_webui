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


def test_symlink_escape_logs_actionable_warning(tmp_path, caplog):
    """配置根用「子项软链指向外部」时必须留下可诊断日志，而不是静默返回空。"""
    import logging

    from web.services.config.paths import safe_config_subdir, safe_resolve

    configs_root = tmp_path / "external-root"
    configs_root.mkdir()
    repo_configs = tmp_path / "repo" / "configs"
    (repo_configs / "gui-methods").mkdir(parents=True)
    (repo_configs / "gui-methods" / "lora.toml").write_text("[base]\n", encoding="utf-8")
    (configs_root / "gui-methods").symlink_to(repo_configs / "gui-methods")

    with caplog.at_level(logging.WARNING, logger="web.services.config.paths"):
        assert safe_config_subdir("gui-methods", configs_dir=configs_root) is None
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("config path rejected" in msg for msg in messages)
    assert any("safe_config_subdir" in msg for msg in messages)
    assert any(str(configs_root) in msg for msg in messages)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="web.services.config.paths"):
        assert safe_resolve("configs/gui-methods/lora.toml", root=tmp_path, configs_dir=configs_root) is None
    assert any("config path rejected" in rec.getMessage() for rec in caplog.records)


def test_parent_traversal_stays_silent(tmp_path, caplog):
    """真正的 .. 穿越属于常规拒绝，不应刷 warning 日志。"""
    import logging

    from web.services.config.paths import safe_config_subdir

    configs_root = tmp_path / "configs"
    configs_root.mkdir()

    with caplog.at_level(logging.WARNING, logger="web.services.config.paths"):
        assert safe_config_subdir("../escape", configs_dir=configs_root) is None
    assert not [rec for rec in caplog.records if "config path rejected" in rec.getMessage()]

