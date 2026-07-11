"""Structured schema warnings for raw save/patch HTTP contracts."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from web.routes import config as config_routes
from web.services import config_service


class _FakeRequest:
    def __init__(self, payload: dict[str, Any] | None = None, query: dict[str, str] | None = None):
        self._payload = payload or {}
        self.query = query or {}
        self.app: dict[str, Any] = {}

    async def json(self) -> dict[str, Any]:
        return self._payload


def _json(response) -> dict[str, Any]:
    import json

    return json.loads(response.text or "{}")


def _write_tree(tmp_path: Path) -> str:
    configs = tmp_path / "configs"
    imported = configs / "imported"
    imported.mkdir(parents=True)
    (imported / "lora.toml").write_text(
        "\n".join(
            [
                'output_name = "demo"',
                'network_module = "networks.lora_anima"',
                "max_train_steps = 10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return "configs/imported/lora.toml"


def test_save_raw_file_returns_structured_warnings(tmp_path: Path, monkeypatch):
    rel = _write_tree(tmp_path)
    monkeypatch.setattr(config_service, "ROOT", tmp_path)
    monkeypatch.setattr(config_service, "CONFIGS_DIR", tmp_path / "configs")
    config_service.set_configs_root(tmp_path / "configs")

    content = (tmp_path / "configs" / "imported" / "lora.toml").read_text(encoding="utf-8")
    content += "\ncustom_unknown_key = 123\n"
    result = config_service.save_raw_file(rel, content)
    assert isinstance(result, tuple)
    assert len(result) == 3
    ok, msg, warnings = result
    assert ok is True
    assert isinstance(warnings, list)
    assert warnings, "unknown key should surface as schema warning"
    assert any("custom_unknown_key" in str(w) or "unknown" in str(w).lower() for w in warnings)


def test_http_raw_put_includes_warnings_field(tmp_path: Path, monkeypatch):
    rel = _write_tree(tmp_path)
    monkeypatch.setattr(config_service, "ROOT", tmp_path)
    monkeypatch.setattr(config_service, "CONFIGS_DIR", tmp_path / "configs")
    config_service.set_configs_root(tmp_path / "configs")
    content = (tmp_path / "configs" / "imported" / "lora.toml").read_text(encoding="utf-8")
    content += "\ncustom_unknown_key = 1\n"

    response = asyncio.run(
        config_routes.handle_raw_put(
            _FakeRequest(payload={"file": rel, "content": content})  # type: ignore[arg-type]
        )
    )
    assert response.status == 200
    payload = _json(response)
    assert payload["ok"] is True
    assert isinstance(payload.get("warnings"), list)
    assert payload["warnings"]


def test_http_raw_patch_includes_warnings_field(tmp_path: Path, monkeypatch):
    rel = _write_tree(tmp_path)
    monkeypatch.setattr(config_service, "ROOT", tmp_path)
    monkeypatch.setattr(config_service, "CONFIGS_DIR", tmp_path / "configs")
    config_service.set_configs_root(tmp_path / "configs")

    response = asyncio.run(
        config_routes.handle_raw_patch(
            _FakeRequest(payload={"file": rel, "values": {"custom_user_flag": True}})  # type: ignore[arg-type]
        )
    )
    assert response.status == 200
    payload = _json(response)
    assert payload["ok"] is True
    assert isinstance(payload.get("warnings"), list)
    assert payload["warnings"]
    assert "changed" in payload
