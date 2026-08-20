from __future__ import annotations

import asyncio
import json
from pathlib import Path

from web.routes import config as config_routes
from web.routes import training as training_routes

from tests.web_config_test_support import (
    _JsonRequest,
    _patch_config_service_paths,
    _write_minimal_config_tree,
)


def _payload(response) -> dict:
    return json.loads(response.text or "{}")


def test_dragon_training_preview_save_as_patch_and_preflight_use_isolated_config_root(
    tmp_path: Path,
    monkeypatch,
):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    source = configs / "imported" / "lora.toml"
    original = source.read_text(encoding="utf-8")

    preview_response = asyncio.run(config_routes.handle_raw_patch_preview(_JsonRequest({
        "file": "configs/imported/lora.toml",
        "values": {"output_name": "dragon-preview", "max_train_steps": 120},
    })))
    assert preview_response.status == 200
    preview = _payload(preview_response)
    assert set(preview["changed"]) == {"output_name", "max_train_steps"}
    assert 'output_name = "dragon-preview"' in preview["content"]
    assert source.read_text(encoding="utf-8") == original

    save_as_response = asyncio.run(config_routes.handle_raw_save_as(_JsonRequest({
        "file": "configs/imported/dragon_copy.toml",
        "content": preview["content"],
    })))
    assert save_as_response.status == 200
    copy_path = configs / "imported" / "dragon_copy.toml"
    assert copy_path.exists()
    assert 'max_train_steps = 120' in copy_path.read_text(encoding="utf-8")

    collision_response = asyncio.run(config_routes.handle_raw_save_as(_JsonRequest({
        "file": "configs/imported/dragon_copy.toml",
        "content": 'output_name = "overwrite"\n',
    })))
    assert collision_response.status == 400
    assert "已存在" in _payload(collision_response)["error"]
    assert 'output_name = "dragon-preview"' in copy_path.read_text(encoding="utf-8")

    patch_response = asyncio.run(config_routes.handle_raw_patch(_JsonRequest({
        "file": "configs/imported/dragon_copy.toml",
        "values": {"output_name": "dragon-saved"},
    })))
    assert patch_response.status == 200
    patch = _payload(patch_response)
    assert patch["changed"] == ["output_name"]
    assert 'output_name = "dragon-saved"' in copy_path.read_text(encoding="utf-8")

    preflight_response = asyncio.run(training_routes.handle_preflight(_JsonRequest({
        "variant": "lora",
        "preset": "default",
        "methods_subdir": "imported",
        "config_file": "configs/imported/dragon_copy.toml",
    })))
    assert preflight_response.status == 200
    preflight = _payload(preflight_response)
    assert set(preflight) >= {"ok", "summary", "checks", "errors", "warnings"}
    assert preflight["summary"]["checks"] == len(preflight["checks"])
    assert all(set(check) >= {"level", "key", "message"} for check in preflight["checks"])


def test_dragon_training_patch_rejects_blank_output_without_writing(
    tmp_path: Path,
    monkeypatch,
):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    path = configs / "imported" / "lora.toml"
    path.write_text('output_name = "keep-me"\n', encoding="utf-8")

    response = asyncio.run(config_routes.handle_raw_patch(_JsonRequest({
        "file": "configs/imported/lora.toml",
        "values": {"output_name": "   "},
    })))
    assert response.status == 400
    assert "output_name 不能为空" in _payload(response)["error"]
    assert path.read_text(encoding="utf-8") == 'output_name = "keep-me"\n'
