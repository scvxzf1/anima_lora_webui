from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest
import toml
from PIL import Image

from web.routes import config as config_routes
from web.services import config_service
from web.services.config import _legacy as legacy_config
from web.services.config import datasets as config_datasets
from web.services.config import metadata as config_metadata
from web.services.config import paths as config_paths

from tests import web_config_test_support as _web_config_support

globals().update(
    {
        name: value
        for name, value in vars(_web_config_support).items()
        if name == "Path" or not name.startswith("__")
    }
)

def test_sample_prompts_module_imports_without_facade_cycle():
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    script = (
        "import sys; "
        "import web.services.config.sample_prompts as sample_prompts; "
        "assert callable(sample_prompts._normalize_prompt_file_path); "
        "assert sample_prompts._normalize_config_rel_path('configs/sample_prompts.txt') == 'configs/sample_prompts.txt'; "
        "assert 'web.services.config_service' not in sys.modules; "
        "assert 'web.services.config._legacy' not in sys.modules"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr or result.stdout

def test_sample_prompts_roundtrip_preserves_comments_blank_lines_and_spacing(tmp_path: Path, monkeypatch):
    root = tmp_path
    configs = root / "configs"
    configs.mkdir()
    monkeypatch.setattr(config_service, "ROOT", root)

    original = "# 角色 A\n\n  masterpiece, best quality  \n# 角色 B\nsolo, 1girl\n"
    saved = config_service.save_sample_prompts_file(original, "configs/sample_prompts.txt")
    loaded = config_service.load_sample_prompts_file("configs/sample_prompts.txt")

    assert (configs / "sample_prompts.txt").read_text(encoding="utf-8") == original
    assert saved["content"] == original
    assert loaded["content"] == original
    assert loaded["prompts"] == ["masterpiece, best quality", "solo, 1girl"]

    legacy_text = "# legacy\nsolo\n"
    monkeypatch.setattr(legacy_config, "ROOT", root)
    legacy_saved = legacy_config.save_sample_prompts_file(legacy_text, "configs/legacy_sample_prompts.txt")
    legacy_loaded = legacy_config.load_sample_prompts_file("configs/legacy_sample_prompts.txt")

    assert legacy_saved["content"] == legacy_text
    assert legacy_loaded["prompts"] == ["solo"]

def test_sample_prompts_save_can_fork_to_training_config_specific_file(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)

    saved = config_service.save_sample_prompts_file(
        "solo, character a\n",
        "configs/sample_prompts.txt",
        train_config_file="configs/imported/lora.toml",
    )

    assert saved["file"] == "configs/sample-prompts/imported/lora.txt"
    assert (configs / "sample_prompts.txt").exists() is False
    assert (configs / "sample-prompts" / "imported" / "lora.txt").read_text(encoding="utf-8") == "solo, character a\n"

def test_sample_prompts_save_rejects_training_config_outside_configs(tmp_path: Path, monkeypatch):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="训练配置文件路径不合法"):
        config_service.save_sample_prompts_file(
            "solo\n",
            "configs/sample_prompts.txt",
            train_config_file="../outside.toml",
        )

def test_sample_prompts_route_rejects_prompt_file_outside_configs(tmp_path: Path, monkeypatch):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)

    response = asyncio.run(
        config_routes.handle_sample_prompts_get(
            _QueryRequest({"file": "../outside.txt"}),  # type: ignore[arg-type]
        )
    )

    assert response.status == 400
    assert _json_response_payload(response)["error"] == "提示词文件路径不能包含 .."

def test_sample_prompts_put_route_forks_to_training_config_specific_file(
    tmp_path: Path,
    monkeypatch,
):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)

    response = asyncio.run(
        config_routes.handle_sample_prompts_put(
            _JsonRequest(
                {
                    "file": "configs/sample_prompts.txt",
                    "train_config_file": "configs/imported/lora.toml",
                    "content": "# keep comment\n\nsolo, character a\n",
                }
            ),  # type: ignore[arg-type]
        )
    )
    payload = _json_response_payload(response)

    assert response.status == 200
    assert payload["file"] == "configs/sample-prompts/imported/lora.txt"
    assert payload["prompts"] == ["solo, character a"]
    assert (configs / "sample_prompts.txt").exists() is False
    assert (
        configs / "sample-prompts" / "imported" / "lora.txt"
    ).read_text(encoding="utf-8") == "# keep comment\n\nsolo, character a\n"

