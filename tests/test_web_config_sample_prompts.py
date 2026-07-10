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
    monkeypatch.setattr(config_service, "CONFIGS_DIR", configs)
    monkeypatch.setattr(
        config_service,
        "DEFAULT_SAMPLE_PROMPTS_FILE",
        str(configs / "sample_prompts.txt"),
    )

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


def test_sample_prompts_fork_uses_external_configs_root(tmp_path: Path, monkeypatch):
    """外置 configs_root 时，fork 后的 sample prompts 必须落在外置根内。"""
    root = tmp_path / "project"
    external = tmp_path / "ext-configs"
    root.mkdir()
    # 先写在项目 configs，再由 helper 挪到外置根（模拟外置部署）
    project_configs = root / "configs"
    (project_configs / "gui-methods").mkdir(parents=True)
    (project_configs / "gui-methods" / "lora.toml").write_text(
        'output_name = "demo"\n',
        encoding="utf-8",
    )
    _patch_external_config_service_paths(monkeypatch, root, external)

    from web.services.config import sample_prompts as sample_prompts_mod

    monkeypatch.setattr(sample_prompts_mod, "ROOT", root, raising=False)
    monkeypatch.setattr(sample_prompts_mod, "CONFIGS_DIR", external, raising=False)
    monkeypatch.setattr(
        sample_prompts_mod,
        "DEFAULT_SAMPLE_PROMPTS_FILE",
        str(external / "sample_prompts.txt"),
        raising=False,
    )

    # 项目内再造一个假 configs，确保实现不会误写这里
    leak_dir = root / "configs" / "sample-prompts" / "gui-methods"
    leak_dir.mkdir(parents=True)

    saved = config_service.save_sample_prompts_file(
        "a\nb\n",
        train_config_file="configs/gui-methods/lora.toml",
    )
    saved_path = external / "sample-prompts" / "gui-methods" / "lora.txt"
    project_leak = leak_dir / "lora.txt"

    assert saved["ok"] is True
    assert saved["file"] == "configs/sample-prompts/gui-methods/lora.txt"
    assert saved_path.exists()
    assert saved_path.read_text(encoding="utf-8").splitlines()[0].strip() == "a"
    assert project_leak.exists() is False


def test_sample_prompts_load_uses_external_configs_root(tmp_path: Path, monkeypatch):
    """外置 configs_root 时，读取也必须走外置根。"""
    root = tmp_path / "project"
    external = tmp_path / "ext-configs"
    root.mkdir()
    project_configs = root / "configs"
    project_configs.mkdir()
    (project_configs / "sample_prompts.txt").write_text(
        "# note\nsolo, from external\n",
        encoding="utf-8",
    )
    _patch_external_config_service_paths(monkeypatch, root, external)

    # 外置挪走后，项目内再建一个旧路径文件，证明不能读到这里
    (root / "configs").mkdir(exist_ok=True)
    (root / "configs" / "sample_prompts.txt").write_text(
        "solo, from project\n",
        encoding="utf-8",
    )

    from web.services.config import sample_prompts as sample_prompts_mod

    monkeypatch.setattr(sample_prompts_mod, "ROOT", root, raising=False)
    monkeypatch.setattr(sample_prompts_mod, "CONFIGS_DIR", external, raising=False)
    monkeypatch.setattr(
        sample_prompts_mod,
        "DEFAULT_SAMPLE_PROMPTS_FILE",
        str(external / "sample_prompts.txt"),
        raising=False,
    )

    loaded = config_service.load_sample_prompts_file("configs/sample_prompts.txt")
    assert loaded["prompts"] == ["solo, from external"]

