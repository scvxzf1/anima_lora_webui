"""Shared helpers for web config test modules."""

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



from web.services import config_service
from web.services.config import _legacy as legacy_config


def _json_response_payload(response) -> dict[str, Any]:
    return json.loads(response.text or "{}")


def _write_minimal_config_tree(root: Path) -> tuple[Path, Path]:
    configs = root / "configs"
    (configs / "imported").mkdir(parents=True)
    (configs / "datasets").mkdir(parents=True)
    (root / "tasks.py").write_text("print('tasks')\n", encoding="utf-8")
    (root / "library" / "preprocess").mkdir(parents=True)
    (root / "library" / "__init__.py").write_text("", encoding="utf-8")
    (root / "library" / "preprocess" / "__init__.py").write_text("", encoding="utf-8")
    preprocess_dir = root / "scripts" / "preprocess"
    preprocess_dir.mkdir(parents=True)
    (root / "scripts" / "__init__.py").write_text("", encoding="utf-8")
    (root / "scripts" / "tasks").mkdir(parents=True)
    (root / "scripts" / "tasks" / "__init__.py").write_text("", encoding="utf-8")
    (root / "scripts" / "tasks" / "preprocess.py").write_text("", encoding="utf-8")
    for rel_path in [
        preprocess_dir / "resize_images.py",
        preprocess_dir / "cache_latents.py",
        preprocess_dir / "cache_text_embeddings.py",
    ]:
        rel_path.write_text("from library.preprocess import resize_to_buckets\n", encoding="utf-8")
    (configs / "base.toml").write_text(
        "\n".join(
            [
                'source_image_dir = "image_dataset"',
                'resized_image_dir = "post_image_dataset/resized"',
                'lora_cache_dir = "post_image_dataset/lora"',
                'pretrained_model_name_or_path = "models/diffusion_models/anima-base-v1.0.safetensors"',
                'qwen3 = "models/text_encoders/qwen_3_06b_base.safetensors"',
                'vae = "models/vae/qwen_image_vae.safetensors"',
            ]
        ),
        encoding="utf-8",
    )
    (configs / "presets.toml").write_text("[default]\n", encoding="utf-8")
    (configs / "imported" / "lora.toml").write_text(
        'dataset_config = "configs/datasets/lora.toml"\n',
        encoding="utf-8",
    )
    dataset_path = configs / "datasets" / "lora.toml"
    return configs, dataset_path


def _write_step_estimate_dataset(root: Path, dataset_path: Path) -> None:
    image_dir = root / "post_image_dataset" / "a_resized"
    image_dir.mkdir(parents=True)
    for idx in range(3):
        Image.new("RGB", (8, 8), color=(idx, 20, 40)).save(image_dir / f"{idx}.png")
    dataset_path.write_text(
        "\n".join(
            [
                "[[datasets]]",
                "",
                "[[datasets.subsets]]",
                'image_dir = "post_image_dataset/a_resized"',
                "num_repeats = 5",
            ]
        ),
        encoding="utf-8",
    )


def _patch_config_service_paths(monkeypatch, root: Path) -> None:
    from web.services.config import dataset_editor as dataset_editor_impl
    from web.services.config import dataset_preset_paths as dataset_preset_paths_impl
    from web.services.config import dataset_presets_api as dataset_presets_api_impl
    from web.services.config import dataset_rows as dataset_rows_impl
    from web.services.config import datasets as datasets_impl
    from web.services.config import dataset_media as dataset_media_impl
    from web.services.config import estimation as estimation_impl
    from web.services.config import file_group_core as file_group_core_impl
    from web.services.config import file_group_runtime as file_group_runtime_impl
    from web.services.config import file_group_ops as file_group_ops_impl
    from web.services.config import file_groups as file_groups_impl
    from web.services.config import raw_files as raw_files_impl

    configs = root / "configs"
    for name, value in (
        ("ROOT", root),
        ("CONFIGS_DIR", configs),
        ("DATASET_PRESETS_DIR", configs / "datasets"),
        ("GUI_METHODS_DIR", configs / "gui-methods"),
        ("IMPORTED_CONFIGS_DIR", configs / "imported"),
        ("PRESETS_FILE", configs / "presets.toml"),
        ("WEB_FILE_GROUPS_FILE", configs / "web-file-groups.toml"),
        ("WEB_USER_LOCKS_FILE", configs / "web-user-locks.toml"),
    ):
        for module in (
            config_service,
            legacy_config,
            datasets_impl,
            dataset_media_impl,
            dataset_editor_impl,
            dataset_preset_paths_impl,
            dataset_presets_api_impl,
            dataset_rows_impl,
            file_groups_impl,
            file_group_core_impl,
            file_group_runtime_impl,
            file_group_ops_impl,
            estimation_impl,
        ):
            monkeypatch.setattr(module, name, value, raising=False)
        monkeypatch.setattr(raw_files_impl, name, value, raising=False)


def _write_selected_checkpoint_preflight_config(
    tmp_path: Path, monkeypatch, extra_lines: list[str]
) -> None:
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    source_dir = tmp_path / "image_dataset" / "selected"
    source_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(source_dir / "sample.png")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "anima.safetensors").write_bytes(b"model")
    (tmp_path / "models" / "qwen.safetensors").write_bytes(b"qwen")
    (tmp_path / "models" / "vae.safetensors").write_bytes(b"vae")
    selected_config = configs / "imported" / "selected.toml"
    selected_config.write_text(
        "\n".join(
            [
                'source_image_dir = "image_dataset/selected"',
                'pretrained_model_name_or_path = "models/anima.safetensors"',
                'qwen3 = "models/qwen.safetensors"',
                'vae = "models/vae.safetensors"',
                *extra_lines,
            ]
        ),
        encoding="utf-8",
    )


def _patch_external_config_service_paths(monkeypatch, root: Path, configs: Path) -> None:
    from web.services.config import dataset_editor as dataset_editor_impl
    from web.services.config import dataset_preset_paths as dataset_preset_paths_impl
    from web.services.config import dataset_presets_api as dataset_presets_api_impl
    from web.services.config import dataset_rows as dataset_rows_impl
    from web.services.config import datasets as datasets_impl
    from web.services.config import dataset_media as dataset_media_impl
    from web.services.config import estimation as estimation_impl
    from web.services.config import file_group_core as file_group_core_impl
    from web.services.config import file_group_runtime as file_group_runtime_impl
    from web.services.config import file_group_ops as file_group_ops_impl
    from web.services.config import file_groups as file_groups_impl
    from web.services.config import raw_files as raw_files_impl

    project_configs = root / "configs"
    if project_configs.exists():
        project_configs.rename(configs)
    for name, value in (
        ("ROOT", root),
        ("CONFIGS_DIR", configs),
        ("DATASET_PRESETS_DIR", configs / "datasets"),
        ("GUI_METHODS_DIR", configs / "gui-methods"),
        ("IMPORTED_CONFIGS_DIR", configs / "imported"),
        ("PRESETS_FILE", configs / "presets.toml"),
        ("WEB_FILE_GROUPS_FILE", configs / "web-file-groups.toml"),
        ("WEB_USER_LOCKS_FILE", configs / "web-user-locks.toml"),
    ):
        for module in (
            config_service,
            legacy_config,
            datasets_impl,
            dataset_media_impl,
            dataset_editor_impl,
            dataset_preset_paths_impl,
            dataset_presets_api_impl,
            dataset_rows_impl,
            file_groups_impl,
            file_group_core_impl,
            file_group_runtime_impl,
            file_group_ops_impl,
            estimation_impl,
        ):
            monkeypatch.setattr(module, name, value, raising=False)
        monkeypatch.setattr(raw_files_impl, name, value, raising=False)


def _patch_output_root(monkeypatch, output_root: Path) -> None:
    monkeypatch.setattr(config_service, "resolve_output_root", lambda: output_root.resolve())
    monkeypatch.setattr(
        config_service,
        "_display_settings_path",
        lambda path: _display_test_path(Path(path), output_root.parents[1]),
    )


def _display_test_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


class _JsonRequest:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    async def json(self) -> dict:
        return self._payload


class _QueryRequest:
    def __init__(self, query: dict[str, str] | None = None) -> None:
        self.query = query or {}
