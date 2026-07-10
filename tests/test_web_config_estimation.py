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

def test_estimation_module_imports_without_facade_cycle():
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    script = (
        "import sys; "
        "import web.services.config.estimation as estimation; "
        "assert callable(estimation.estimate_training_steps); "
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

def test_step_estimate_defaults_max_train_steps_to_disabled_when_epoch_missing(tmp_path: Path, monkeypatch):
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    _write_step_estimate_dataset(tmp_path, dataset_path)

    estimate = config_service.estimate_training_steps("lora", "default", "imported")

    assert estimate["steps_per_epoch"] == 15
    assert estimate["max_train_epochs"] is None
    assert estimate["max_train_steps"] == 0
    assert estimate["uses_max_train_epochs"] is False
    assert estimate["duration_configured"] is False
    assert estimate["duration_mode"] == "unset"
    assert estimate["total_steps"] == 0

def test_step_estimate_uses_explicit_max_train_steps_when_epoch_missing(tmp_path: Path, monkeypatch):
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    _write_step_estimate_dataset(tmp_path, dataset_path)
    (configs / "imported" / "lora.toml").write_text(
        "\n".join(
            [
                'dataset_config = "configs/datasets/lora.toml"',
                "max_train_steps = 1600",
            ]
        ),
        encoding="utf-8",
    )

    estimate = config_service.estimate_training_steps("lora", "default", "imported")

    assert estimate["steps_per_epoch"] == 15
    assert estimate["max_train_epochs"] is None
    assert estimate["max_train_steps"] == 1600
    assert estimate["uses_max_train_epochs"] is False
    assert estimate["duration_configured"] is True
    assert estimate["duration_mode"] == "steps"
    assert estimate["total_steps"] == 1600

def test_step_estimate_prefers_epochs_over_max_train_steps(tmp_path: Path, monkeypatch):
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    _write_step_estimate_dataset(tmp_path, dataset_path)
    (configs / "imported" / "lora.toml").write_text(
        "\n".join(
            [
                'dataset_config = "configs/datasets/lora.toml"',
                "max_train_epochs = 2",
                "max_train_steps = 1600",
            ]
        ),
        encoding="utf-8",
    )

    estimate = config_service.estimate_training_steps("lora", "default", "imported")

    assert estimate["steps_per_epoch"] == 15
    assert estimate["max_train_epochs"] == 2
    assert estimate["max_train_steps"] == 1600
    assert estimate["uses_max_train_epochs"] is True
    assert estimate["duration_configured"] is True
    assert estimate["duration_mode"] == "epochs"
    assert estimate["total_steps"] == 30

def test_step_estimate_counts_trigger_clone_weight(tmp_path: Path, monkeypatch):
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    source_dir = tmp_path / "image_dataset" / "character"
    nested_dir = source_dir / "nested"
    nested_dir.mkdir(parents=True)
    for idx in range(4):
        target_dir = nested_dir if idx == 3 else source_dir
        Image.new("RGB", (8, 8), color=(idx, 20, 40)).save(target_dir / f"{idx}.png")
    resized_dir = tmp_path / "post_image_dataset" / "character_resized"
    resized_dir.mkdir(parents=True)
    for idx in range(3):
        Image.new("RGB", (8, 8), color=(idx, 60, 40)).save(resized_dir / f"{idx}.png")
    dataset_path.write_text(
        "\n".join(
            [
                "[[datasets]]",
                "",
                "[[datasets.subsets]]",
                'image_dir = "post_image_dataset/character_resized"',
                'cache_dir = "post_image_dataset/character_lora"',
                "recursive = true",
                "num_repeats = 5",
                'custom_attributes = {source_dir = "image_dataset/character", trigger_clone = {enabled = true, prompt = "my_character", num_repeats = 2}}',
            ]
        ),
        encoding="utf-8",
    )

    estimate = config_service.estimate_training_steps("lora", "default", "imported")

    assert estimate["weighted_image_count"] == 23
    assert estimate["train_image_count"] == 7
    assert estimate["steps_per_epoch"] == 23
    row = estimate["datasets"][0]
    assert row["weighted_image_count"] == 15
    assert row["trigger_clone_image_count"] == 4
    assert row["trigger_clone_weighted_image_count"] == 8

def test_estimation_helpers_remain_available_from_legacy_module(tmp_path: Path, monkeypatch):
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    _write_step_estimate_dataset(tmp_path, dataset_path)
    monkeypatch.setattr(legacy_config, "ROOT", tmp_path)
    monkeypatch.setattr(legacy_config, "CONFIGS_DIR", configs)
    monkeypatch.setattr(legacy_config, "DATASET_PRESETS_DIR", configs / "datasets")
    monkeypatch.setattr(legacy_config, "GUI_METHODS_DIR", configs / "gui-methods")
    monkeypatch.setattr(legacy_config, "IMPORTED_CONFIGS_DIR", configs / "imported")
    monkeypatch.setattr(legacy_config, "PRESETS_FILE", configs / "presets.toml")
    monkeypatch.setattr(legacy_config, "WEB_FILE_GROUPS_FILE", configs / "web-file-groups.toml")
    monkeypatch.setattr(legacy_config, "WEB_USER_LOCKS_FILE", configs / "web-user-locks.toml")

    expected_shims = ("estimate_training_steps",)
    assert tuple(legacy_config._ESTIMATION_SHIM_NAMES) == expected_shims
    for name in expected_shims:
        assert getattr(legacy_config, name) is legacy_config._ESTIMATION_SHIMS[name]
        assert (
            getattr(legacy_config, name).__doc__
            == f"Compatibility shim forwarding to web.services.config.estimation.{name}."
        )

    estimate = legacy_config.estimate_training_steps("lora", "default", "imported")

    assert estimate["steps_per_epoch"] == 15
    assert estimate["max_train_steps"] == 0
    assert estimate["duration_mode"] == "unset"
    assert estimate["dataset_num_repeats"] == 5
    assert estimate["weighted_image_count"] == 15
    assert legacy_config.estimate_training_steps is legacy_config._ESTIMATION_SHIMS["estimate_training_steps"]

