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



# Split from test_web_config_service.py

from tests import web_config_test_support as _web_config_support

globals().update(
    {
        name: value
        for name, value in vars(_web_config_support).items()
        if name == "Path" or not name.startswith("__")
    }
)

def test_datasets_direct_path_and_text_helpers_work_without_facade_snapshot(tmp_path: Path):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    script = f"""
import sys
from pathlib import Path

import web.services.config.datasets as datasets

root = Path({str(tmp_path)!r})
configs = root / "configs"
(configs / "datasets").mkdir(parents=True)
dataset_path = configs / "datasets" / "direct.toml"
dataset_path.write_text("[[datasets]]\\n", encoding="utf-8")

datasets.ROOT = root
datasets.CONFIGS_DIR = configs
datasets.GUI_METHODS_DIR = configs / "gui-methods"
datasets.IMPORTED_CONFIGS_DIR = configs / "imported"
datasets.PRESETS_FILE = configs / "presets.toml"
datasets.WEB_FILE_GROUPS_FILE = configs / "web-file-groups.toml"
datasets.WEB_USER_LOCKS_FILE = configs / "web-user-locks.toml"
datasets.DATASET_PRESETS_DIR = configs / "datasets"

import web.services.config.dataset_editor as dataset_editor
import web.services.config.dataset_rows as dataset_rows
for module in (dataset_editor, dataset_rows):
    module.ROOT = root
    module.CONFIGS_DIR = configs

resolved = datasets._dataset_config_path_from_cfg.__wrapped__({{"dataset_config": "configs/datasets/direct.toml"}})
classified = datasets._classify_nl_tag_caption_text.__wrapped__("1girl, blue hair, smile, looking at viewer")

assert resolved == dataset_path.resolve()
assert datasets._dataset_path_value("image_dataset/hero", {{}}) == "image_dataset/hero"
assert classified["kind"] == "tag"
assert "web.services.config_service" not in sys.modules
assert "web.services.config._legacy" not in sys.modules
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_save_dataset_editor_does_not_overwrite_dataset_when_train_patch_fails(tmp_path: Path, monkeypatch):
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    original_dataset = "# keep me\n[[datasets]]\nresolution = 512\n"
    dataset_path.write_text(original_dataset, encoding="utf-8")
    _patch_config_service_paths(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="TOML 更新失败"):
        config_service.save_dataset_editor(
            "lora",
            "default",
            "imported",
            [
                {
                    "source_dir": "new_source",
                    "image_dir": "new_resized",
                    "cache_dir": "new_cache",
                    "num_repeats": 2,
                }
            ],
            train_file="configs/imported/lora.toml",
            train_content='dataset_config = "configs/datasets/lora.toml"\ninvalid = [\n',
        )

    assert dataset_path.read_text(encoding="utf-8") == original_dataset


def test_save_dataset_editor_restores_dataset_when_train_write_fails(tmp_path: Path, monkeypatch):
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    original_dataset = "# original\n[[datasets]]\nresolution = 512\n"
    dataset_path.write_text(original_dataset, encoding="utf-8")
    _patch_config_service_paths(monkeypatch, tmp_path)

    def fail_train_save(rel_path: str, content: str, **kwargs):
        if rel_path == "configs/imported/lora.toml":
            return False, "训练 TOML 写入失败"
        return original_save_raw_file(rel_path, content, **kwargs)  # returns (ok,msg,warnings)

    original_save_raw_file = config_service.save_raw_file
    monkeypatch.setattr(config_service, "save_raw_file", fail_train_save)

    with pytest.raises(ValueError, match="训练 TOML 写入失败"):
        config_service.save_dataset_editor(
            "lora",
            "default",
            "imported",
            [
                {
                    "source_dir": "new_source",
                    "image_dir": "new_resized",
                    "cache_dir": "new_cache",
                    "num_repeats": 2,
                }
            ],
            train_file="configs/imported/lora.toml",
        )

    assert dataset_path.read_text(encoding="utf-8") == original_dataset


def test_raw_patch_ignores_dataset_picker_ui_field(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    train_rel = "configs/imported/lora.toml"

    ok, msg, content, changed, _warnings = config_service.patch_raw_file_values(
        train_rel,
        {
            "dataset_config_picker": "configs/datasets/character_a.toml",
            "precision_preference": "fp32",
            "output_name": "clean",
        },
    )

    assert ok is True, msg
    assert changed == ["output_name"]
    assert 'output_name = "clean"' in content
    assert "dataset_config_picker" not in content
    assert "precision_preference" not in content
    assert "dataset_config_picker" not in (configs / "imported" / "lora.toml").read_text(encoding="utf-8")
    assert "precision_preference" not in (configs / "imported" / "lora.toml").read_text(encoding="utf-8")


def test_save_dataset_editor_accepts_source_only_rows(tmp_path: Path, monkeypatch):
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    (configs / "imported" / "lora.toml").write_text(
        "\n".join(
            [
                'dataset_config = "configs/datasets/lora.toml"',
                "train_batch_size = 3",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)

    result = config_service.save_dataset_editor(
        "lora",
        "default",
        "imported",
        [
            {
                "source_dir": "image_dataset/source_only",
                "num_repeats": 3,
                "settings": {"resolution": 768},
            }
        ],
        train_file="configs/imported/lora.toml",
    )

    assert result["ok"] is True
    assert result["datasets"][0]["source_dir"] == "image_dataset/source_only"
    assert result["datasets"][0]["image_dir"].endswith("source_only_resized")
    assert result["datasets"][0]["cache_dir"].endswith("source_only_lora_cache")
    data = toml.loads(dataset_path.read_text(encoding="utf-8"))
    assert data["datasets"][0]["batch_size"] == 3
    subset = data["datasets"][0]["subsets"][0]
    assert subset["custom_attributes"]["source_dir"] == "image_dataset/source_only"
    assert subset["image_dir"].endswith("source_only_resized")
    assert subset["cache_dir"].endswith("source_only_lora_cache")


def test_save_dataset_editor_uses_pending_form_config_values(tmp_path: Path, monkeypatch):
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    (configs / "imported" / "lora.toml").write_text(
        "\n".join(
            [
                'dataset_config = "configs/datasets/lora.toml"',
                "train_batch_size = 3",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)

    result = config_service.save_dataset_editor(
        "lora",
        "default",
        "imported",
        [
            {
                "source_dir": "image_dataset/source_only",
                "num_repeats": 1,
                "settings": {"resolution": 768},
            }
        ],
        config_values={"train_batch_size": 5},
        train_file="configs/imported/lora.toml",
    )

    assert result["ok"] is True
    data = toml.loads(dataset_path.read_text(encoding="utf-8"))
    assert data["datasets"][0]["batch_size"] == 5


def test_save_dataset_editor_preserves_regularization_fields_to_training_config(tmp_path: Path, monkeypatch):
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    train_path = configs / "imported" / "lora.toml"
    train_path.write_text(
        "\n".join(
            [
                'dataset_config = "configs/datasets/lora.toml"',
                "prior_loss_weight = 1.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)

    result = config_service.save_dataset_editor(
        "lora",
        "default",
        "imported",
        [
            {
                "source_dir": "image_dataset/train",
                "image_dir": "post_image_dataset/train_resized",
                "cache_dir": "post_image_dataset/train_cache",
                "num_repeats": 2,
            },
            {
                "source_dir": "image_dataset/reg",
                "image_dir": "post_image_dataset/reg_resized",
                "cache_dir": "post_image_dataset/reg_cache",
                "num_repeats": 1,
                "is_reg": True,
                "settings": {"prior_loss_weight": 2.5},
            },
        ],
        train_file="configs/imported/lora.toml",
    )

    assert result["ok"] is True
    data = toml.loads(dataset_path.read_text(encoding="utf-8"))
    assert data["datasets"][1]["subsets"][0]["is_reg"] is True
    assert data["datasets"][1]["prior_loss_weight"] == 2.5

    train_data = toml.loads(train_path.read_text(encoding="utf-8"))
    assert train_data["source_image_dir"] == "image_dataset/train"
    assert train_data["prior_loss_weight"] == 2.5


def test_save_dataset_editor_merges_partial_regularization_settings(tmp_path: Path, monkeypatch):
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)

    result = config_service.save_dataset_editor(
        "lora",
        "default",
        "imported",
        [
            {
                "source_dir": "image_dataset/train",
                "image_dir": "post_image_dataset/train_resized",
                "cache_dir": "post_image_dataset/train_cache",
                "num_repeats": 2,
            },
            {
                "source_dir": "image_dataset/reg",
                "image_dir": "post_image_dataset/reg_resized",
                "cache_dir": "post_image_dataset/reg_cache",
                "num_repeats": 1,
                "is_reg": True,
                "settings": {"prior_loss_weight": 2.5},
            },
        ],
        defaults={"resolution": 768, "batch_size": 1, "prior_loss_weight": 1.0},
        train_file="configs/imported/lora.toml",
    )

    assert result["ok"] is True
    data = toml.loads(dataset_path.read_text(encoding="utf-8"))
    assert data["datasets"][1]["resolution"] == 768
    assert data["datasets"][1]["prior_loss_weight"] == 2.5


def test_save_dataset_editor_accepts_top_level_regularization_weight(tmp_path: Path, monkeypatch):
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)

    result = config_service.save_dataset_editor(
        "lora",
        "default",
        "imported",
        [
            {
                "source_dir": "image_dataset/train",
                "image_dir": "post_image_dataset/train_resized",
                "cache_dir": "post_image_dataset/train_cache",
            },
            {
                "source_dir": "image_dataset/reg",
                "image_dir": "post_image_dataset/reg_resized",
                "cache_dir": "post_image_dataset/reg_cache",
                "is_reg": True,
                "prior_loss_weight": 2.5,
            },
        ],
        defaults={"resolution": 768, "batch_size": 1, "prior_loss_weight": 1.0},
        train_file="configs/imported/lora.toml",
    )

    assert result["ok"] is True
    data = toml.loads(dataset_path.read_text(encoding="utf-8"))
    assert data["datasets"][1]["resolution"] == 768
    assert data["datasets"][1]["prior_loss_weight"] == 2.5


def test_load_dataset_editor_uses_selected_training_config_dataset(tmp_path: Path, monkeypatch):
    configs, default_dataset_path = _write_minimal_config_tree(tmp_path)
    default_dataset_path.write_text(
        "\n".join(
            [
                "[[datasets]]",
                "resolution = 1024",
                "max_bucket_reso = 1024",
                "",
                "[[datasets.subsets]]",
                'image_dir = "post_image_dataset/default_resized"',
                'cache_dir = "post_image_dataset/default_cache"',
                'custom_attributes = { source_dir = "image_dataset/default" }',
            ]
        ),
        encoding="utf-8",
    )
    selected_dataset_path = configs / "datasets" / "selected.toml"
    selected_dataset_path.write_text(
        "\n".join(
            [
                "[[datasets]]",
                "resolution = 768",
                "max_bucket_reso = 768",
                "bucket_reso_steps = 32",
                "",
                "[[datasets.subsets]]",
                'image_dir = "post_image_dataset/selected_resized"',
                'cache_dir = "post_image_dataset/selected_cache"',
                'custom_attributes = { source_dir = "image_dataset/selected" }',
            ]
        ),
        encoding="utf-8",
    )
    override_dataset_path = configs / "datasets" / "override.toml"
    override_dataset_path.write_text(
        "\n".join(
            [
                "[[datasets]]",
                "resolution = 640",
                "max_bucket_reso = 640",
                "",
                "[[datasets.subsets]]",
                'image_dir = "post_image_dataset/override_resized"',
                'custom_attributes = { source_dir = "image_dataset/override" }',
            ]
        ),
        encoding="utf-8",
    )
    (configs / "imported" / "selected.toml").write_text(
        'dataset_config = "configs/datasets/selected.toml"\n',
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)

    selected = config_service.load_dataset_editor(
        "lora",
        "default",
        "imported",
        config_file="configs/imported/selected.toml",
    )
    overridden = config_service.load_dataset_editor(
        "lora",
        "default",
        "imported",
        config_file="configs/imported/selected.toml",
        dataset_config="configs/datasets/override.toml",
    )

    assert selected["dataset_config"] == "configs/datasets/selected.toml"
    assert selected["datasets"][0]["source_dir"] == "image_dataset/selected"
    assert selected["datasets"][0]["settings"]["resolution"] == 768
    assert selected["datasets"][0]["settings"]["max_bucket_reso"] == 768
    assert selected["datasets"][0]["settings"]["bucket_reso_steps"] == 32

    assert overridden["dataset_config"] == "configs/datasets/override.toml"
    assert overridden["datasets"][0]["source_dir"] == "image_dataset/override"
    assert overridden["datasets"][0]["settings"]["resolution"] == 640


def test_preflight_uses_selected_config_file_dataset_paths(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    (configs / "base.toml").write_text(
        "\n".join(
            [
                'source_image_dir = "missing_default_source"',
                'resized_image_dir = "missing_default_resized"',
                'lora_cache_dir = "missing_default_cache"',
                'pretrained_model_name_or_path = "models/anima.safetensors"',
                'qwen3 = "models/qwen.safetensors"',
                'vae = "models/vae.safetensors"',
            ]
        ),
        encoding="utf-8",
    )
    source_dir = tmp_path / "image_dataset" / "selected"
    source_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(source_dir / "sample.png")
    selected_config = configs / "imported" / "selected.toml"
    selected_config.write_text(
        "\n".join(
            [
                'source_image_dir = "image_dataset/selected"',
                'pretrained_model_name_or_path = "models/anima.safetensors"',
                'qwen3 = "models/qwen.safetensors"',
                'vae = "models/vae.safetensors"',
            ]
        ),
        encoding="utf-8",
    )

    result = config_service.preflight_training_config(
        "lora",
        "default",
        "imported",
        config_file="configs/imported/selected.toml",
    )

    source_checks = [item for item in result["checks"] if item["key"] == "source_image_dir"]
    assert source_checks[-1]["level"] == "ok"
    assert source_checks[-1]["path"] == "image_dataset/selected"
    assert "output_dir" not in {item["key"] for item in result["checks"]}
    env_checks = [item for item in result["checks"] if item["key"] == "preprocess_environment"]
    assert env_checks[-1]["level"] == "ok"


def test_preflight_allows_plain_web_config_with_missing_dataset_config_but_valid_source(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    source_dir = tmp_path / "image_dataset" / "selected"
    source_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(source_dir / "sample.png")
    selected_config = configs / "imported" / "selected.toml"
    selected_config.write_text(
        "\n".join(
            [
                'source_image_dir = "image_dataset/selected"',
                'pretrained_model_name_or_path = "models/anima.safetensors"',
                'qwen3 = "models/qwen.safetensors"',
                'vae = "models/vae.safetensors"',
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "anima.safetensors").write_bytes(b"model")
    (tmp_path / "models" / "qwen.safetensors").write_bytes(b"qwen")
    (tmp_path / "models" / "vae.safetensors").write_bytes(b"vae")

    result = config_service.preflight_training_config(
        "lora",
        "default",
        "imported",
        config_file="configs/imported/selected.toml",
    )

    assert result["ok"] is True
    keys = {item["key"] for item in result["checks"]}
    assert "dataset_config" not in keys


def test_preflight_runtime_config_checks_all_dataset_groups(tmp_path: Path, monkeypatch):
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    run_dir = tmp_path / "output" / "runs" / "522-20260523-114514"
    source_a = tmp_path / "image_dataset" / "a"
    source_b = tmp_path / "image_dataset" / "b"
    resized_a = run_dir / "dataset_cache" / "dataset-01" / "resized"
    cache_a = run_dir / "dataset_cache" / "dataset-01" / "lora"
    resized_b = run_dir / "dataset_cache" / "dataset-02" / "resized"
    cache_b = run_dir / "dataset_cache" / "dataset-02" / "lora"
    source_a.mkdir(parents=True)
    source_b.mkdir(parents=True)
    (run_dir / "model_cache").mkdir(parents=True)
    (run_dir / "training_output").mkdir(parents=True)
    resized_a.mkdir(parents=True)
    cache_a.mkdir(parents=True)
    resized_b.mkdir(parents=True)
    cache_b.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(resized_a / "a.png")
    Image.new("RGB", (8, 8), color=(60, 40, 20)).save(resized_b / "b.png")
    runtime_config = run_dir / "config.runtime.toml"
    dataset_config = run_dir / "dataset.runtime.toml"
    runtime_config.write_text(
        "\n".join(
            [
                'source_image_dir = "image_dataset/a"',
                'resized_image_dir = "output/runs/522-20260523-114514/dataset_cache/dataset-01/resized"',
                'lora_cache_dir = "output/runs/522-20260523-114514/dataset_cache/dataset-01/lora"',
                'dataset_config = "output/runs/522-20260523-114514/dataset.runtime.toml"',
                "cache_latents_to_disk = true",
                "cache_text_encoder_outputs_to_disk = true",
                'pretrained_model_name_or_path = "models/anima.safetensors"',
                'qwen3 = "models/qwen.safetensors"',
                'vae = "models/vae.safetensors"',
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "anima.safetensors").write_bytes(b"model")
    (tmp_path / "models" / "qwen.safetensors").write_bytes(b"qwen")
    (tmp_path / "models" / "vae.safetensors").write_bytes(b"vae")
    dataset_config.write_text(
        "\n".join(
            [
                "[[datasets]]",
                "",
                "[[datasets.subsets]]",
                'image_dir = "output/runs/522-20260523-114514/dataset_cache/dataset-01/resized"',
                'cache_dir = "output/runs/522-20260523-114514/dataset_cache/dataset-01/lora"',
                'custom_attributes = { source_dir = "image_dataset/a" }',
                "",
                "[[datasets]]",
                "",
                "[[datasets.subsets]]",
                'image_dir = "output/runs/522-20260523-114514/dataset_cache/dataset-02/resized"',
                'cache_dir = "bad-cache-file"',
                'custom_attributes = { source_dir = "image_dataset/b" }',
            ]
        ),
        encoding="utf-8",
    )

    result = config_service.preflight_training_config(
        "lora",
        "default",
        "imported",
        config_file="output/runs/522-20260523-114514/config.runtime.toml",
    )

    keys = {item["key"] for item in result["checks"]}
    assert "dataset_2_cache_dir" in keys
    assert result["ok"] is False


def test_preflight_runtime_config_checks_cache_sidecars_per_dataset(tmp_path: Path, monkeypatch):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    run_dir = tmp_path / "output" / "runs" / "522-20260523-114514"
    source_a = tmp_path / "image_dataset" / "a"
    source_b = tmp_path / "image_dataset" / "b"
    resized_a = run_dir / "dataset_cache" / "dataset-01" / "resized"
    cache_a = run_dir / "dataset_cache" / "dataset-01" / "lora"
    resized_b = run_dir / "dataset_cache" / "dataset-02" / "resized"
    cache_b = run_dir / "dataset_cache" / "dataset-02" / "lora"
    for path in (source_a, source_b, resized_a, cache_a, resized_b, cache_b, run_dir / "model_cache", run_dir / "training_output"):
        path.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(resized_a / "a.png")
    Image.new("RGB", (8, 8), color=(60, 40, 20)).save(resized_b / "b.png")
    (cache_a / "a.npz").write_bytes(b"latent")
    (cache_a / "a_anima_te.safetensors").write_bytes(b"te")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "anima.safetensors").write_bytes(b"model")
    (tmp_path / "models" / "qwen.safetensors").write_bytes(b"qwen")
    (tmp_path / "models" / "vae.safetensors").write_bytes(b"vae")
    runtime_config = run_dir / "config.runtime.toml"
    dataset_config = run_dir / "dataset.runtime.toml"
    runtime_config.write_text(
        "\n".join(
            [
                f'dataset_config = "{dataset_config.relative_to(tmp_path).as_posix()}"',
                f'source_image_dir = "{source_a.relative_to(tmp_path).as_posix()}"',
                f'resized_image_dir = "{resized_a.relative_to(tmp_path).as_posix()}"',
                f'lora_cache_dir = "{cache_a.relative_to(tmp_path).as_posix()}"',
                "cache_latents_to_disk = true",
                "cache_text_encoder_outputs_to_disk = true",
                'pretrained_model_name_or_path = "models/anima.safetensors"',
                'qwen3 = "models/qwen.safetensors"',
                'vae = "models/vae.safetensors"',
            ]
        ),
        encoding="utf-8",
    )
    dataset_config.write_text(
        "\n".join(
            [
                "[[datasets]]",
                "[[datasets.subsets]]",
                f'image_dir = "{resized_a.relative_to(tmp_path).as_posix()}"',
                f'cache_dir = "{cache_a.relative_to(tmp_path).as_posix()}"',
                f'custom_attributes = {{ source_dir = "{source_a.relative_to(tmp_path).as_posix()}" }}',
                "",
                "[[datasets]]",
                "[[datasets.subsets]]",
                f'image_dir = "{resized_b.relative_to(tmp_path).as_posix()}"',
                f'cache_dir = "{cache_b.relative_to(tmp_path).as_posix()}"',
                f'custom_attributes = {{ source_dir = "{source_b.relative_to(tmp_path).as_posix()}" }}',
            ]
        ),
        encoding="utf-8",
    )

    result = config_service.preflight_training_config(
        "lora",
        "default",
        "imported",
        config_file=runtime_config.relative_to(tmp_path).as_posix(),
    )

    checks = {item["key"]: item for item in result["checks"]}
    assert checks["latent_cache"]["level"] == "ok"
    assert checks["text_cache"]["level"] == "ok"
    assert checks["dataset_2_latent_cache"]["level"] == "warning"
    assert checks["dataset_2_text_cache"]["level"] == "warning"


def test_dataset_groups_are_dataset_only_and_presets_list_returns_groups(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    (configs / "web-file-groups.toml").write_text(
        "\n".join(
            [
                "[[groups]]",
                'id = "datasets"',
                'label = "数据集配置"',
                "open = true",
                "locked = false",
                "trainable = false",
                'patterns = ["configs/datasets/*.toml"]',
                'exclude = ["configs/datasets/character_b.toml"]',
                "",
                "[[groups]]",
                'id = "imported"',
                'label = "导入配置"',
                "open = true",
                "locked = false",
                "trainable = true",
                'methods_subdir = "imported"',
                'patterns = ["configs/imported/*.toml"]',
            ]
        ),
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)
    config_service.save_dataset_preset(
        "configs/datasets/character_a.toml",
        [{"source_dir": "image_dataset/a", "image_dir": "post/a", "cache_dir": "cache/a", "num_repeats": 1}],
        {},
    )
    config_service.save_dataset_preset(
        "configs/datasets/character_b.toml",
        [{"source_dir": "image_dataset/b", "image_dir": "post/b", "cache_dir": "cache/b", "num_repeats": 1}],
        {},
    )

    ok, message, group = config_service.create_config_file_group("角色数据集", kind="dataset")
    assert ok is True, message
    assert group is not None
    assert group["kind"] == "dataset"
    assert group["trainable"] is False
    assert group["open"] is False

    ok, message, moved = config_service.move_config_file_to_group(
        "configs/datasets/character_a.toml",
        group["id"],
    )
    assert ok is True, message
    assert moved is not None
    assert [item["path"] for item in moved["files"]] == ["configs/datasets/character_a.toml"]

    all_group_ids = [item["id"] for item in config_service.list_config_file_groups(kind="all")]
    training_group_ids = [item["id"] for item in config_service.list_config_file_groups(kind="training")]
    dataset_group_ids = [item["id"] for item in config_service.list_config_file_groups(kind="dataset")]
    assert "imported" in all_group_ids
    assert "datasets" in all_group_ids
    assert group["id"] in all_group_ids
    assert training_group_ids == ["imported"]
    assert "datasets" in dataset_group_ids
    assert group["id"] in dataset_group_ids
    assert "imported" not in dataset_group_ids
    with pytest.raises(ValueError, match="kind"):
        config_service.list_config_file_groups(kind="sample")

    response = asyncio.run(config_routes.handle_file_groups(_QueryRequest({})))
    assert response.status == 200
    body = json.loads(response.text)
    assert [item["id"] for item in body] == ["imported"]

    response = asyncio.run(config_routes.handle_file_groups(_QueryRequest({"kind": "dataset"})))
    assert response.status == 200
    body = json.loads(response.text)
    assert "datasets" in [item["id"] for item in body]
    assert "imported" not in [item["id"] for item in body]

    response = asyncio.run(config_routes.handle_file_groups(_QueryRequest({"kind": "all"})))
    assert response.status == 200
    body = json.loads(response.text)
    assert "datasets" in [item["id"] for item in body]
    assert "imported" in [item["id"] for item in body]

    ok, message, _group = config_service.move_config_file_to_group(
        "configs/datasets/character_a.toml",
        "imported",
    )
    assert ok is False
    assert "数据集预设只能移动到数据集分组" in message

    listed = config_service.list_dataset_presets()
    assert listed["ok"] is True
    assert listed["groups"][0]["id"] == "unfiled_datasets"
    assert listed["groups"][0]["open"] is True
    assert listed["groups"][0]["files"][0]["path"] == "configs/datasets/character_b.toml"
    assert any(item["id"] == group["id"] for item in listed["groups"])
    dataset_group = next(item for item in listed["groups"] if item["id"] == group["id"])
    assert dataset_group["open"] is False
    assert dataset_group["files"][0]["path"] == "configs/datasets/character_a.toml"


def test_dataset_group_specs_accept_windows_slashes(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    (configs / "datasets" / "character_a.toml").write_text(
        'source_image_dir = "image_dataset/a"\n',
        encoding="utf-8",
    )
    (configs / "datasets" / "character_b.toml").write_text(
        'source_image_dir = "image_dataset/b"\n',
        encoding="utf-8",
    )
    (configs / "web-file-groups.toml").write_text(
        "\n".join(
            [
                "[[groups]]",
                'id = "datasets"',
                'label = "数据集配置"',
                "open = true",
                "locked = false",
                "trainable = false",
                'patterns = ["configs\\\\datasets\\\\*.toml"]',
                'exclude = ["configs\\\\datasets\\\\character_b.toml"]',
                'order = ["configs\\\\datasets\\\\character_a.toml"]',
                "",
            ]
        ),
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)

    groups = config_service.list_config_file_groups(kind="dataset")

    assert groups[0]["files"][0]["path"] == "configs/datasets/character_a.toml"
    assert all(item["path"] != "configs/datasets/character_b.toml" for item in groups[0]["files"])
    saved = (configs / "web-file-groups.toml").read_text(encoding="utf-8")
    assert "configs\\\\datasets" in saved


def test_moving_dataset_group_file_cleans_legacy_windows_paths(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    for stem in ["character_a", "character_b"]:
        (configs / "datasets" / f"{stem}.toml").write_text(
            f'source_image_dir = "image_dataset/{stem}"\n',
            encoding="utf-8",
        )
    (configs / "web-file-groups.toml").write_text(
        "\n".join(
            [
                "[[groups]]",
                'id = "datasets"',
                'label = "数据集配置"',
                "open = true",
                "locked = false",
                "trainable = false",
                'files = ["configs\\\\datasets\\\\character_a.toml", "configs/datasets/character_b.toml"]',
                'order = ["configs\\\\datasets\\\\character_a.toml", "configs/datasets/character_b.toml"]',
                "",
                "[[groups]]",
                'id = "custom_datasets"',
                'label = "角色数据集"',
                "open = true",
                "locked = false",
                "trainable = false",
                'kind = "dataset"',
                "user_managed = true",
                "files = []",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)

    ok, message, moved = config_service.move_config_file_to_group(
        "configs/datasets/character_a.toml",
        "custom_datasets",
    )

    assert ok is True, message
    assert [item["path"] for item in moved["files"]] == ["configs/datasets/character_a.toml"]
    groups = config_service.list_config_file_groups(kind="dataset")
    datasets = next(group for group in groups if group["id"] == "datasets")
    assert [item["path"] for item in datasets["files"]] == ["configs/datasets/character_b.toml"]
    saved = (configs / "web-file-groups.toml").read_text(encoding="utf-8")
    assert "configs\\\\datasets\\\\character_a.toml" not in saved
    assert 'configs/datasets/character_a.toml' in saved


def test_dataset_preset_save_read_list_and_apply(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)

    saved = config_service.save_dataset_preset(
        "configs/datasets/character_a.toml",
        [
            {
                "source_dir": "image_dataset/a",
                "image_dir": "post_image_dataset/a_resized",
                "cache_dir": "post_image_dataset/a_cache",
                "num_repeats": 3,
            },
            {
                "source_dir": "image_dataset/b",
                "image_dir": "post_image_dataset/b_resized",
                "cache_dir": "post_image_dataset/b_cache",
                "num_repeats": 2,
            },
        ],
        {"resolution": 768, "batch_size": 1, "enable_bucket": True},
    )

    assert saved["ok"] is True
    assert saved["file"] == "configs/datasets/character_a.toml"

    loaded = config_service.load_dataset_preset("configs/datasets/character_a.toml")
    assert loaded["summary"]["dataset_count"] == 2
    assert loaded["summary"]["repeat_total"] == 5
    assert loaded["defaults"]["resolution"] == 768

    listed = config_service.list_dataset_presets()
    assert "configs/datasets/character_a.toml" in [item["path"] for item in listed["presets"]]

    applied = config_service.apply_dataset_preset_to_training_config(
        "configs/datasets/character_a.toml",
        "configs/imported/lora.toml",
    )
    assert applied["ok"] is True
    train_text = (configs / "imported" / "lora.toml").read_text(encoding="utf-8")
    assert 'dataset_config = "configs/datasets/character_a.toml"' in train_text
    assert 'source_image_dir = "image_dataset/a"' in train_text
    assert 'resized_image_dir = "post_image_dataset/a_resized"' in train_text
    assert 'lora_cache_dir = "post_image_dataset/a_cache"' in train_text
    assert "prior_loss_weight = 1.0" in train_text


def test_dataset_preset_remains_available_from_legacy_module(tmp_path: Path, monkeypatch):
    _write_minimal_config_tree(tmp_path)
    configs = tmp_path / "configs"
    monkeypatch.setattr(legacy_config, "ROOT", tmp_path)
    monkeypatch.setattr(legacy_config, "CONFIGS_DIR", configs)
    monkeypatch.setattr(legacy_config, "DATASET_PRESETS_DIR", configs / "datasets")
    monkeypatch.setattr(legacy_config, "GUI_METHODS_DIR", configs / "gui-methods")
    monkeypatch.setattr(legacy_config, "IMPORTED_CONFIGS_DIR", configs / "imported")
    monkeypatch.setattr(legacy_config, "PRESETS_FILE", configs / "presets.toml")
    monkeypatch.setattr(legacy_config, "WEB_FILE_GROUPS_FILE", configs / "web-file-groups.toml")
    monkeypatch.setattr(legacy_config, "WEB_USER_LOCKS_FILE", configs / "web-user-locks.toml")

    saved = legacy_config.save_dataset_preset(
        "configs/datasets/legacy_direct.toml",
        [
            {
                "source_dir": "image_dataset/legacy",
                "image_dir": "post_image_dataset/legacy_resized",
                "cache_dir": "post_image_dataset/legacy_cache",
                "num_repeats": 2,
            },
            {
                "source_dir": "image_dataset/legacy_reg",
                "image_dir": "post_image_dataset/legacy_reg_resized",
                "cache_dir": "post_image_dataset/legacy_reg_cache",
                "num_repeats": 1,
                "is_reg": True,
                "settings": {"prior_loss_weight": 1.5},
            },
        ],
        {"resolution": 640, "batch_size": 1, "prior_loss_weight": 1.5},
    )
    loaded = legacy_config.load_dataset_preset("configs/datasets/legacy_direct.toml")
    listed = legacy_config.list_dataset_presets()
    applied = legacy_config.apply_dataset_preset_to_training_config(
        "configs/datasets/legacy_direct.toml",
        "configs/imported/lora.toml",
    )

    assert saved["ok"] is True
    assert "configs/datasets/legacy_direct.toml" in [item["path"] for item in listed["presets"]]
    group_file = listed["groups"][0]["files"][0]
    assert group_file["summary"]["repeat_total"] == 3
    assert group_file["summary"]["train_dataset_count"] == 1
    assert group_file["summary"]["reg_dataset_count"] == 1
    assert loaded["summary"]["dataset_count"] == 2
    assert loaded["summary"]["train_dataset_count"] == 1
    assert loaded["summary"]["reg_dataset_count"] == 1
    assert loaded["summary"]["repeat_total"] == 3
    assert loaded["defaults"]["resolution"] == 640
    assert loaded["datasets"][1]["is_reg"] is True
    assert applied["values"]["source_image_dir"] == "image_dataset/legacy"
    assert applied["values"]["prior_loss_weight"] == 1.5


def test_legacy_dataset_exports_forward_to_split_dataset_module():
    from web.services.config import datasets as dataset_impl

    missing = []
    not_forwarded = []
    for name in dataset_impl.__all__:
        exported = getattr(legacy_config, name, None)
        if exported is None:
            missing.append(name)
            continue
        doc = str(getattr(exported, "__doc__", "") or "")
        if "web.services.config.datasets" not in doc:
            not_forwarded.append(name)

    assert missing == []
    assert not_forwarded == []


def test_legacy_dataset_private_helpers_stay_explicit_legacy_shims():
    from web.services.config import datasets as dataset_impl

    expected_private_shims = (
        "_dataset_preset_summary",
        "_dataset_preset_groups_for_ui",
        "_is_dataset_group_for_ui",
        "_dataset_summary_from_rows",
    )
    for name in expected_private_shims:
        exported = getattr(legacy_config, name)
        assert name not in dataset_impl.__all__
        assert name in legacy_config._DATASET_SHIM_NAMES
        assert (
            exported.__doc__
            == f"Compatibility shim forwarding to web.services.config.datasets.{name}."
        )

    summary = legacy_config._dataset_summary_from_rows(
        [{"source_dir": "image_dataset/a", "num_repeats": 2}],
        {"resolution": 512, "batch_size": 1},
    )
    assert summary["dataset_count"] == 1
    assert summary["repeat_total"] == 2


def test_legacy_dataset_path_and_row_helpers_forward_to_split_module(monkeypatch):
    from web.services.config import datasets as dataset_impl

    def sentinel(name: str):
        def impl(*args, **kwargs):
            return {"name": name, "args": args, "kwargs": kwargs}

        return impl

    helper_args = {
        "_is_allowed_dataset_config_path": (Path("configs/datasets/a.toml"),),
        "_dataset_config_rel_path": ({}, "lora", "gui-methods"),
        "_training_config_rel_path": ("lora", "gui-methods"),
        "_single_dataset_config_from_cfg": ({"source_image_dir": "image_dataset/a"},),
        "_dataset_defaults_from_config": ({"datasets": []},),
        "_dataset_defaults_from_dataset": ({"subsets": []}, {}),
        "_normalize_dataset_row_settings": ({"resolution": 512},),
        "_fill_missing_dataset_row_settings": ([{"source_dir": "image_dataset/a"}], {"resolution": 512}),
        "_normalize_preprocess_dataset_settings": ({"resolution": 512},),
        "_trigger_clone_should_persist": ({"enabled": True},),
        "_nl_tag_mix_enabled": ({"nl_tag_mix": {"enabled": True}},),
        "_preprocess_settings_from_custom_attributes": ({"preprocess": {"resolution": 512}},),
        "_preprocess_settings_for_runtime_attrs": ({"resolution": 512},),
        "_dataset_row_settings": ({"settings": {"resolution": 512}}, {"resolution": 768}),
        "_first_dataset_settings": ([{"settings": {"resolution": 512}}],),
        "_first_dataset_value": ({"datasets": [{"resolution": 512}]}, "resolution", 1024),
        "_dataset_path_value": ("{source_image_dir}/a", {"source_image_dir": "image_dataset"}),
    }
    for name in helper_args:
        monkeypatch.setattr(dataset_impl, name, sentinel(name))

    for name, args in helper_args.items():
        result = getattr(legacy_config, name)(*args)
        assert result["name"] == name


def test_legacy_dataset_preview_and_caption_helpers_forward_to_split_module(monkeypatch):
    from web.services.config import datasets as dataset_impl

    def sentinel(name: str):
        def impl(*args, **kwargs):
            return {"name": name, "args": args, "kwargs": kwargs}

        return impl

    source_dir = Path("image_dataset/a")
    image_file = source_dir / "hero.png"
    helper_args = {
        "_list_dataset_image_files": (source_dir, 8),
        "_dataset_image_preview_meta": (image_file,),
        "_dataset_image_dimensions": (image_file,),
        "_dataset_caption_meta": (image_file, ".txt", source_dir, source_dir),
        "_caption_source_mode_label": ("auto",),
        "_caption_extension_for_detected_mode": ("auto", ".txt"),
        "_format_caption_preview_text": (["a", "b"],),
        "_dataset_caption_detection_summary": ([{"caption": {"ok": True, "detected_mode": "txt"}}],),
        "_caption_detection_counts_text": ({"txt": 1}, 1),
        "_dataset_preview_empty_message": (source_dir, "source"),
        "_safe_file_stem": ("my dataset.toml",),
        "_dataset_image_files": (source_dir, {".png"}),
        "_count_images": (source_dir, {".png"}),
        "_count_source_images": (source_dir, {".png"}),
        "_dataset_num_repeats": ({"dataset_config": "configs/datasets/a.toml"},),
        "_nl_tag_mix_available_count": (source_dir, {".png"}),
        "_nl_tag_mix_caption_path_and_text": (image_file,),
        "_nl_tag_mix_caption_counts": (source_dir,),
    }
    for name in helper_args:
        monkeypatch.setattr(dataset_impl, name, sentinel(name))

    kwargs_by_name = {
        "_dataset_image_preview_meta": {
            "preset_file": "configs/datasets/a.toml",
            "dataset_index": 0,
            "source": "training",
            "caption_extension": ".txt",
            "prefer_json_caption": False,
            "caption_source_mode": "auto",
            "source_dir": source_dir,
            "train_dir": source_dir,
        }
    }
    for name, args in helper_args.items():
        result = getattr(legacy_config, name)(*args, **kwargs_by_name.get(name, {}))
        assert result["name"] == name


def test_dataset_preset_save_read_apply_preserves_regularization_fields(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)

    saved = config_service.save_dataset_preset(
        "configs/datasets/regularized.toml",
        [
            {
                "source_dir": "image_dataset/train",
                "image_dir": "post_image_dataset/train_resized",
                "cache_dir": "post_image_dataset/train_cache",
                "num_repeats": 2,
            },
            {
                "source_dir": "image_dataset/reg",
                "image_dir": "post_image_dataset/reg_resized",
                "cache_dir": "post_image_dataset/reg_cache",
                "num_repeats": 1,
                "is_reg": True,
                "settings": {"prior_loss_weight": 2.5},
            },
        ],
        {"resolution": 768, "batch_size": 1, "prior_loss_weight": 2.5},
    )

    data = toml.loads(saved["content"])
    assert data["datasets"][0]["prior_loss_weight"] == 2.5
    assert data["datasets"][1]["prior_loss_weight"] == 2.5
    assert "is_reg" not in data["datasets"][0]["subsets"][0]
    assert data["datasets"][1]["subsets"][0]["is_reg"] is True

    loaded = config_service.load_dataset_preset("configs/datasets/regularized.toml")
    assert loaded["defaults"]["prior_loss_weight"] == 2.5
    assert loaded["datasets"][0]["is_reg"] is False
    assert loaded["datasets"][1]["is_reg"] is True
    assert loaded["datasets"][1]["settings"]["prior_loss_weight"] == 2.5
    assert loaded["summary"]["train_dataset_count"] == 1
    assert loaded["summary"]["reg_dataset_count"] == 1

    applied = config_service.apply_dataset_preset_to_training_config(
        "configs/datasets/regularized.toml",
        "configs/imported/lora.toml",
    )
    assert applied["ok"] is True
    assert applied["values"]["source_image_dir"] == "image_dataset/train"
    assert applied["values"]["prior_loss_weight"] == 2.5
    train_text = (configs / "imported" / "lora.toml").read_text(encoding="utf-8")
    assert 'dataset_config = "configs/datasets/regularized.toml"' in train_text
    assert 'source_image_dir = "image_dataset/train"' in train_text
    assert 'resized_image_dir = "post_image_dataset/train_resized"' in train_text
    assert 'lora_cache_dir = "post_image_dataset/train_cache"' in train_text
    assert "prior_loss_weight = 2.5" in train_text


def test_dataset_preset_apply_uses_regularization_row_weight(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)

    config_service.save_dataset_preset(
        "configs/datasets/regularized-row-weight.toml",
        [
            {
                "source_dir": "image_dataset/train",
                "image_dir": "post_image_dataset/train_resized",
                "cache_dir": "post_image_dataset/train_cache",
                "num_repeats": 2,
                "settings": {"prior_loss_weight": 1.0},
            },
            {
                "source_dir": "image_dataset/reg",
                "image_dir": "post_image_dataset/reg_resized",
                "cache_dir": "post_image_dataset/reg_cache",
                "num_repeats": 1,
                "is_reg": True,
                "settings": {"prior_loss_weight": 2.5},
            },
        ],
        {"resolution": 768, "batch_size": 1, "prior_loss_weight": 1.0},
    )

    applied = config_service.apply_dataset_preset_to_training_config(
        "configs/datasets/regularized-row-weight.toml",
        "configs/imported/lora.toml",
    )

    assert applied["ok"] is True
    assert applied["defaults"]["prior_loss_weight"] == 1.0
    assert applied["values"]["prior_loss_weight"] == 2.5
    train_text = (configs / "imported" / "lora.toml").read_text(encoding="utf-8")
    assert "prior_loss_weight = 2.5" in train_text


def test_dataset_preset_diagnose_reports_scan_context(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    config_service.save_dataset_preset(
        "configs/datasets/character_a.toml",
        [{"source_dir": "image_dataset/a", "image_dir": "post/a", "cache_dir": "cache/a", "num_repeats": 4}],
        {"resolution": 768},
    )

    report = config_service.diagnose_dataset_presets("configs/datasets/character_a.toml")

    assert report["ok"] is True
    assert report["root"] == str(tmp_path.resolve())
    assert report["absolute_dataset_dir"] == str((configs / "datasets").resolve())
    assert report["file_count"] == 1
    assert report["listed_count"] == 1
    assert report["target"] == "configs/datasets/character_a.toml"
    assert report["groups"][0]["files"] == ["configs/datasets/character_a.toml"]
    assert report["files"][0]["selected"] is True
    assert report["files"][0]["summary"]["repeat_total"] == 4

    response = asyncio.run(config_routes.handle_dataset_presets_diagnose(_QueryRequest({
        "file": "configs/datasets/character_a.toml",
    })))
    assert response.status == 200
    body = json.loads(response.text)
    assert body["target"] == "configs/datasets/character_a.toml"


def test_dataset_preset_listing_normalizes_windows_display_paths(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    config_service.save_dataset_preset(
        "configs/datasets/hikarucs.toml",
        [{"source_dir": "train_set/hikaru", "image_dir": "train_set/hikaru_resized", "cache_dir": "train_set/hikaru_cache", "num_repeats": 6}],
        {},
    )

    original_display_path = config_service._display_path

    def windows_display_path(path: Path) -> str:
        text = original_display_path(path)
        if text.startswith("configs/datasets/"):
            return text.replace("/", "\\")
        return text

    monkeypatch.setattr(config_service, "_display_path", windows_display_path)

    listed = config_service.list_dataset_presets()
    assert listed["ok"] is True
    assert [item["path"] for item in listed["presets"]] == ["configs/datasets/hikarucs.toml"]
    assert listed["groups"][0]["files"][0]["path"] == "configs/datasets/hikarucs.toml"

    report = config_service.diagnose_dataset_presets("configs/datasets/hikarucs.toml")
    assert report["listed_count"] == 1
    assert report["files"][0]["path"] == "configs/datasets/hikarucs.toml"
    assert report["files"][0]["selected"] is True


def test_list_dataset_presets_reuses_dataset_group_meta(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    preset_path = configs / "datasets" / "character.toml"
    preset_path.write_text(
        "\n".join(
            [
                "[[datasets]]",
                "resolution = 1024",
                "batch_size = 1",
                "",
                "[[datasets.subsets]]",
                'image_dir = "post/character"',
                'cache_dir = "cache/character"',
                'custom_attributes = { source_dir = "image_dataset/character" }',
                "num_repeats = 3",
            ]
        ),
        encoding="utf-8",
    )

    calls = {"groups": 0, "meta": 0}

    def fake_list_config_file_groups(kind=None):
        calls["groups"] += 1
        assert kind == "dataset"
        return [{
            "id": "datasets",
            "label": "数据集配置",
            "open": True,
            "locked": False,
            "group_locked": False,
            "user_group_locked": False,
            "system_locked": False,
            "lockable": False,
            "user_managed": False,
            "kind": "dataset",
            "renamable": False,
            "deletable": False,
            "movable": False,
            "trainable": False,
            "methods_subdir": "",
            "files": [{
                "path": "configs/datasets/character.toml",
                "label": "character.toml",
                "filename": "character.toml",
                "group": "datasets",
                "group_label": "数据集配置",
                "locked": False,
                "group_locked": False,
                "user_group_locked": False,
                "system_locked": False,
                "user_locked": False,
                "lock_reason": "",
                "lock_reason_label": "",
                "restorable": False,
                "open": True,
                "trainable": False,
                "method": "character",
                "methods_subdir": "",
            }],
        }]

    def fake_get_config_file_meta(rel_path, *args, **kwargs):
        calls["meta"] += 1
        raise AssertionError(f"unexpected meta lookup for {rel_path}")

    monkeypatch.setattr(config_service, "list_config_file_groups", fake_list_config_file_groups)
    monkeypatch.setattr(config_service, "get_config_file_meta", fake_get_config_file_meta)

    listed = config_service.list_dataset_presets()

    assert listed["ok"] is True
    assert calls == {"groups": 1, "meta": 0}
    assert [item["path"] for item in listed["presets"]] == ["configs/datasets/character.toml"]
    assert listed["presets"][0]["readonly"] is False
    assert listed["groups"][0]["files"][0]["path"] == "configs/datasets/character.toml"
    assert listed["groups"][0]["files"][0]["summary"]["repeat_total"] == 3


def test_import_dataset_preset_parses_and_normalizes_uploaded_toml(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    content = "\n".join(
        [
            "[general]",
            "keep_tokens = 4",
            "",
            "[[datasets]]",
            "resolution = 768",
            "",
            "[[datasets.subsets]]",
            'image_dir = "post_image_dataset/a_resized"',
            'cache_dir = "post_image_dataset/a_cache"',
            "num_repeats = 6",
            "",
            "[datasets.subsets.custom_attributes]",
            'source_dir = "image_dataset/a"',
            "",
        ]
    )

    imported = config_service.import_dataset_preset("character a.toml", content)

    assert imported["ok"] is True
    assert imported["file"] == "configs/datasets/character_a.toml"
    assert imported["summary"]["repeat_total"] == 6
    loaded = config_service.load_dataset_preset(imported["file"])
    assert loaded["defaults"]["resolution"] == 768
    assert loaded["defaults"]["keep_tokens"] == 4
    assert loaded["datasets"][0]["source_dir"] == "image_dataset/a"

    response = asyncio.run(config_routes.handle_dataset_preset_import(_JsonRequest({
        "name": "character b.toml",
        "content": content.replace("image_dataset/a", "image_dataset/b"),
    })))
    assert response.status == 200
    body = json.loads(response.text)
    assert body["file"] == "configs/datasets/character_b.toml"


def test_import_dataset_preset_rejects_existing_target(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    existing = configs / "datasets" / "character_a.toml"
    existing.write_text(
        'source_image_dir = "keep"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="已存在"):
        config_service.import_dataset_preset(
            "character_a.toml",
            'source_image_dir = "image_dataset/a"\n',
        )

    assert existing.read_text(encoding="utf-8") == 'source_image_dir = "keep"\n'


def test_save_dataset_preset_logs_terminal_context(tmp_path: Path, monkeypatch, caplog):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)

    with caplog.at_level(logging.INFO, logger="web.services.config_service"):
        config_service.save_dataset_preset(
            "configs/datasets/character_a.toml",
            [{"source_dir": "image_dataset/a", "image_dir": "post/a", "cache_dir": "cache/a", "num_repeats": 1}],
            {},
        )

    assert any(
        "saved dataset preset file=configs/datasets/character_a.toml" in record.message
        and f"root={tmp_path.resolve()}" in record.message
        for record in caplog.records
    )


def test_dataset_preset_put_overwrite_false_preserves_existing_file(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    preset_path = configs / "datasets" / "character_a.toml"
    original = "# keep existing\n[[datasets]]\nresolution = 512\n"
    preset_path.write_text(original, encoding="utf-8")

    response = asyncio.run(config_routes.handle_dataset_preset_put(_JsonRequest({
        "file": "configs/datasets/character_a.toml",
        "overwrite": False,
        "datasets": [{
            "source_dir": "image_dataset/new",
            "image_dir": "post_image_dataset/new_resized",
            "cache_dir": "post_image_dataset/new_cache",
            "num_repeats": 1,
        }],
    })))

    assert response.status == 400
    body = json.loads(response.text)
    assert "已存在" in body["error"]
    assert preset_path.read_text(encoding="utf-8") == original


def test_dataset_preset_save_preserves_explicit_training_and_cache_dirs(tmp_path: Path, monkeypatch):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)

    saved = config_service.save_dataset_preset(
        "configs/datasets/preserve_paths.toml",
        [{
            "source_dir": "image_dataset/source",
            "image_dir": "output/runs/run-a/dataset_cache/dataset-01/resized",
            "cache_dir": "output/runs/run-a/dataset_cache/dataset-01/lora",
            "num_repeats": 4,
        }],
        {},
    )

    data = toml.loads(saved["content"])
    subset = data["datasets"][0]["subsets"][0]
    assert subset["image_dir"] == "output/runs/run-a/dataset_cache/dataset-01/resized"
    assert subset["cache_dir"] == "output/runs/run-a/dataset_cache/dataset-01/lora"
    assert subset["custom_attributes"]["source_dir"] == "image_dataset/source"


def test_dataset_preset_save_read_preserves_nl_tag_mix(tmp_path: Path, monkeypatch):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)

    saved = config_service.save_dataset_preset(
        "configs/datasets/mix.toml",
        [{
            "source_dir": "image_dataset/mixed",
            "image_dir": "post_image_dataset/mixed_resized",
            "cache_dir": "post_image_dataset/mixed_cache",
            "num_repeats": 2,
            "nl_tag_mix": {"enabled": True, "tag_ratio": 70},
        }],
        {},
    )

    data = toml.loads(saved["content"])
    attrs = data["datasets"][0]["subsets"][0]["custom_attributes"]
    assert attrs["nl_tag_mix"] == {"enabled": True, "tag_ratio": 0.7}

    loaded = config_service.load_dataset_preset("configs/datasets/mix.toml")
    assert loaded["datasets"][0]["nl_tag_mix"] == {"enabled": True, "tag_ratio": 0.7}


def test_dataset_preset_save_read_preserves_trigger_clone(tmp_path: Path, monkeypatch):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)

    saved = config_service.save_dataset_preset(
        "configs/datasets/trigger_clone.toml",
        [{
            "source_dir": "image_dataset/character",
            "image_dir": "post_image_dataset/character_resized",
            "cache_dir": "post_image_dataset/character_cache",
            "num_repeats": 2,
            "trigger_clone": {"enabled": True, "prompt": "my_character", "num_repeats": 3},
        }],
        {},
    )

    data = toml.loads(saved["content"])
    attrs = data["datasets"][0]["subsets"][0]["custom_attributes"]
    assert attrs["trigger_clone"] == {
        "enabled": True,
        "prompt": "my_character",
        "num_repeats": 3,
    }

    loaded = config_service.load_dataset_preset("configs/datasets/trigger_clone.toml")
    assert loaded["datasets"][0]["trigger_clone"] == {
        "enabled": True,
        "prompt": "my_character",
        "num_repeats": 3,
    }


def test_dataset_preset_save_read_preserves_subset_filter_and_zero_validation_split(
    tmp_path: Path,
    monkeypatch,
):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)

    saved = config_service.save_dataset_preset(
        "configs/datasets/filtered.toml",
        [{
            "source_dir": "image_dataset/filtered",
            "image_dir": "post_image_dataset/filtered_resized",
            "cache_dir": "post_image_dataset/filtered_cache",
            "num_repeats": 1,
            "recursive": False,
            "path_pattern": "char_a/*",
            "settings": {
                "validation_split": 0,
                "validation_seed": 0,
            },
        }],
        {},
    )

    data = toml.loads(saved["content"])
    dataset = data["datasets"][0]
    subset = dataset["subsets"][0]
    assert dataset["validation_split"] == 0
    assert dataset["validation_seed"] == 0
    assert subset["recursive"] is False
    assert subset["path_pattern"] == "char_a/*"

    loaded = config_service.load_dataset_preset("configs/datasets/filtered.toml")
    assert loaded["datasets"][0]["recursive"] is False
    assert loaded["datasets"][0]["path_pattern"] == "char_a/*"
    assert loaded["datasets"][0]["settings"]["validation_split"] == 0
    assert loaded["datasets"][0]["settings"]["validation_seed"] == 0


def test_dataset_preset_fixed_validation_zero_defaults_to_validation_off(
    tmp_path: Path,
    monkeypatch,
):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)

    saved = config_service.save_dataset_preset(
        "configs/datasets/no-validation.toml",
        [{
            "source_dir": "image_dataset/no-validation",
            "image_dir": "post_image_dataset/no-validation-resized",
            "cache_dir": "post_image_dataset/no-validation-cache",
            "num_repeats": 1,
            "settings": {
                "validation_split_num": 0,
            },
        }],
        {},
    )

    data = toml.loads(saved["content"])
    dataset = data["datasets"][0]
    assert dataset["validation_split"] == 0
    assert "validation_split_num" not in dataset

    loaded = config_service.load_dataset_preset("configs/datasets/no-validation.toml")
    settings = loaded["datasets"][0]["settings"]
    assert settings["validation_split"] == 0
    assert settings["validation_split_num"] == 0


def test_dataset_rows_for_estimate_inherits_top_level_path_pattern(
    tmp_path: Path,
    monkeypatch,
):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)

    rows = config_service._dataset_rows_for_estimate(
        {
            "source_image_dir": "image_dataset/filtered",
            "resized_image_dir": "post_image_dataset/filtered_resized",
            "lora_cache_dir": "post_image_dataset/filtered_cache",
            "path_pattern": "char_a/*",
        }
    )

    assert rows[0]["path_pattern"] == "char_a/*"


def test_runtime_dataset_doc_can_prefer_train_batch_size():
    doc = config_service._build_dataset_config_doc(
        [{
            "source_dir": "image_dataset/source",
            "image_dir": "post_image_dataset/resized",
            "cache_dir": "post_image_dataset/lora",
            "num_repeats": 1,
            "settings": {"resolution": 1024, "batch_size": 1},
        }],
        {"train_batch_size": 2},
        prefer_train_batch_size=True,
    )

    data = toml.loads(doc)
    assert data["datasets"][0]["batch_size"] == 2


def test_runtime_dataset_doc_hides_preprocess_settings_from_training_schema():
    doc = config_service._build_dataset_config_doc(
        [{
            "source_dir": "image_dataset/source",
            "image_dir": "post_image_dataset/resized",
            "cache_dir": "post_image_dataset/lora",
            "num_repeats": 1,
            "settings": {
                "resolution": 768,
                "batch_size": 1,
                "enable_bucket": True,
                "min_bucket_reso": 256,
                "max_bucket_reso": 768,
                "bucket_reso_steps": 32,
                "bucket_no_upscale": True,
                "validation_split_num": 4,
            },
        }],
        {"train_batch_size": 2},
        prefer_train_batch_size=True,
        include_preprocess_settings=False,
    )

    data = toml.loads(doc)
    dataset = data["datasets"][0]
    for key in config_service.PREPROCESS_DATASET_SETTING_KEYS:
        assert key not in dataset
    assert dataset["batch_size"] == 2
    assert dataset["validation_split_num"] == 4

    attrs = dataset["subsets"][0]["custom_attributes"]
    assert attrs["preprocess"]["resolution"] == 768
    assert attrs["preprocess"]["bucket_reso_steps"] == 32
    assert attrs["preprocess"]["bucket_no_upscale"] is True

    from library.config.loader import ConfigSanitizer

    ConfigSanitizer(support_dropout=True).sanitize_user_config(data)


def test_system_dataset_preset_is_readonly_but_can_be_saved_as(tmp_path: Path, monkeypatch):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    system_preset = tmp_path / "configs" / "datasets" / "ip_adapter.toml"
    hidden_preset = tmp_path / "configs" / "datasets" / "easycontrol.toml"
    visible_system_preset = tmp_path / "configs" / "datasets" / "visible_system.toml"
    system_preset.write_text(
        "\n".join(
            [
                "[[datasets]]",
                "resolution = 1024",
                "batch_size = 1",
                "",
                "[[datasets.subsets]]",
                'image_dir = "ip-adapter-dataset"',
                'cache_dir = "post_image_dataset/ip_adapter"',
                "num_repeats = 1",
                'custom_attributes = {source_dir = "ip-adapter-dataset"}',
            ]
        ),
        encoding="utf-8",
    )
    hidden_preset.write_text(system_preset.read_text(encoding="utf-8"), encoding="utf-8")
    visible_system_preset.write_text(system_preset.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(
        config_datasets,
        "SYSTEM_DATASET_PRESET_FILES",
        frozenset({"configs/datasets/visible_system.toml"}),
    )

    with pytest.raises(ValueError, match="只读"):
        config_service.save_dataset_preset(
            "configs/datasets/ip_adapter.toml",
            [{"source_dir": "x", "image_dir": "y", "cache_dir": "z", "num_repeats": 1}],
            {},
        )

    with pytest.raises(ValueError, match="不能删除"):
        config_service.delete_dataset_preset("configs/datasets/ip_adapter.toml")

    copied = config_service.save_dataset_preset_as(
        "ip_adapter_copy",
        [{"source_dir": "x", "image_dir": "y", "cache_dir": "z", "num_repeats": 1}],
        {},
    )
    assert copied["file"] == "configs/datasets/ip_adapter_copy.toml"

    listed = config_service.list_dataset_presets()
    paths = [item["path"] for item in listed["presets"]]
    assert "configs/datasets/ip_adapter.toml" not in paths
    assert "configs/datasets/easycontrol.toml" not in paths
    system_item = next(item for item in listed["presets"] if item["path"] == "configs/datasets/visible_system.toml")
    assert system_item["readonly"] is True
    assert system_item["system_preset"] is True
    assert system_item["summary"]["dataset_count"] == 1
    assert listed["groups"][0]["files"][0]["summary"]["dataset_count"] == 1

    report = config_service.diagnose_dataset_presets()
    assert report["hidden_count"] == 2


def test_step_estimate_empty_dataset_config_override_clears_preset(tmp_path: Path, monkeypatch):
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    _write_step_estimate_dataset(tmp_path, dataset_path)
    base_resized = tmp_path / "image_dataset_resized"
    base_resized.mkdir(parents=True)
    for idx in range(2):
        Image.new("RGB", (8, 8), color=(idx, 80, 120)).save(base_resized / f"base-{idx}.png")

    with_dataset = config_service.estimate_training_steps("lora", "default", "imported")
    cleared = config_service.estimate_training_steps("lora", "default", "imported", dataset_config="")

    assert with_dataset["steps_per_epoch"] == 15
    assert with_dataset["dataset_num_repeats"] == 5
    assert cleared["steps_per_epoch"] == 2
    assert cleared["dataset_num_repeats"] == 1
    assert cleared["resized_dir"].endswith("image_dataset_resized")


def test_step_estimate_uses_selected_training_config_dataset(tmp_path: Path, monkeypatch):
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    _write_step_estimate_dataset(tmp_path, dataset_path)
    selected_image_dir = tmp_path / "post_image_dataset" / "selected_resized"
    selected_image_dir.mkdir(parents=True)
    for idx in range(4):
        Image.new("RGB", (8, 8), color=(idx, 90, 120)).save(selected_image_dir / f"selected-{idx}.png")
    selected_dataset_path = configs / "datasets" / "selected.toml"
    selected_dataset_path.write_text(
        "\n".join(
            [
                "[[datasets]]",
                "",
                "[[datasets.subsets]]",
                'image_dir = "post_image_dataset/selected_resized"',
                "num_repeats = 2",
            ]
        ),
        encoding="utf-8",
    )
    (configs / "imported" / "selected.toml").write_text(
        'dataset_config = "configs/datasets/selected.toml"\n',
        encoding="utf-8",
    )

    default_estimate = config_service.estimate_training_steps(
        "lora",
        "default",
        "imported",
    )
    selected_estimate = config_service.estimate_training_steps(
        "lora",
        "default",
        "imported",
        config_file="configs/imported/selected.toml",
    )

    assert default_estimate["steps_per_epoch"] == 15
    assert selected_estimate["steps_per_epoch"] == 8
    assert selected_estimate["dataset_num_repeats"] == 2
    assert selected_estimate["resized_dir"].endswith("post_image_dataset/selected_resized")


def test_step_estimate_resolves_training_dataset_under_external_configs_root(
    tmp_path: Path,
    monkeypatch,
):
    root = tmp_path / "repo"
    configs = tmp_path / "external-configs"
    root.mkdir()
    _write_minimal_config_tree(root)
    _patch_external_config_service_paths(monkeypatch, root, configs)

    image_dir = root / "post_image_dataset" / "selected_resized"
    image_dir.mkdir(parents=True)
    for idx in range(4):
        Image.new("RGB", (8, 8), color=(idx, 90, 120)).save(image_dir / f"selected-{idx}.png")
    (configs / "datasets" / "selected.toml").write_text(
        "\n".join(
            [
                "[[datasets]]",
                "",
                "[[datasets.subsets]]",
                'image_dir = "post_image_dataset/selected_resized"',
                "num_repeats = 5",
            ]
        ),
        encoding="utf-8",
    )
    (configs / "imported" / "selected.toml").write_text(
        "\n".join(
            [
                'dataset_config = "configs/datasets/selected.toml"',
                "max_train_epochs = 2",
            ]
        ),
        encoding="utf-8",
    )

    estimate = config_service.estimate_training_steps(
        "lora",
        "default",
        "imported",
        config_file="configs/imported/selected.toml",
    )

    assert estimate["dataset_num_repeats"] == 5
    assert estimate["weighted_image_count"] == 20
    assert estimate["steps_per_epoch"] == 20
    assert estimate["total_steps"] == 40


def test_dataset_preset_writes_independent_dataset_settings_per_path(tmp_path: Path, monkeypatch):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)

    saved = config_service.save_dataset_preset(
        "configs/datasets/multi_bucket.toml",
        [
            {
                "source_dir": "image_dataset/a",
                "image_dir": "post_image_dataset/a_resized",
                "cache_dir": "post_image_dataset/a_cache",
                "num_repeats": 2,
                "settings": {
                    "resolution": 768,
                    "min_bucket_reso": 256,
                    "max_bucket_reso": 768,
                    "bucket_reso_steps": 32,
                    "bucket_no_upscale": True,
                    "validation_split_num": 4,
                    "validation_seed": 7,
                    "caption_extension": ".caption",
                    "caption_source_mode": "json",
                },
            },
            {
                "source_dir": "image_dataset/b",
                "image_dir": "post_image_dataset/b_resized",
                "cache_dir": "post_image_dataset/b_cache",
                "num_repeats": 5,
                "settings": {
                    "resolution": 1024,
                    "min_bucket_reso": 384,
                    "max_bucket_reso": 1344,
                    "bucket_reso_steps": 64,
                    "bucket_no_upscale": False,
                    "validation_split": 0.1,
                    "validation_seed": 99,
                    "caption_source_mode": "auto",
                },
            },
        ],
        {"caption_extension": ".txt", "keep_tokens": 2},
    )

    data = toml.loads(saved["content"])
    assert "prefer_json_caption" not in data["general"]
    assert "caption_source_mode" not in data["general"]
    assert len(data["datasets"]) == 2
    assert data["datasets"][0]["resolution"] == 768
    assert data["datasets"][0]["caption_source_mode"] == "json"
    assert data["datasets"][0]["caption_extension"] == ".caption"
    assert data["datasets"][0]["max_bucket_reso"] == 768
    assert data["datasets"][0]["bucket_reso_steps"] == 32
    assert data["datasets"][0]["bucket_no_upscale"] is True
    assert data["datasets"][0]["validation_split_num"] == 4
    assert data["datasets"][1]["resolution"] == 1024
    assert data["datasets"][1]["caption_source_mode"] == "auto"
    assert data["datasets"][1]["min_bucket_reso"] == 384
    assert data["datasets"][1]["max_bucket_reso"] == 1344
    assert data["datasets"][1]["validation_split"] == 0.1

    loaded = config_service.load_dataset_preset("configs/datasets/multi_bucket.toml")
    assert loaded["datasets"][0]["settings"]["resolution"] == 768
    assert loaded["datasets"][0]["settings"]["caption_source_mode"] == "json"
    assert loaded["datasets"][0]["settings"]["caption_extension"] == ".caption"
    assert loaded["datasets"][1]["settings"]["max_bucket_reso"] == 1344
    assert loaded["datasets"][1]["settings"]["caption_source_mode"] == "auto"


def test_dataset_preset_load_migrates_legacy_general_prefer_json_caption(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    preset_path = configs / "datasets" / "legacy_json.toml"
    preset_path.write_text(
        "\n".join(
            [
                "[general]",
                'caption_extension = ".txt"',
                "keep_tokens = 2",
                "prefer_json_caption = true",
                "",
                "[[datasets]]",
                "resolution = 768",
                "batch_size = 1",
                "",
                "[[datasets.subsets]]",
                'image_dir = "post_image_dataset/a_resized"',
                'cache_dir = "post_image_dataset/a_cache"',
                'custom_attributes = {source_dir = "image_dataset/a"}',
                "",
                "[[datasets]]",
                "resolution = 1024",
                "batch_size = 1",
                "",
                "[[datasets.subsets]]",
                'image_dir = "post_image_dataset/b_resized"',
                'cache_dir = "post_image_dataset/b_cache"',
                'custom_attributes = {source_dir = "image_dataset/b"}',
            ]
        ),
        encoding="utf-8",
    )

    loaded = config_service.load_dataset_preset("configs/datasets/legacy_json.toml")

    assert loaded["defaults"]["prefer_json_caption"] is True
    assert loaded["defaults"]["caption_source_mode"] == "json"
    assert [row["settings"]["prefer_json_caption"] for row in loaded["datasets"]] == [True, True]
    assert [row["settings"]["caption_source_mode"] for row in loaded["datasets"]] == ["json", "json"]


def test_dataset_preset_image_preview_reads_training_images_and_captions(tmp_path: Path, monkeypatch):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    source_dir = tmp_path / "image_dataset" / "a"
    image_dir = tmp_path / "post_image_dataset" / "a_resized"
    source_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    Image.new("RGB", (8, 6), color=(120, 20, 40)).save(image_dir / "hero.png")
    (source_dir / "hero.txt").write_text("1girl, blue eyes", encoding="utf-8")

    config_service.save_dataset_preset(
        "configs/datasets/preview.toml",
        [{
            "source_dir": "image_dataset/a",
            "image_dir": "post_image_dataset/a_resized",
            "cache_dir": "post_image_dataset/a_cache",
            "num_repeats": 2,
        }],
        {"caption_extension": ".txt", "keep_tokens": 1},
    )

    listing = config_service.list_dataset_preset_images("configs/datasets/preview.toml", 0)

    assert listing["ok"] is True
    assert listing["total"] == 1
    assert listing["images"][0]["name"] == "hero.png"
    assert listing["images"][0]["width"] == 8
    assert listing["images"][0]["height"] == 6
    assert listing["images"][0]["total_pixels"] == 48
    assert listing["images"][0]["caption"]["ok"] is True
    assert listing["images"][0]["caption"]["text"] == "1girl, blue eyes"
    assert "dataset_index=0" in listing["images"][0]["url"]


def test_dataset_preset_image_preview_prefers_json_caption(tmp_path: Path, monkeypatch):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    source_dir = tmp_path / "image_dataset" / "a"
    image_dir = tmp_path / "post_image_dataset" / "a_resized"
    source_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    Image.new("RGB", (8, 6), color=(120, 20, 40)).save(image_dir / "hero.png")
    (source_dir / "hero.txt").write_text("txt fallback", encoding="utf-8")
    (source_dir / "hero.json").write_text(
        json.dumps(
            {
                "quality": "newest, safe",
                "count": "1girl",
                "artist": "@artist",
                "appearance": ["blue eyes"],
                "tags": ["looking at viewer"],
                "environment": ["sky"],
                "nl": "A fixed tail.",
            }
        ),
        encoding="utf-8",
    )

    config_service.save_dataset_preset(
        "configs/datasets/preview_json.toml",
        [{
            "source_dir": "image_dataset/a",
            "image_dir": "post_image_dataset/a_resized",
            "cache_dir": "post_image_dataset/a_cache",
            "num_repeats": 2,
        }],
        {"caption_extension": ".txt", "keep_tokens": 1, "prefer_json_caption": True},
    )

    listing = config_service.list_dataset_preset_images("configs/datasets/preview_json.toml", 0)

    caption = listing["images"][0]["caption"]
    assert listing["prefer_json_caption"] is True
    assert listing["caption_source_mode"] == "json"
    assert caption["ok"] is True
    assert caption["extension"] == ".json"
    assert caption["detected_mode"] == "json"
    assert caption["format_label"] == "AnimaLoraToolkit .json"
    assert caption["text"] == (
        "newest, safe, 1girl, @artist, blue eyes, looking at viewer, sky. "
        "A fixed tail."
    )


def test_dataset_preset_image_preview_auto_detects_captions_json(tmp_path: Path, monkeypatch):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    source_dir = tmp_path / "image_dataset" / "a"
    image_dir = tmp_path / "post_image_dataset" / "a_resized"
    source_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    Image.new("RGB", (8, 6), color=(120, 20, 40)).save(source_dir / "hero.png")
    Image.new("RGB", (8, 6), color=(120, 20, 40)).save(image_dir / "hero.png")
    (source_dir / "hero.txt").write_text("txt fallback", encoding="utf-8")
    (source_dir / "captions.json").write_text(
        json.dumps({"hero.png": ["caption one", "caption two"]}),
        encoding="utf-8",
    )

    config_service.save_dataset_preset(
        "configs/datasets/preview_captions_json.toml",
        [{
            "source_dir": "image_dataset/a",
            "image_dir": "post_image_dataset/a_resized",
            "cache_dir": "post_image_dataset/a_cache",
            "num_repeats": 2,
        }],
        {"caption_extension": ".txt", "keep_tokens": 1, "caption_source_mode": "auto"},
    )

    listing = config_service.list_dataset_preset_images("configs/datasets/preview_captions_json.toml", 0)

    caption = listing["images"][0]["caption"]
    assert listing["caption_source_mode"] == "auto"
    assert "DiffPipeForge captions.json 1 张" in listing["caption_summary"]
    assert "共 2 条标注" in listing["caption_summary"]
    assert caption["ok"] is True
    assert caption["extension"] == "captions.json"
    assert caption["detected_mode"] == "captions_json"
    assert caption["caption_count"] == 2
    assert caption["text"] == "1. caption one\n2. caption two"


def test_dataset_preview_image_resolver_rejects_files_outside_selected_row(tmp_path: Path, monkeypatch):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    image_dir = tmp_path / "post_image_dataset" / "a_resized"
    other_dir = tmp_path / "post_image_dataset" / "other"
    image_dir.mkdir(parents=True)
    other_dir.mkdir(parents=True)
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(image_dir / "ok.png")
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(other_dir / "bad.png")
    config_service.save_dataset_preset(
        "configs/datasets/preview_guard.toml",
        [{
            "source_dir": "image_dataset/a",
            "image_dir": "post_image_dataset/a_resized",
            "cache_dir": "post_image_dataset/a_cache",
            "num_repeats": 1,
        }],
        {},
    )

    resolved = config_service.resolve_dataset_preview_image(
        "configs/datasets/preview_guard.toml",
        0,
        "post_image_dataset/a_resized/ok.png",
    )
    assert resolved.name == "ok.png"

    with pytest.raises(ValueError, match="不属于当前数据集路径"):
        config_service.resolve_dataset_preview_image(
            "configs/datasets/preview_guard.toml",
            0,
            "post_image_dataset/other/bad.png",
        )


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



def test_dataset_config_doc_roundtrips_stage_schedule():
    """数据集 TOML 应能保存并读回 stage_schedule*。"""
    doc = config_service._build_dataset_config_doc(
        [{
            "source_dir": "image_dataset/source",
            "image_dir": "post_image_dataset/resized",
            "cache_dir": "post_image_dataset/lora",
            "num_repeats": 1,
            "settings": {"resolution": 1024, "batch_size": 1},
        }],
        {
            "caption_extension": ".txt",
            "keep_tokens": 3,
            "stage_schedule_enabled": True,
            "stage_schedule": [
                {"name": "low", "subset_index": 0, "start_pct": 0.0, "end_pct": 0.4},
                {"name": "high", "subset_index": 0, "start_pct": 0.4, "end_pct": 1.0},
            ],
        },
    )
    data = toml.loads(doc)
    assert data["stage_schedule_enabled"] is True
    assert len(data["stage_schedule"]) == 2
    assert data["stage_schedule"][0]["name"] == "low"
    assert data["stage_schedule"][1]["end_pct"] == 1.0


def test_save_dataset_preset_roundtrips_stage_schedule(tmp_path: Path, monkeypatch):
    configs = tmp_path / "configs"
    (configs / "datasets").mkdir(parents=True)
    _patch_config_service_paths(monkeypatch, tmp_path)

    saved = config_service.save_dataset_preset(
        "configs/datasets/stage_demo.toml",
        [{
            "source_dir": "image_dataset/source",
            "image_dir": "post_image_dataset/resized",
            "cache_dir": "post_image_dataset/lora",
            "num_repeats": 1,
            "settings": {"resolution": 1024},
        }],
        defaults={"resolution": 1024},
        overwrite=True,
        stage_schedule_enabled=True,
        stage_schedule=[
            {"name": "a", "subset_index": 0, "start_pct": 0, "end_pct": 0.5},
            {"name": "b", "subset_index": 0, "start_pct": 0.5, "end_pct": 1},
        ],
    )
    assert saved["ok"]
    assert saved["stage_schedule_enabled"] is True
    loaded = config_service.load_dataset_preset("configs/datasets/stage_demo.toml")
    assert loaded["stage_schedule_enabled"] is True
    assert len(loaded["stage_schedule"]) == 2
    text = (configs / "datasets" / "stage_demo.toml").read_text(encoding="utf-8")
    assert "stage_schedule_enabled = true" in text.lower()
    assert "stage_schedule" in text


def test_save_dataset_preset_rejects_regularization_with_stage_schedule(
    tmp_path: Path,
    monkeypatch,
):
    (tmp_path / "configs" / "datasets").mkdir(parents=True)
    _patch_config_service_paths(monkeypatch, tmp_path)
    rows = [
        {
            "source_dir": "image_dataset/train",
            "image_dir": "post_image_dataset/train",
            "cache_dir": "post_image_dataset/train-cache",
        },
        {
            "source_dir": "image_dataset/reg",
            "image_dir": "post_image_dataset/reg",
            "cache_dir": "post_image_dataset/reg-cache",
            "is_reg": True,
        },
    ]

    with pytest.raises(ValueError, match="分阶段调度暂不支持正则化数据集"):
        config_service.save_dataset_preset(
            "configs/datasets/reg-stage.toml",
            rows,
            overwrite=True,
            stage_schedule_enabled=True,
            stage_schedule=[
                {
                    "name": "train",
                    "subset_index": 0,
                    "start_pct": 0,
                    "end_pct": 1,
                }
            ],
        )


def test_dataset_preset_route_rejects_invalid_stage_schedule_without_overwrite(
    tmp_path: Path,
    monkeypatch,
):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    rel_path = "configs/datasets/stage_route.toml"
    rows = [{
        "source_dir": "image_dataset/source",
        "image_dir": "post_image_dataset/resized",
        "cache_dir": "post_image_dataset/lora",
        "num_repeats": 1,
    }]
    valid_response = asyncio.run(config_routes.handle_dataset_preset_put(_JsonRequest({
        "file": rel_path,
        "datasets": rows,
        "stage_schedule_enabled": True,
        "stage_schedule": [
            {"name": "first", "subset_index": 0, "start_pct": 0.0, "end_pct": 0.5},
            {"name": "second", "subset_index": 0, "start_pct": 0.5, "end_pct": 1.0},
        ],
    })))
    assert valid_response.status == 200
    preset_path = configs / "datasets" / "stage_route.toml"
    valid_content = preset_path.read_text(encoding="utf-8")

    invalid_response = asyncio.run(config_routes.handle_dataset_preset_put(_JsonRequest({
        "file": rel_path,
        "datasets": rows,
        "stage_schedule_enabled": True,
        "stage_schedule": [
            {"name": "gap-a", "subset_index": 0, "start_pct": 0.0, "end_pct": 0.4},
            {"name": "gap-b", "subset_index": 0, "start_pct": 0.6, "end_pct": 1.0},
        ],
    })))

    assert invalid_response.status == 400
    body = json.loads(invalid_response.text)
    assert "stage_schedule invalid" in body["error"]
    assert "未贴齐" in body["error"]
    assert preset_path.read_text(encoding="utf-8") == valid_content


def test_dataset_editor_loads_stage_schedule_from_linked_dataset_file(tmp_path: Path, monkeypatch):
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    dataset_path.write_text(
        "\n".join(
            [
                "stage_schedule_enabled = true",
                'stage_schedule = [{name = "first", subset_index = 0, start_pct = 0.0, end_pct = 1.0}]',
                "",
                "[[datasets]]",
                "resolution = 1024",
                "",
                "[[datasets.subsets]]",
                'image_dir = "post_image_dataset/resized"',
                'cache_dir = "post_image_dataset/lora"',
                'custom_attributes = {source_dir = "image_dataset/source"}',
            ]
        ),
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)

    loaded = config_service.load_dataset_editor(
        "lora",
        "default",
        "imported",
        config_file="configs/imported/lora.toml",
    )

    assert loaded["dataset_config"] == "configs/datasets/lora.toml"
    assert loaded["stage_schedule_enabled"] is True
    assert loaded["stage_schedule"][0]["name"] == "first"


def test_dataset_editor_save_preserves_dataset_owned_stage_schedule(tmp_path: Path, monkeypatch):
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    dataset_path.write_text(
        "\n".join(
            [
                "stage_schedule_enabled = true",
                'stage_schedule = [{name = "dataset", subset_index = 0, start_pct = 0.0, end_pct = 1.0}]',
                "",
                "[[datasets]]",
                "resolution = 1024",
                "",
                "[[datasets.subsets]]",
                'image_dir = "post_image_dataset/old_resized"',
                'cache_dir = "post_image_dataset/old_cache"',
                'custom_attributes = {source_dir = "image_dataset/old"}',
            ]
        ),
        encoding="utf-8",
    )
    (configs / "imported" / "lora.toml").write_text(
        "\n".join(
            [
                'dataset_config = "configs/datasets/lora.toml"',
                "stage_schedule_enabled = false",
                'stage_schedule = [{name = "legacy-training", subset_index = 0, start_pct = 0.0, end_pct = 1.0}]',
            ]
        ),
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)

    saved = config_service.save_dataset_editor(
        "lora",
        "default",
        "imported",
        [{
            "source_dir": "image_dataset/new",
            "image_dir": "post_image_dataset/new_resized",
            "cache_dir": "post_image_dataset/new_cache",
            "num_repeats": 1,
        }],
        train_file="configs/imported/lora.toml",
    )

    persisted = toml.loads(dataset_path.read_text(encoding="utf-8"))
    assert saved["stage_schedule_enabled"] is True
    assert saved["stage_schedule"][0]["name"] == "dataset"
    assert persisted["stage_schedule_enabled"] is True
    assert persisted["stage_schedule"][0]["name"] == "dataset"


def test_merge_stage_schedule_from_dataset_config_prefers_dataset_file(tmp_path: Path, monkeypatch):
    configs = tmp_path / "configs"
    (configs / "datasets").mkdir(parents=True)
    dataset_path = configs / "datasets" / "with_stage.toml"
    dataset_path.write_text(
        (
            "[general]\n"
            "caption_extension = \".txt\"\n"
            "keep_tokens = 3\n"
            "stage_schedule_enabled = true\n"
            "stage_schedule = [\n"
            "  { name = \"low\", subset_index = 0, start_pct = 0.0, end_pct = 0.3 },\n"
            "  { name = \"high\", subset_index = 0, start_pct = 0.3, end_pct = 1.0 },\n"
            "]\n\n"
            "[[datasets]]\n"
            "resolution = 1024\n"
            "batch_size = 1\n\n"
            "[[datasets.subsets]]\n"
            "image_dir = \"post_image_dataset/resized\"\n"
            "cache_dir = \"post_image_dataset/lora\"\n"
            "num_repeats = 1\n"
            "custom_attributes = {source_dir = \"image_dataset/source\"}\n"
        ),
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)
    from web.services.config.dataset_rows import merge_stage_schedule_from_dataset_config

    merged = merge_stage_schedule_from_dataset_config({
        "dataset_config": "configs/datasets/with_stage.toml",
        "stage_schedule_enabled": False,
        "stage_schedule": [{"name": "legacy", "subset_index": 0, "start_pct": 0, "end_pct": 1}],
    })
    assert merged["stage_schedule_enabled"] is True
    assert merged["stage_schedule"][0]["name"] == "low"
    assert merged["stage_schedule"][1]["start_pct"] == 0.3
