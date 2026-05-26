from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
import toml
from PIL import Image

from web.routes import config as config_routes
from web.services import config_service


def test_spd_cli_config_is_exposed_as_method_variant(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    (configs / "methods").mkdir()
    (configs / "methods" / "spd.toml").write_text(
        'output_name = "anima_spd"\niterations = 4000\n',
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)

    assert "spd" in config_service.list_methods()
    assert config_service.list_variants("spd") == ["spd"]


def test_web_variants_follow_variant_family_metadata(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    gui_methods = configs / "gui-methods"
    gui_methods.mkdir()
    (gui_methods / "lora.toml").write_text(
        '[variant]\nfamily = "lora"\norder = 10\n',
        encoding="utf-8",
    )
    (gui_methods / "lokr.toml").write_text(
        '[variant]\nfamily = "lora"\norder = 12\n',
        encoding="utf-8",
    )
    (gui_methods / "hydralora.toml").write_text(
        '[variant]\nfamily = "hydralora"\norder = 20\n',
        encoding="utf-8",
    )
    (gui_methods / "hydralora-8gb.toml").write_text(
        '[variant]\nfamily = "hydralora"\norder = 10\n',
        encoding="utf-8",
    )
    (gui_methods / "custom").mkdir()
    (gui_methods / "custom" / "user_variant.toml").write_text(
        'output_name = "user_variant"\n',
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)

    assert config_service.list_variants("lora") == ["lora", "lokr", "custom/user_variant"]
    assert config_service.list_variants("hydralora") == [
        "hydralora-8gb",
        "hydralora",
        "custom/user_variant",
    ]


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
        return original_save_raw_file(rel_path, content, **kwargs)

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


def test_raw_patch_ignores_dataset_picker_ui_field(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    train_rel = "configs/imported/lora.toml"

    ok, msg, content, changed = config_service.patch_raw_file_values(
        train_rel,
        {
            "dataset_config_picker": "configs/datasets/character_a.toml",
            "output_name": "clean",
        },
    )

    assert ok is True, msg
    assert changed == ["output_name"]
    assert 'output_name = "clean"' in content
    assert "dataset_config_picker" not in content
    assert "dataset_config_picker" not in (configs / "imported" / "lora.toml").read_text(encoding="utf-8")


def test_blank_preset_template_can_receive_global_model_paths(tmp_path: Path, monkeypatch):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    template = "\n".join(
        [
            'output_name = "anima"',
            'pretrained_model_name_or_path = "template-base.safetensors"',
            'qwen3 = "template-qwen.safetensors"',
            'vae = "template-vae.safetensors"',
        ]
    )

    ok, msg, content, changed = config_service.preview_raw_file_patch(
        "configs/imported/new_blank.toml",
        {
            "pretrained_model_name_or_path": "${ANIMA_DIT_MODEL}",
            "qwen3": "/models/qwen.safetensors",
            "vae": "models/custom_vae.safetensors",
        },
        content=template,
    )

    assert ok is True, msg
    assert changed == ["pretrained_model_name_or_path", "qwen3", "vae"]
    data = toml.loads(content)
    assert data["pretrained_model_name_or_path"] == "${ANIMA_DIT_MODEL}"
    assert data["qwen3"] == "/models/qwen.safetensors"
    assert data["vae"] == "models/custom_vae.safetensors"


def test_save_dataset_editor_accepts_source_only_rows(tmp_path: Path, monkeypatch):
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
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
    subset = data["datasets"][0]["subsets"][0]
    assert subset["custom_attributes"]["source_dir"] == "image_dataset/source_only"
    assert subset["image_dir"].endswith("source_only_resized")
    assert subset["cache_dir"].endswith("source_only_lora_cache")


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


def test_preflight_reports_missing_preprocess_environment_file(tmp_path: Path, monkeypatch):
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
    (tmp_path / "library" / "preprocess" / "__init__.py").unlink()

    result = config_service.preflight_training_config(
        "lora",
        "default",
        "imported",
        config_file="configs/imported/selected.toml",
    )

    assert result["ok"] is False
    errors = [item for item in result["errors"] if item["key"] == "preprocess_environment"]
    assert errors
    assert "预处理启动环境异常" in errors[-1]["message"]
    assert "library/preprocess/__init__.py" in errors[-1]["message"]


def test_preflight_ignores_legacy_cache_fields_for_plain_web_config(tmp_path: Path, monkeypatch):
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    source_dir = tmp_path / "image_dataset" / "selected"
    source_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(source_dir / "sample.png")
    bad_resized = tmp_path / "bad-resized-file"
    bad_cache = tmp_path / "bad-cache-file"
    bad_resized.write_text("not a dir", encoding="utf-8")
    bad_cache.write_text("not a dir", encoding="utf-8")
    selected_config = configs / "imported" / "selected.toml"
    selected_config.write_text(
        "\n".join(
            [
                'source_image_dir = "image_dataset/selected"',
                'resized_image_dir = "bad-resized-file"',
                'lora_cache_dir = "bad-cache-file"',
                'dataset_config = "configs/datasets/lora.toml"',
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
    dataset_path.write_text(
        "\n".join(
            [
                "[[datasets]]",
                "[[datasets.subsets]]",
                'image_dir = "bad-resized-file"',
                'cache_dir = "bad-cache-file"',
                "num_repeats = 1",
                'custom_attributes = { source_dir = "image_dataset/selected" }',
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

    keys = {item["key"] for item in result["checks"]}
    assert "resized_image_dir" not in keys
    assert "lora_cache_dir" not in keys
    assert not any(key.startswith("dataset_") and (key.endswith("_image_dir") or key.endswith("_cache_dir")) for key in keys)
    assert result["ok"] is True


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


def test_is_allowed_training_config_accepts_runtime_configs_under_output_root(tmp_path: Path, monkeypatch):
    _patch_config_service_paths(monkeypatch, tmp_path)
    output_root = tmp_path / "output" / "runs"
    runtime_config = output_root / "522-20260523-114514" / "config.runtime.toml"
    runtime_config.parent.mkdir(parents=True)
    (runtime_config.parent / "model_cache").mkdir()
    (runtime_config.parent / "dataset_cache").mkdir()
    (runtime_config.parent / "training_output").mkdir()
    runtime_config.write_text('output_dir = "output/runs/522-20260523-114514/training_output"\n', encoding="utf-8")
    monkeypatch.setattr(config_service, "resolve_output_root", lambda: output_root.resolve())

    path = config_service._config_file_path(str(runtime_config))

    assert path == runtime_config.resolve()


def test_output_run_training_config_rejects_non_runtime_snapshots(tmp_path: Path, monkeypatch):
    _patch_config_service_paths(monkeypatch, tmp_path)
    output_root = tmp_path / "output" / "runs"
    run_dir = output_root / "522-20260523-114514"
    run_dir.mkdir(parents=True)
    (run_dir / "config.original.toml").write_text('output_name = "original"\n', encoding="utf-8")
    (run_dir / "dataset.runtime.toml").write_text("[[datasets]]\n", encoding="utf-8")
    monkeypatch.setattr(config_service, "resolve_output_root", lambda: output_root.resolve())

    with pytest.raises(ValueError, match="config.runtime.toml"):
        config_service._config_file_path("output/runs/522-20260523-114514/config.original.toml")
    with pytest.raises(ValueError, match="config.runtime.toml"):
        config_service._config_file_path("output/runs/522-20260523-114514/dataset.runtime.toml")


def test_runtime_config_tree_allowed_when_output_root_changed(tmp_path: Path, monkeypatch):
    _patch_config_service_paths(monkeypatch, tmp_path)
    old_output_root = tmp_path.parent / "old-output-root"
    new_output_root = tmp_path.parent / "new-output-root"
    runtime_config = old_output_root / "522-20260523-114514" / "config.runtime.toml"
    runtime_config.parent.mkdir(parents=True)
    (runtime_config.parent / "model_cache").mkdir()
    (runtime_config.parent / "dataset_cache").mkdir()
    (runtime_config.parent / "training_output").mkdir()
    runtime_config.write_text(f'output_dir = "{(runtime_config.parent / "training_output").as_posix()}"\n', encoding="utf-8")
    monkeypatch.setattr(config_service, "resolve_output_root", lambda: new_output_root.resolve())

    path = config_service._config_file_path(str(runtime_config))

    assert path == runtime_config.resolve()


def test_external_runtime_config_requires_web_run_tree(tmp_path: Path, monkeypatch):
    _patch_config_service_paths(monkeypatch, tmp_path)
    output_root = tmp_path.parent / "output-root"
    runtime_config = tmp_path.parent / "loose" / "config.runtime.toml"
    runtime_config.parent.mkdir(parents=True)
    runtime_config.write_text("output_dir = \"somewhere\"\n", encoding="utf-8")
    monkeypatch.setattr(config_service, "resolve_output_root", lambda: output_root.resolve())

    try:
        config_service._config_file_path(str(runtime_config))
    except ValueError as exc:
        assert "全局输出文件夹" in str(exc)
    else:
        raise AssertionError("外部 runtime 配置缺少 Web 运行目录结构时必须拒绝")


def test_locked_user_group_cannot_be_deleted(tmp_path: Path, monkeypatch):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "web-file-groups.toml").write_text(
        "\n".join(
            [
                "[[groups]]",
                'id = "custom_group"',
                'label = "自定义分组"',
                "open = true",
                "locked = false",
                "trainable = true",
                "user_managed = true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (configs / "web-user-locks.toml").write_text(
        'locked_groups = ["custom_group"]\n',
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)

    group = config_service.list_config_file_groups()[0]
    assert group["user_group_locked"] is True
    assert group["renamable"] is True
    assert group["deletable"] is False

    ok, message, renamed = config_service.rename_config_file_group("custom_group", "锁定但可重命名")
    assert ok is True
    assert message == "分组已重命名"
    assert renamed["label"] == "锁定但可重命名"

    ok, message = config_service.delete_config_file_group("custom_group")
    assert ok is False
    assert "已锁定" in message


def test_unlocked_default_group_can_be_deleted_without_hiding_files(tmp_path: Path, monkeypatch):
    configs = tmp_path / "configs"
    imported = configs / "imported"
    imported.mkdir(parents=True)
    (imported / "demo.toml").write_text('output_name = "demo"\n', encoding="utf-8")
    (configs / "web-file-groups.toml").write_text(
        "\n".join(
            [
                "[[groups]]",
                'id = "imported"',
                'label = "导入配置"',
                "open = true",
                "locked = false",
                "trainable = true",
                'methods_subdir = "imported"',
                'patterns = ["configs/imported/*.toml"]',
                "",
            ]
        ),
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)

    group = config_service.list_config_file_groups()[0]
    assert group["id"] == "imported"
    assert group["deletable"] is True

    ok, message = config_service.delete_config_file_group("imported")
    assert ok is True
    assert "已保留" in message

    groups = config_service.list_config_file_groups()
    assert [group["id"] for group in groups] == ["unfiled_imported"]
    assert groups[0]["deletable"] is True
    assert [item["path"] for item in groups[0]["files"]] == ["configs/imported/demo.toml"]


def test_unlocked_default_group_can_be_renamed(tmp_path: Path, monkeypatch):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "web-file-groups.toml").write_text(
        "\n".join(
            [
                "[[groups]]",
                'id = "imported"',
                'label = "导入配置"',
                "open = true",
                "locked = false",
                "trainable = true",
                'methods_subdir = "imported"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)

    group = config_service.list_config_file_groups()[0]
    assert group["renamable"] is True

    ok, message, renamed = config_service.rename_config_file_group("imported", "常用导入配置")
    assert ok is True
    assert message == "分组已重命名"
    assert renamed["label"] == "常用导入配置"


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


def test_system_dataset_preset_is_readonly_but_can_be_saved_as(tmp_path: Path, monkeypatch):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    system_preset = tmp_path / "configs" / "datasets" / "ip_adapter.toml"
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


def test_imported_config_can_move_to_rokkotsu_group(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    train_file = configs / "imported" / "copy.toml"
    train_file.write_text('output_name = "copy"\n', encoding="utf-8")
    (configs / "web-file-groups.toml").write_text(
        "\n".join(
            [
                "[[groups]]",
                'id = "rokkotsu_goddess"',
                'label = "肋骨女神配置"',
                "open = true",
                "locked = false",
                "trainable = true",
                'methods_subdir = "imported"',
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

    ok, msg, group = config_service.move_config_file_to_group(
        "configs/imported/copy.toml",
        "rokkotsu_goddess",
    )

    assert ok is True, msg
    assert group is not None
    assert group["id"] == "rokkotsu_goddess"
    assert [item["path"] for item in group["files"]] == ["configs/imported/copy.toml"]


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
                },
            },
        ],
        {"caption_extension": ".txt", "keep_tokens": 2},
    )

    data = toml.loads(saved["content"])
    assert len(data["datasets"]) == 2
    assert data["datasets"][0]["resolution"] == 768
    assert data["datasets"][0]["max_bucket_reso"] == 768
    assert data["datasets"][0]["bucket_reso_steps"] == 32
    assert data["datasets"][0]["bucket_no_upscale"] is True
    assert data["datasets"][0]["validation_split_num"] == 4
    assert data["datasets"][1]["resolution"] == 1024
    assert data["datasets"][1]["min_bucket_reso"] == 384
    assert data["datasets"][1]["max_bucket_reso"] == 1344
    assert data["datasets"][1]["validation_split"] == 0.1

    loaded = config_service.load_dataset_preset("configs/datasets/multi_bucket.toml")
    assert loaded["datasets"][0]["settings"]["resolution"] == 768
    assert loaded["datasets"][1]["settings"]["max_bucket_reso"] == 1344


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


def test_output_runs_list_reads_direct_run_dirs_sorted(tmp_path: Path, monkeypatch):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    _patch_output_root(monkeypatch, tmp_path / "output" / "runs")
    root = tmp_path / "output" / "runs"
    older = root / "older-20260523-110000"
    newer = root / "newer-20260523-120000"
    nested = newer / "nested"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    nested.mkdir()
    (older / "config.original.toml").write_text('output_name = "older"\n', encoding="utf-8")
    (newer / "config.original.toml").write_text('output_name = "newer"\n', encoding="utf-8")
    (newer / "config.runtime.toml").write_text('output_name = "runtime"\n', encoding="utf-8")
    (nested / "config.original.toml").write_text('output_name = "nested"\n', encoding="utf-8")
    old_ts = 1_800_000_000
    new_ts = 1_800_000_100
    for path in (older, older / "config.original.toml"):
        os.utime(path, (old_ts, old_ts))
    for path in (newer, newer / "config.original.toml", newer / "config.runtime.toml"):
        os.utime(path, (new_ts, new_ts))

    result = config_service.list_output_runs()

    assert result["ok"] is True
    assert result["output_root"] == "output/runs"
    assert [item["name"] for item in result["runs"]] == [
        "newer-20260523-120000",
        "older-20260523-110000",
    ]
    assert [item["kind"] for item in result["runs"][0]["files"]] == ["original", "runtime"]


def test_output_run_read_allows_only_fixed_files_under_run(tmp_path: Path, monkeypatch):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    _patch_output_root(monkeypatch, tmp_path / "output" / "runs")
    run = tmp_path / "output" / "runs" / "522-20260523-114514"
    run.mkdir(parents=True)
    (run / "config.original.toml").write_text('output_name = "original"\n', encoding="utf-8")
    (run / "config.runtime.toml").write_text('output_name = "runtime"\n', encoding="utf-8")
    (run / "dataset.runtime.toml").write_text("[[datasets]]\n", encoding="utf-8")

    original = config_service.load_output_run_config("522-20260523-114514", "original")
    runtime = config_service.load_output_run_config("522-20260523-114514", "runtime")
    dataset = config_service.load_output_run_config("522-20260523-114514", "dataset")

    assert original["readonly"] is True
    assert original["content"] == 'output_name = "original"\n'
    assert runtime["content"] == 'output_name = "runtime"\n'
    assert dataset["content"] == "[[datasets]]\n"
    with pytest.raises(ValueError, match="直接目录名"):
        config_service.load_output_run_config("../522-20260523-114514", "original")
    with pytest.raises(ValueError, match="kind"):
        config_service.load_output_run_config("522-20260523-114514", "../config")


def test_output_run_save_as_copies_original_only_and_never_overwrites(tmp_path: Path, monkeypatch):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    _patch_output_root(monkeypatch, tmp_path / "output" / "runs")
    run = tmp_path / "output" / "runs" / "522-20260523-114514"
    run.mkdir(parents=True)
    (run / "config.original.toml").write_text('output_name = "original"\n', encoding="utf-8")
    (run / "config.runtime.toml").write_text('output_name = "runtime"\n', encoding="utf-8")

    saved = config_service.save_output_run_config_as(
        "522-20260523-114514",
        "copied_from_run",
        "imported",
    )

    assert saved["ok"] is True
    assert saved["file"] == "configs/imported/copied_from_run.toml"
    copied_path = tmp_path / "configs" / "imported" / "copied_from_run.toml"
    assert copied_path.read_text(encoding="utf-8") == 'output_name = "original"\n'
    with pytest.raises(ValueError, match="已存在"):
        config_service.save_output_run_config_as("522-20260523-114514", "copied_from_run", "imported")


def test_output_run_save_as_rejects_missing_or_invalid_original(tmp_path: Path, monkeypatch):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    _patch_output_root(monkeypatch, tmp_path / "output" / "runs")
    missing = tmp_path / "output" / "runs" / "legacy-20260523-114514"
    invalid = tmp_path / "output" / "runs" / "bad-20260523-114514"
    missing.mkdir(parents=True)
    invalid.mkdir(parents=True)
    (invalid / "config.original.toml").write_text("invalid = [\n", encoding="utf-8")

    with pytest.raises(ValueError, match="没有 config.original.toml"):
        config_service.save_output_run_config_as("legacy-20260523-114514", "legacy_copy", "imported")
    with pytest.raises(ValueError, match="TOML 语法错误"):
        config_service.save_output_run_config_as("bad-20260523-114514", "bad_copy", "imported")


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
    configs = root / "configs"
    monkeypatch.setattr(config_service, "ROOT", root)
    monkeypatch.setattr(config_service, "CONFIGS_DIR", configs)
    monkeypatch.setattr(config_service, "DATASET_PRESETS_DIR", configs / "datasets")
    monkeypatch.setattr(config_service, "GUI_METHODS_DIR", configs / "gui-methods")
    monkeypatch.setattr(config_service, "IMPORTED_CONFIGS_DIR", configs / "imported")
    monkeypatch.setattr(config_service, "PRESETS_FILE", configs / "presets.toml")
    monkeypatch.setattr(config_service, "WEB_FILE_GROUPS_FILE", configs / "web-file-groups.toml")
    monkeypatch.setattr(config_service, "WEB_USER_LOCKS_FILE", configs / "web-user-locks.toml")


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
