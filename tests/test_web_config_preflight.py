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



# Split: preflight domain

from tests import web_config_test_support as _web_config_support

globals().update(
    {
        name: value
        for name, value in vars(_web_config_support).items()
        if name == "Path" or not name.startswith("__")
    }
)

def test_preflight_module_imports_without_facade_cycle():
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    script = (
        "import sys; "
        "import web.services.config.preflight as preflight; "
        "assert callable(preflight.preflight_training_config); "
        "assert callable(preflight.training_sample_sampler_status); "
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


def test_preflight_resolves_display_config_path_under_external_configs_root(tmp_path: Path, monkeypatch):
    root = tmp_path / "repo"
    configs = tmp_path / "external-configs"
    root.mkdir()
    _write_minimal_config_tree(root)
    _patch_external_config_service_paths(monkeypatch, root, configs)
    (configs / "imported").mkdir(parents=True, exist_ok=True)
    external_config = configs / "imported" / "rokkotsu_goddess_528_tag.toml"
    external_config.write_text('output_name = "rokkotsu_goddess_528_tag"\n', encoding="utf-8")

    path = config_service._config_file_path("configs/imported/rokkotsu_goddess_528_tag.toml")

    assert path == external_config.resolve()


def test_preflight_fills_blank_model_paths_from_global_settings(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    (configs / "web-ui-settings.toml").write_text(
        "\n".join(
            [
                "[global]",
                'pretrained_model_name_or_path = "models/global-anima.safetensors"',
                'qwen3 = "models/global-qwen.safetensors"',
                'vae = "models/global-vae.safetensors"',
            ]
        ),
        encoding="utf-8",
    )
    source_dir = tmp_path / "image_dataset" / "selected"
    source_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(source_dir / "sample.png")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "global-anima.safetensors").write_bytes(b"model")
    (tmp_path / "models" / "global-qwen.safetensors").write_bytes(b"qwen")
    (tmp_path / "models" / "global-vae.safetensors").write_bytes(b"vae")
    selected_config = configs / "imported" / "selected.toml"
    selected_config.write_text(
        "\n".join(
            [
                'source_image_dir = "image_dataset/selected"',
                'pretrained_model_name_or_path = ""',
                'qwen3 = ""',
                'vae = ""',
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

    checks = {item["key"]: item for item in result["checks"]}
    assert checks["pretrained_model_name_or_path"]["level"] == "ok"
    assert checks["pretrained_model_name_or_path"]["path"] == "models/global-anima.safetensors"
    assert checks["qwen3"]["level"] == "ok"
    assert checks["qwen3"]["path"] == "models/global-qwen.safetensors"
    assert checks["vae"]["level"] == "ok"
    assert checks["vae"]["path"] == "models/global-vae.safetensors"
    assert "pretrained_model_name_or_path" not in {item["key"] for item in result["errors"]}


def test_preflight_rejects_blank_output_name(tmp_path: Path, monkeypatch):
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
                'output_name = "   "',
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

    output_checks = [item for item in result["errors"] if item["key"] == "output_name"]
    assert result["ok"] is False
    assert output_checks[-1]["message"] == "输出名称未填写"


def test_preflight_rejects_selective_checkpoint_with_full_checkpointing(
    tmp_path: Path, monkeypatch
):
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
                "blocks_to_swap = 24",
                'selective_checkpoint = "mlp_only"',
                "gradient_checkpointing = true",
                "unsloth_offload_checkpointing = false",
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

    errors = [item for item in result["errors"] if item["key"] == "gradient_checkpointing"]
    assert result["ok"] is False
    assert "selective_checkpoint" in errors[-1]["message"]


def test_preflight_remains_available_from_legacy_module(tmp_path: Path, monkeypatch):
    _write_selected_checkpoint_preflight_config(
        tmp_path,
        monkeypatch,
        [
            "gradient_checkpointing = true",
            "cpu_offload_checkpointing = false",
            "unsloth_offload_checkpointing = false",
        ],
    )
    configs = tmp_path / "configs"
    monkeypatch.setattr(legacy_config, "ROOT", tmp_path)
    monkeypatch.setattr(legacy_config, "CONFIGS_DIR", configs)
    monkeypatch.setattr(legacy_config, "DATASET_PRESETS_DIR", configs / "datasets")
    monkeypatch.setattr(legacy_config, "GUI_METHODS_DIR", configs / "gui-methods")
    monkeypatch.setattr(legacy_config, "IMPORTED_CONFIGS_DIR", configs / "imported")
    monkeypatch.setattr(legacy_config, "PRESETS_FILE", configs / "presets.toml")
    monkeypatch.setattr(legacy_config, "WEB_FILE_GROUPS_FILE", configs / "web-file-groups.toml")
    monkeypatch.setattr(legacy_config, "WEB_USER_LOCKS_FILE", configs / "web-user-locks.toml")

    result = legacy_config.preflight_training_config(
        "lora",
        "default",
        "imported",
        config_file="configs/imported/selected.toml",
    )

    assert result["ok"] is True
    env_checks = [item for item in result["checks"] if item["key"] == "preprocess_environment"]
    assert env_checks[-1]["level"] == "ok"


def test_legacy_preflight_exports_forward_to_split_preflight_module():
    from web.services.config import preflight as preflight_impl

    missing = []
    not_forwarded = []
    for name in preflight_impl.__all__:
        exported = getattr(legacy_config, name, None)
        if exported is None:
            missing.append(name)
            continue
        doc = str(getattr(exported, "__doc__", "") or "")
        if "web.services.config.preflight" not in doc:
            not_forwarded.append(name)

    assert missing == []
    assert not_forwarded == []
    for name in (
        "set_user_file_lock",
        "set_user_group_lock",
        "create_config_file_group",
        "rename_config_file_group",
        "delete_config_file_group",
        "reorder_config_file_group",
        "move_config_file_to_group",
        "place_config_file_in_group",
        "place_config_file_group",
        "reorder_config_file_in_group",
        "restore_system_presets",
        "list_config_files",
        "list_config_file_groups",
        "export_config_file_group_archive",
        "get_config_file_meta",
    ):
        assert getattr(legacy_config, name) is legacy_config._FILE_GROUPS_SHIMS[name]


def test_preflight_helpers_remain_available_from_legacy_module():
    expected_shims = (
        "preflight_training_config",
        "_load_training_config_for_web_run",
        "_config_file_path",
        "is_web_runtime_config",
        "training_sample_sampler_status",
        "apply_global_model_path_defaults",
        "_check_training_images",
        "_check_dataset_source_paths",
        "_check_dataset_bucket_settings",
        "_check_dataset_paths",
        "_check_cache_sidecars",
    )
    assert tuple(legacy_config._PREFLIGHT_SHIM_NAMES) == expected_shims
    assert tuple(legacy_config._PREFLIGHT_SHIMS) == expected_shims
    for name in expected_shims:
        assert getattr(legacy_config, name) is legacy_config._PREFLIGHT_SHIMS[name]
        assert (
            getattr(legacy_config, name).__doc__
            == f"Compatibility shim forwarding to web.services.config.preflight.{name}."
        )

    assert legacy_config.training_sample_sampler_status("ddim") == ("euler", "legacy")
    for name in expected_shims:
        assert getattr(legacy_config, name) is legacy_config._PREFLIGHT_SHIMS[name]


def test_legacy_preflight_private_helpers_forward_to_split_module(monkeypatch):
    from web.services.config import preflight as preflight_impl

    calls: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}

    def sentinel(name: str):
        def impl(*args, **kwargs):
            calls[name] = (args, kwargs)
            return {"name": name, "args": args, "kwargs": kwargs}

        return impl

    def add(*args):
        return args

    cfg = {"sample_prompts": "configs/sample_prompts.txt"}
    config_path = Path("configs/imported/config.runtime.toml")
    cache_dirs = [(1, Path("post_image_dataset/lora"), False)]
    helper_args = {
        "_inspect_network_weight": ("weights/demo.safetensors",),
        "_check_network_weights": (cfg, add, "lora", "default", "gui-methods", None),
        "_check_training_sample_config": (cfg, add),
        "_config_path_from_display_path": ("configs/imported/lora.toml",),
        "_is_allowed_training_config_path": (config_path,),
        "_is_web_runtime_config_tree": (config_path,),
        "_is_output_run_snapshot_config": (config_path,),
        "_has_web_runtime_dirs": (config_path.parent,),
        "_looks_like_web_runtime_config": (cfg,),
        "_check_web_preprocess_environment": (add,),
        "_web_python_executable": (),
        "_check_cache_sidecar_pattern": (
            add,
            cache_dirs,
            "*.npz",
            "latent_cache",
            "VAE latent 缓存",
            "未找到 .npz latent 缓存",
        ),
    }
    for name in helper_args:
        impl_name = "_inspect_network_weight_impl" if name == "_inspect_network_weight" else name
        monkeypatch.setattr(preflight_impl, impl_name, sentinel(name))

    for name, args in helper_args.items():
        kwargs = {}
        if name == "_inspect_network_weight":
            kwargs = {
                "variant": "lora",
                "preset": "default",
                "methods_subdir": "gui-methods",
                "config_file": None,
                "cfg": cfg,
            }
        result = getattr(legacy_config, name)(*args, **kwargs)
        assert result["name"] == name
        assert calls[name] == (args, kwargs)


def test_preflight_allows_block_swap_with_standard_gradient_checkpointing(
    tmp_path: Path, monkeypatch
):
    _write_selected_checkpoint_preflight_config(
        tmp_path,
        monkeypatch,
        [
            "blocks_to_swap = 8",
            "gradient_checkpointing = true",
            "cpu_offload_checkpointing = false",
            "unsloth_offload_checkpointing = false",
            'selective_checkpoint = "off"',
        ],
    )

    result = config_service.preflight_training_config(
        "lora",
        "default",
        "imported",
        config_file="configs/imported/selected.toml",
    )

    checkpoint_errors = [
        item
        for item in result["errors"]
        if item["key"]
        in {
            "blocks_to_swap",
            "gradient_checkpointing",
            "cpu_offload_checkpointing",
            "unsloth_offload_checkpointing",
        }
    ]
    assert result["ok"] is True
    assert checkpoint_errors == []


def test_preflight_warns_lokr_full_checkpoint_pins_torch_compile_budget(
    tmp_path: Path, monkeypatch
):
    _write_selected_checkpoint_preflight_config(
        tmp_path,
        monkeypatch,
        [
            "use_lokr = true",
            "blocks_to_swap = 26",
            "gradient_checkpointing = true",
            "torch_compile = true",
            "cpu_offload_checkpointing = false",
            "unsloth_offload_checkpointing = false",
            'selective_checkpoint = "off"',
        ],
    )

    result = config_service.preflight_training_config(
        "lora",
        "default",
        "imported",
        config_file="configs/imported/selected.toml",
    )

    checkpoint_errors = [
        item
        for item in result["errors"]
        if item["key"]
        in {
            "blocks_to_swap",
            "gradient_checkpointing",
            "torch_compile",
            "cpu_offload_checkpointing",
            "unsloth_offload_checkpointing",
        }
    ]
    warnings = {item["key"]: item["message"] for item in result["warnings"]}
    assert result["ok"] is True
    assert checkpoint_errors == []
    assert "LoKr" in warnings["torch_compile"]
    assert "Dynamo graph/accumulated 预算" in warnings["torch_compile"]
    assert "稳定 graph 查找顺序" in warnings["torch_compile"]


def test_preflight_rejects_block_swap_with_unsloth_offload_checkpointing(
    tmp_path: Path, monkeypatch
):
    _write_selected_checkpoint_preflight_config(
        tmp_path,
        monkeypatch,
        [
            "blocks_to_swap = 8",
            "gradient_checkpointing = true",
            "unsloth_offload_checkpointing = true",
            'selective_checkpoint = "off"',
        ],
    )

    result = config_service.preflight_training_config(
        "lora",
        "default",
        "imported",
        config_file="configs/imported/selected.toml",
    )

    errors = [
        item
        for item in result["errors"]
        if item["key"] == "unsloth_offload_checkpointing"
    ]
    assert result["ok"] is False
    assert "普通 gradient_checkpointing" in errors[-1]["message"]


def test_preflight_rejects_dop_without_class_prompt(tmp_path: Path, monkeypatch):
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
                "prior_preservation_weight = 0.1",
                'diff_output_preservation_trigger = "sks"',
                'diff_output_preservation_class = ""',
                "blank_prompt_preservation = false",
                "use_text_cache = true",
                "cache_llm_adapter_outputs = true",
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

    errors = [item for item in result["errors"] if item["key"] == "diff_output_preservation_class"]
    assert result["ok"] is False
    assert "DOP" in errors[-1]["message"]


def test_preflight_blocks_runtime_config_reusing_history_training_output_dir(
    tmp_path: Path, monkeypatch
):
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    models = tmp_path / "models"
    models.mkdir()
    (models / "anima.safetensors").write_bytes(b"model")
    (models / "qwen.safetensors").write_bytes(b"qwen")
    (models / "vae.safetensors").write_bytes(b"vae")
    source_dir = tmp_path / "image_dataset" / "selected"
    resized_dir = tmp_path / "output" / "web-runs" / "old-run" / "dataset_cache" / "dataset-01" / "resized"
    cache_dir = tmp_path / "output" / "web-runs" / "old-run" / "dataset_cache" / "dataset-01" / "lora"
    source_dir.mkdir(parents=True)
    resized_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(source_dir / "sample.png")
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(resized_dir / "sample.png")
    (source_dir / "sample.txt").write_text("1girl, solo\n", encoding="utf-8")
    dataset_path.write_text(
        "\n".join(
            [
                "[[datasets]]",
                "",
                "[[datasets.subsets]]",
                'source_dir = "image_dataset/selected"',
                'image_dir = "output/web-runs/old-run/dataset_cache/dataset-01/resized"',
                'cache_dir = "output/web-runs/old-run/dataset_cache/dataset-01/lora"',
            ]
        ),
        encoding="utf-8",
    )

    run_dir = tmp_path / "output" / "web-runs" / "old-run"
    (run_dir / "model_cache" / "logs").mkdir(parents=True)
    (run_dir / "training_output").mkdir(parents=True)
    runtime_config = run_dir / "config.runtime.toml"
    runtime_config.write_text(
        "\n".join(
            [
                'output_name = "demo"',
                'output_dir = "output/web-runs/old-run/training_output"',
                'logging_dir = "output/web-runs/old-run/model_cache/logs"',
                'dataset_config = "configs/datasets/lora.toml"',
                'pretrained_model_name_or_path = "models/anima.safetensors"',
                'qwen3 = "models/qwen.safetensors"',
                'vae = "models/vae.safetensors"',
            ]
        ),
        encoding="utf-8",
    )
    history_task = configs / "web-training-history" / "20260615-120000-training-demo"
    history_task.mkdir(parents=True)
    (history_task / "meta.json").write_text(
        json.dumps(
            {
                "id": history_task.name,
                "name": "旧训练",
                "job": "training",
                "state": "done",
                "run_dir": "output/web-runs/old-run",
                "output_dir": "output/web-runs/old-run/training_output",
                "training_output_dir": "output/web-runs/old-run/training_output",
            }
        ),
        encoding="utf-8",
    )

    result = config_service.preflight_training_config(
        "lora",
        "default",
        "imported",
        config_file="output/web-runs/old-run/config.runtime.toml",
    )

    output_checks = [item for item in result["errors"] if item["key"] == "output_dir"]
    assert result["ok"] is False
    assert output_checks
    assert output_checks[-1]["path"] == "output/web-runs/old-run/training_output"
    assert "已有历史训练输出目录" in output_checks[-1]["message"]
    assert "完整续训" in output_checks[-1]["message"]


def test_preflight_warns_when_sample_prompts_have_no_schedule(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    selected_config = configs / "imported" / "selected.toml"
    selected_config.write_text(
        "\n".join([
            'sample_prompts = "configs/sample_prompts.txt"',
            'source_image_dir = "image_dataset/selected"',
        ]),
        encoding="utf-8",
    )

    result = config_service.preflight_training_config(
        "lora",
        "default",
        "imported",
        config_file="configs/imported/selected.toml",
    )

    messages = [item["message"] for item in result["warnings"]]
    assert any("未启用训练前、按轮或按步采样" in message for message in messages)


def test_preflight_warns_for_dual_sample_schedule_and_legacy_sampler(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    selected_config = configs / "imported" / "selected.toml"
    selected_config.write_text(
        "\n".join([
            'sample_prompts = "configs/sample_prompts.txt"',
            "sample_every_n_epochs = 1",
            "sample_every_n_steps = 500",
            'sample_sampler = "ddim"',
            'source_image_dir = "image_dataset/selected"',
        ]),
        encoding="utf-8",
    )

    result = config_service.preflight_training_config(
        "lora",
        "default",
        "imported",
        config_file="configs/imported/selected.toml",
    )

    warnings = {item["key"]: item["message"] for item in result["warnings"]}
    assert "同时启用按轮和按步采样" in warnings["sample_schedule"]
    assert "会按 euler 兼容处理" in warnings["sample_sampler"]


def test_preflight_checks_manual_network_weights_before_launch(tmp_path: Path, monkeypatch):
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
                'network_weights = "weights/demo_lokr.safetensors"',
                "dim_from_weights = true",
            ]
        ),
        encoding="utf-8",
    )
    called: dict[str, str] = {}

    def fake_inspect(path: str, **kwargs) -> dict:
        called["path"] = path
        called["variant"] = kwargs["variant"]
        called["config_file"] = kwargs["config_file"]
        return {
            "compatible": False,
            "message": "LoKr 权重需要当前变体为 lokr",
            "kind": "LoKr",
            "abs_path": str(tmp_path / "weights" / "demo_lokr.safetensors"),
        }

    monkeypatch.setattr(config_service, "_inspect_network_weight", fake_inspect)

    result = config_service.preflight_training_config(
        "lora",
        "default",
        "imported",
        config_file="configs/imported/selected.toml",
    )

    weight_checks = [item for item in result["checks"] if item["key"] == "network_weights"]
    assert called == {
        "path": "weights/demo_lokr.safetensors",
        "variant": "lora",
        "config_file": "configs/imported/selected.toml",
    }
    assert result["ok"] is False
    assert weight_checks[-1]["level"] == "error"
    assert "LoKr 权重需要当前变体为 lokr" in weight_checks[-1]["message"]




def test_preflight_rejects_max_bucket_below_resolution(tmp_path: Path, monkeypatch):
    """resolution > max_bucket_reso 应在启动前报错，避免预处理中途 assert。"""
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    source_dir = tmp_path / "image_dataset" / "hires"
    source_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64), color=(10, 20, 30)).save(source_dir / "sample.png")
    dataset_path.write_text(
        "\n".join(
            [
                "[[datasets]]",
                "resolution = 1536",
                "min_bucket_reso = 256",
                "max_bucket_reso = 1024",
                "enable_bucket = true",
                "",
                "[[datasets.subsets]]",
                'image_dir = "post_image_dataset/hires_resized"',
                'cache_dir = "post_image_dataset/hires_lora"',
                'custom_attributes = {source_dir = "image_dataset/hires"}',
            ]
        ),
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)

    result = config_service.preflight_training_config(
        "lora",
        "default",
        "imported",
        config_file="configs/imported/lora.toml",
    )

    bucket_checks = [item for item in result["checks"] if item["key"] in {"dataset_bucket", "dataset_1_bucket"} or item["key"].endswith("_bucket")]
    assert result["ok"] is False
    assert any("max_bucket_reso" in item.get("message", "") for item in result.get("errors") or [])
    assert bucket_checks
    assert bucket_checks[-1]["level"] == "error"


def test_preflight_rejects_source_dir_with_only_cache_sidecars(tmp_path: Path, monkeypatch):
    """源目录只有 latent/text 缓存、没有图片时，应在预处理前直接报错。"""
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    source_dir = tmp_path / "image_dataset" / "cache_only"
    source_dir.mkdir(parents=True)
    (source_dir / "sample_0896x1152_anima.npz").write_bytes(b"npz")
    (source_dir / "sample_anima_te.safetensors").write_bytes(b"te")
    dataset_path.write_text(
        "\n".join(
            [
                "[[datasets]]",
                "",
                "[[datasets.subsets]]",
                'image_dir = "post_image_dataset/cache_only_resized"',
                'cache_dir = "post_image_dataset/cache_only_lora"',
                'custom_attributes = {source_dir = "image_dataset/cache_only"}',
            ]
        ),
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)

    result = config_service.preflight_training_config(
        "lora",
        "default",
        "imported",
        config_file="configs/imported/lora.toml",
    )

    image_checks = [item for item in result["checks"] if item["key"] == "source_image_dir_images"]
    assert result["ok"] is False
    assert image_checks
    assert image_checks[-1]["level"] == "error"
    assert "没有可训练图片" in image_checks[-1]["message"]


def test_preflight_nl_tag_mix_accepts_single_captioned_directory(tmp_path: Path, monkeypatch):
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    source_dir = tmp_path / "image_dataset" / "mixed"
    source_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(source_dir / "sample.png")
    (source_dir / "sample.txt").write_text(
        "1girl, solo, silver hair, purple eyes, white dress, standing\n",
        encoding="utf-8",
    )
    dataset_path.write_text(
        "\n".join(
            [
                "[[datasets]]",
                "",
                "[[datasets.subsets]]",
                'image_dir = "post_image_dataset/mixed_resized"',
                'cache_dir = "post_image_dataset/mixed_lora"',
                'custom_attributes = {source_dir = "image_dataset/mixed", nl_tag_mix = {enabled = true, tag_ratio = 0.7}}',
            ]
        ),
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)

    result = config_service.preflight_training_config(
        "lora",
        "default",
        "imported",
        config_file="configs/imported/lora.toml",
    )

    mix_checks = [item for item in result["checks"] if item["key"] == "source_image_dir_nl_tag_mix"]
    assert mix_checks == []
    source_checks = [item for item in result["checks"] if item["key"] == "source_image_dir"]
    assert source_checks[-1]["level"] == "ok"


@pytest.mark.parametrize(
    ("caption_source_mode", "sidecar_name", "sidecar_content"),
    [
        (
            "json",
            "sample.json",
            json.dumps({"quality": "newest", "tags": ["1girl", "solo", "blue eyes"]}),
        ),
        (
            "captions_json",
            "captions.json",
            json.dumps({"sample.png": ["1girl, solo, blue eyes, white dress"]}),
        ),
    ],
)
def test_preflight_nl_tag_mix_accepts_structured_caption_sources(
    tmp_path: Path,
    monkeypatch,
    caption_source_mode: str,
    sidecar_name: str,
    sidecar_content: str,
):
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    source_dir = tmp_path / "image_dataset" / "mixed"
    source_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(source_dir / "sample.png")
    (source_dir / sidecar_name).write_text(sidecar_content, encoding="utf-8")
    dataset_path.write_text(
        "\n".join(
            [
                "[[datasets]]",
                f'caption_source_mode = "{caption_source_mode}"',
                "",
                "[[datasets.subsets]]",
                'image_dir = "post_image_dataset/mixed_resized"',
                'cache_dir = "post_image_dataset/mixed_lora"',
                'custom_attributes = {source_dir = "image_dataset/mixed", nl_tag_mix = {enabled = true, tag_ratio = 0.7}}',
            ]
        ),
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)

    result = config_service.preflight_training_config(
        "lora",
        "default",
        "imported",
        config_file="configs/imported/lora.toml",
    )

    caption_checks = [item for item in result["checks"] if item["key"] == "source_image_dir_nl_tag_mix_captions"]
    assert caption_checks == []
    source_checks = [item for item in result["checks"] if item["key"] == "source_image_dir"]
    assert source_checks[-1]["level"] == "ok"


def test_preflight_nl_tag_mix_accepts_recursive_captions_json(
    tmp_path: Path,
    monkeypatch,
):
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    source_dir = tmp_path / "image_dataset" / "mixed"
    nested_dir = source_dir / "nested"
    nested_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(nested_dir / "sample.png")
    (source_dir / "captions.json").write_text(
        json.dumps({"nested/sample.png": ["1girl, solo, blue eyes, white dress"]}),
        encoding="utf-8",
    )
    dataset_path.write_text(
        "\n".join(
            [
                "[[datasets]]",
                'caption_source_mode = "captions_json"',
                "",
                "[[datasets.subsets]]",
                'image_dir = "post_image_dataset/mixed_resized"',
                'cache_dir = "post_image_dataset/mixed_lora"',
                'recursive = true',
                'custom_attributes = {source_dir = "image_dataset/mixed", nl_tag_mix = {enabled = true, tag_ratio = 0.7}}',
            ]
        ),
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)

    result = config_service.preflight_training_config(
        "lora",
        "default",
        "imported",
        config_file="configs/imported/lora.toml",
    )

    caption_checks = [item for item in result["checks"] if item["key"] == "source_image_dir_nl_tag_mix_captions"]
    assert caption_checks == []
    source_checks = [item for item in result["checks"] if item["key"] == "source_image_dir"]
    assert source_checks[-1]["level"] == "ok"


def test_preflight_nl_tag_mix_warns_when_no_readable_captions(tmp_path: Path, monkeypatch):
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    source_dir = tmp_path / "image_dataset" / "mixed"
    source_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(source_dir / "sample.png")
    dataset_path.write_text(
        "\n".join(
            [
                "[[datasets]]",
                "",
                "[[datasets.subsets]]",
                'image_dir = "post_image_dataset/mixed_resized"',
                'cache_dir = "post_image_dataset/mixed_lora"',
                'custom_attributes = {source_dir = "image_dataset/mixed", nl_tag_mix = {enabled = true, tag_ratio = 0.7}}',
            ]
        ),
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)

    result = config_service.preflight_training_config(
        "lora",
        "default",
        "imported",
        config_file="configs/imported/lora.toml",
    )

    caption_checks = [item for item in result["checks"] if item["key"] == "source_image_dir_nl_tag_mix_captions"]
    assert caption_checks[-1]["level"] == "warning"
    assert "全部按 tag 处理" in caption_checks[-1]["message"]


def test_preflight_trigger_clone_requires_prompt(tmp_path: Path, monkeypatch):
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    source_dir = tmp_path / "image_dataset" / "character"
    source_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(source_dir / "sample.png")
    dataset_path.write_text(
        "\n".join(
            [
                "[[datasets]]",
                "",
                "[[datasets.subsets]]",
                'image_dir = "post_image_dataset/character_resized"',
                'cache_dir = "post_image_dataset/character_lora"',
                'custom_attributes = {source_dir = "image_dataset/character", trigger_clone = {enabled = true, prompt = "", num_repeats = 2}}',
            ]
        ),
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)

    result = config_service.preflight_training_config(
        "lora",
        "default",
        "imported",
        config_file="configs/imported/lora.toml",
    )

    prompt_checks = [item for item in result["checks"] if item["key"] == "source_image_dir_trigger_clone_prompt"]
    assert prompt_checks[-1]["level"] == "error"
    assert "触发提示词为空" in prompt_checks[-1]["message"]


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


def test_preflight_runtime_config_reports_caption_source_detection(tmp_path: Path, monkeypatch):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    run_dir = tmp_path / "output" / "runs" / "522-20260523-114514"
    source_dir = tmp_path / "image_dataset" / "a"
    resized_dir = run_dir / "dataset_cache" / "dataset-01" / "resized"
    cache_dir = run_dir / "dataset_cache" / "dataset-01" / "lora"
    for path in (source_dir, resized_dir, cache_dir, run_dir / "model_cache", run_dir / "training_output"):
        path.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(resized_dir / "hero.png")
    (source_dir / "captions.json").write_text(
        json.dumps({"hero.png": ["caption one", "caption two"]}),
        encoding="utf-8",
    )
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
                'caption_source_mode = "auto"',
                "[[datasets.subsets]]",
                f'image_dir = "{resized_dir.relative_to(tmp_path).as_posix()}"',
                f'cache_dir = "{cache_dir.relative_to(tmp_path).as_posix()}"',
                f'custom_attributes = {{ source_dir = "{source_dir.relative_to(tmp_path).as_posix()}" }}',
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

    caption_check = [item for item in result["checks"] if item["key"] == "captions"][-1]
    assert caption_check["level"] == "ok"
    assert "DiffPipeForge captions.json 1 张" in caption_check["message"]
    assert "共 2 条标注" in caption_check["message"]


def test_runtime_preflight_checks_nested_training_images_and_cache_sidecars(
    tmp_path: Path,
    monkeypatch,
):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    source_dir = tmp_path / "image_dataset" / "nested"
    resized_dir = tmp_path / "post_image_dataset" / "nested_resized"
    cache_dir = tmp_path / "post_image_dataset" / "nested_cache"
    for path in (source_dir / "char_a", resized_dir / "char_a", cache_dir / "char_a"):
        path.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(resized_dir / "char_a" / "hero.png")
    (cache_dir / "char_a" / "hero_0008x0008_anima.npz").write_bytes(b"latent")
    (cache_dir / "char_a" / "hero_anima_te.safetensors").write_bytes(b"te")
    dataset_path = configs / "datasets" / "nested.toml"
    dataset_path.write_text(
        "\n".join(
            [
                "[[datasets]]",
                "[[datasets.subsets]]",
                'image_dir = "post_image_dataset/nested_resized"',
                'cache_dir = "post_image_dataset/nested_cache"',
                "recursive = true",
                'path_pattern = "char_a/*"',
                'custom_attributes = { source_dir = "image_dataset/nested" }',
            ]
        ),
        encoding="utf-8",
    )
    checks: list[dict[str, str]] = []

    def add(level, key, message, path=None):
        checks.append({"level": level, "key": key, "message": message})

    cfg = {
        "dataset_config": "configs/datasets/nested.toml",
        "cache_latents_to_disk": True,
        "cache_text_encoder_outputs_to_disk": True,
    }
    config_service._check_training_images(cfg, add)
    config_service._check_cache_sidecars(cfg, add)

    by_key = {item["key"]: item for item in checks}
    assert "training_images" not in by_key
    assert by_key["latent_cache"]["level"] == "ok"
    assert by_key["text_cache"]["level"] == "ok"



def _write_stage_schedule_preflight_config(
    tmp_path: Path,
    monkeypatch,
    *,
    stage_schedule_enabled: bool,
    stage_schedule: list[dict],
    subset_count: int = 1,
) -> str:
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    for name in (
        "diffusion_models/anima-base-v1.0.safetensors",
        "text_encoders/qwen_3_06b_base.safetensors",
        "vae/qwen_image_vae.safetensors",
    ):
        path = models / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model")

    source_dirs = []
    for idx in range(subset_count):
        source = tmp_path / "image_dataset" / f"set{idx}"
        source.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color=(10 + idx, 20, 30)).save(source / "a.png")
        source_dirs.append(source)

    dataset_lines = ["[[datasets]]", ""]
    for source in source_dirs:
        dataset_lines.extend(
            [
                "[[datasets.subsets]]",
                f'image_dir = "{source.as_posix()}"',
                "num_repeats = 1",
                "",
            ]
        )
    dataset_path.write_text("\n".join(dataset_lines), encoding="utf-8")

    selected = configs / "imported" / "stage.toml"
    lines = [
        'output_name = "stage-demo"',
        'dataset_config = "configs/datasets/lora.toml"',
        f"stage_schedule_enabled = {'true' if stage_schedule_enabled else 'false'}",
        "stage_schedule = [",
    ]
    for stage in stage_schedule:
        lines.append(
            "  { "
            f'name = "{stage["name"]}", '
            f'subset_index = {int(stage["subset_index"])}, '
            f'start_pct = {float(stage["start_pct"])}, '
            f'end_pct = {float(stage["end_pct"])}'
            " },"
        )
    lines.append("]")
    selected.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "configs/imported/stage.toml"


def test_preflight_rejects_invalid_stage_schedule_gap(tmp_path: Path, monkeypatch):
    config_file = _write_stage_schedule_preflight_config(
        tmp_path,
        monkeypatch,
        stage_schedule_enabled=True,
        stage_schedule=[
            {"name": "a", "subset_index": 0, "start_pct": 0.0, "end_pct": 0.4},
            {"name": "b", "subset_index": 0, "start_pct": 0.6, "end_pct": 1.0},
        ],
        subset_count=1,
    )
    result = config_service.preflight_training_config(
        "lora",
        "default",
        "imported",
        config_file=config_file,
    )
    assert result["ok"] is False
    stage_checks = [c for c in result["checks"] if c.get("key") == "stage_schedule"]
    assert stage_checks
    assert any("贴齐" in c.get("message", "") or "stage" in c.get("message", "").lower() for c in stage_checks)


def test_preflight_rejects_stage_subset_out_of_range(tmp_path: Path, monkeypatch):
    config_file = _write_stage_schedule_preflight_config(
        tmp_path,
        monkeypatch,
        stage_schedule_enabled=True,
        stage_schedule=[
            {"name": "a", "subset_index": 3, "start_pct": 0.0, "end_pct": 1.0},
        ],
        subset_count=1,
    )
    result = config_service.preflight_training_config(
        "lora",
        "default",
        "imported",
        config_file=config_file,
    )
    assert result["ok"] is False
    stage_checks = [c for c in result["checks"] if c.get("key") == "stage_schedule"]
    assert any("subset_index" in c.get("message", "") for c in stage_checks)


def test_prepare_web_runtime_config_rejects_invalid_stage_schedule(tmp_path: Path, monkeypatch):
    from web.services import training_service

    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    output_root = tmp_path / "output" / "runs"
    output_root.mkdir(parents=True)
    monkeypatch.setattr(training_service, "resolve_output_root", lambda: output_root)
    monkeypatch.setattr(
        "web.services.training.runtime_prepare.resolve_output_root",
        lambda: output_root,
        raising=False,
    )

    source = tmp_path / "image_dataset" / "a"
    source.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(source / "a.png")
    dataset_path.write_text(
        "\n".join(
            [
                "[[datasets]]",
                "",
                "[[datasets.subsets]]",
                f'image_dir = "{source.as_posix()}"',
                "num_repeats = 1",
            ]
        ),
        encoding="utf-8",
    )
    selected = configs / "imported" / "bad-stage.toml"
    selected.write_text(
        "\n".join(
            [
                'output_name = "bad-stage"',
                'dataset_config = "configs/datasets/lora.toml"',
                "stage_schedule_enabled = true",
                "stage_schedule = [",
                '  { name = "a", subset_index = 0, start_pct = 0.0, end_pct = 0.3 },',
                '  { name = "b", subset_index = 0, start_pct = 0.5, end_pct = 1.0 },',
                "]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="stage_schedule"):
        training_service._prepare_web_runtime_config(
            "lora",
            "default",
            "imported",
            source_config_file="configs/imported/bad-stage.toml",
        )


def test_preflight_warns_unknown_config_key(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    for name in (
        "diffusion_models/anima-base-v1.0.safetensors",
        "text_encoders/qwen_3_06b_base.safetensors",
        "vae/qwen_image_vae.safetensors",
    ):
        path = models / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model")
    source = tmp_path / "image_dataset" / "a"
    source.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(source / "a.png")
    selected = configs / "imported" / "unknown.toml"
    selected.write_text(
        "\n".join(
            [
                'output_name = "unknown-demo"',
                'source_image_dir = "image_dataset/a"',
                "custom_unknown_key = true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result = config_service.preflight_training_config(
        "lora",
        "default",
        "imported",
        config_file="configs/imported/unknown.toml",
    )
    warnings = [c for c in result["checks"] if c.get("level") == "warning" and c.get("key") == "schema"]
    assert warnings
    assert any("custom_unknown_key" in c.get("message", "") for c in warnings)
