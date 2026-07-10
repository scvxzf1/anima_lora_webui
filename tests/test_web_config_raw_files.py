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



# Split: raw_files / merge / legacy shims

from tests import web_config_test_support as _web_config_support

globals().update(
    {
        name: value
        for name, value in vars(_web_config_support).items()
        if name == "Path" or not name.startswith("__")
    }
)

def test_raw_files_module_imports_without_facade_cycle():
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    script = (
        "import sys; "
        "import web.services.config.raw_files as raw; "
        "assert 'web.services.config_service' not in sys.modules; "
        "assert 'web.services.config._legacy' not in sys.modules; "
        "assert raw._is_spd_patch_target('configs/methods/spd.toml', {}) is True; "
        "assert raw._is_spd_patch_target('', {'dit_path': 'm', 'data_dir': 'd', 'iterations': 1, 'schedule': {}}) is True; "
        "assert raw._is_spd_patch_target('configs/gui-methods/lora.toml', {}) is False; "
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


@pytest.mark.parametrize(
    "module_name",
    [
        "web.services.config.datasets",
        "web.services.config.file_groups",
        "web.services.config.merge",
        "web.services.config.output_runs",
    ],
)
def test_config_module_facade_sync_preserves_legacy_raw_file_shims(module_name: str):
    import importlib

    module = importlib.import_module(module_name)
    legacy_config._restore_raw_files_shims()
    raw_file_shims = legacy_config._RAW_FILES_SHIMS
    file_group_shims = legacy_config._FILE_GROUPS_SHIMS

    module._sync_from_facade()

    for name, shim in raw_file_shims.items():
        assert getattr(legacy_config, name) is shim
    for name, shim in file_group_shims.items():
        assert getattr(legacy_config, name) is shim
    for name in (
        "load_raw_file",
        "save_raw_file",
        "delete_raw_file",
        "patch_raw_file_values",
        "preview_raw_file_patch",
    ):
        assert getattr(module, name) is getattr(config_service, name)


def test_raw_patch_preview_route_does_not_write_config_file(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    config_path = configs / "imported" / "lora.toml"
    original = config_path.read_text(encoding="utf-8")

    response = asyncio.run(
        config_routes.handle_raw_patch_preview(
            _JsonRequest(
                {
                    "file": "configs/imported/lora.toml",
                    "values": {"output_name": "preview-only"},
                }
            ),  # type: ignore[arg-type]
        )
    )
    payload = _json_response_payload(response)

    assert response.status == 200
    assert payload["ok"] is True
    assert payload["changed"] == ["output_name"]
    assert 'output_name = "preview-only"' in payload["content"]
    assert config_path.read_text(encoding="utf-8") == original


def test_raw_save_as_route_never_overwrites_existing_config(
    tmp_path: Path,
    monkeypatch,
):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    config_path = configs / "imported" / "lora.toml"
    original = config_path.read_text(encoding="utf-8")

    response = asyncio.run(
        config_routes.handle_raw_save_as(
            _JsonRequest(
                {
                    "file": "configs/imported/lora.toml",
                    "content": 'output_name = "should-not-overwrite"\n',
                }
            ),  # type: ignore[arg-type]
        )
    )
    payload = _json_response_payload(response)

    assert response.status == 400
    assert payload["ok"] is False
    assert "配置文件已存在" in payload["error"]
    assert config_path.read_text(encoding="utf-8") == original


def test_raw_put_route_rejects_invalid_toml_without_creating_file(
    tmp_path: Path,
    monkeypatch,
):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    target = configs / "imported" / "broken.toml"

    response = asyncio.run(
        config_routes.handle_raw_put(
            _JsonRequest(
                {
                    "file": "configs/imported/broken.toml",
                    "content": "output_name = [broken\n",
                }
            ),  # type: ignore[arg-type]
        )
    )
    payload = _json_response_payload(response)

    assert response.status == 400
    assert payload["ok"] is False
    assert "TOML 语法错误" in payload["error"]
    assert target.exists() is False


def test_raw_patch_route_rejects_non_object_values(
    tmp_path: Path,
    monkeypatch,
):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    config_path = configs / "imported" / "lora.toml"
    original = config_path.read_text(encoding="utf-8")

    response = asyncio.run(
        config_routes.handle_raw_patch(
            _JsonRequest(
                {
                    "file": "configs/imported/lora.toml",
                    "values": ["output_name", "bad"],
                }
            ),  # type: ignore[arg-type]
        )
    )
    payload = _json_response_payload(response)

    assert response.status == 400
    assert payload["ok"] is False
    assert "字段补丁格式不合法" in payload["error"]
    assert config_path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("value", ["bf16", "fp16", "fp32"])
def test_raw_patch_persists_preprocess_precision_preference(
    tmp_path: Path, monkeypatch, value: str
):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    train_rel = "configs/imported/lora.toml"

    ok, msg, content, changed = config_service.patch_raw_file_values(
        train_rel,
        {
            "preprocess_precision_preference": value,
        },
    )

    assert ok is True, msg
    assert changed == ["preprocess_precision_preference"]
    assert f'preprocess_precision_preference = "{value}"' in content
    assert f'preprocess_precision_preference = "{value}"' in (
        configs / "imported" / "lora.toml"
    ).read_text(encoding="utf-8")


def test_raw_patch_rejects_blank_output_name(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    train_rel = "configs/imported/lora.toml"
    original = (configs / "imported" / "lora.toml").read_text(encoding="utf-8")

    ok, msg, content, changed = config_service.patch_raw_file_values(
        train_rel,
        {
            "output_name": "   ",
        },
    )

    assert ok is False
    assert "output_name 不能为空" in msg
    assert content == ""
    assert changed == []
    assert (configs / "imported" / "lora.toml").read_text(encoding="utf-8") == original


def test_raw_patch_clears_optional_sample_schedule_fields(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    train_rel = "configs/imported/lora.toml"
    (configs / "imported" / "lora.toml").write_text(
        "\n".join(
            [
                'output_name = "anima"',
                "sample_every_n_epochs = 2",
                "sample_every_n_steps = 300",
                "",
            ]
        ),
        encoding="utf-8",
    )

    ok, msg, content, changed = config_service.patch_raw_file_values(
        train_rel,
        {
            "sample_every_n_epochs": "",
            "sample_every_n_steps": None,
        },
    )

    assert ok is True, msg
    assert changed == ["sample_every_n_epochs", "sample_every_n_steps"]
    assert 'output_name = "anima"' in content
    assert "sample_every_n_epochs" not in content
    assert "sample_every_n_steps" not in content
    saved = toml.loads((configs / "imported" / "lora.toml").read_text(encoding="utf-8"))
    assert saved == {"output_name": "anima"}


def test_raw_patch_writes_non_blank_sample_schedule_fields_as_int(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)

    ok, msg, content, changed = config_service.patch_raw_file_values(
        "configs/imported/lora.toml",
        {
            "sample_every_n_epochs": "4",
            "sample_every_n_steps": 500,
        },
    )

    assert ok is True, msg
    assert changed == ["sample_every_n_epochs", "sample_every_n_steps"]
    parsed = toml.loads(content)
    assert parsed["sample_every_n_epochs"] == 4
    assert parsed["sample_every_n_steps"] == 500


def test_raw_patch_clears_optional_max_train_epochs_field(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    train_rel = "configs/imported/lora.toml"
    (configs / "imported" / "lora.toml").write_text(
        "\n".join(
            [
                'output_name = "anima"',
                "max_train_epochs = 4",
                "max_train_steps = 1200",
                "",
            ]
        ),
        encoding="utf-8",
    )

    ok, msg, content, changed = config_service.patch_raw_file_values(
        train_rel,
        {
            "max_train_epochs": "",
        },
    )

    assert ok is True, msg
    assert changed == ["max_train_epochs"]
    assert "max_train_epochs" not in content
    saved = toml.loads((configs / "imported" / "lora.toml").read_text(encoding="utf-8"))
    assert saved == {"output_name": "anima", "max_train_steps": 1200}


def test_raw_patch_writes_non_blank_max_train_epochs_as_int(tmp_path: Path, monkeypatch):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)

    ok, msg, content, changed = config_service.patch_raw_file_values(
        "configs/imported/lora.toml",
        {
            "max_train_epochs": "6",
        },
    )

    assert ok is True, msg
    assert changed == ["max_train_epochs"]
    parsed = toml.loads(content)
    assert parsed["max_train_epochs"] == 6


def test_raw_patch_writes_spd_fields_to_nested_tables(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    spd_path = configs / "imported" / "spd.toml"
    spd_path.write_text(
        "\n".join(
            [
                'dit_path = "models/anima.safetensors"',
                'data_dir = "post_image_dataset/lora"',
                "iterations = 4000",
                "channel_scaling_alpha = 0.1",
                "weight_decay = 0.2",
                "",
                "[network]",
                "rank = 48",
                "channel_scaling_alpha = 0.5",
                "",
                "[optim]",
                "lr = 2e-5",
                "weight_decay = 0.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)

    ok, msg, content, changed = config_service.patch_raw_file_values(
        "configs/imported/spd.toml",
        {
            "channel_scaling_alpha": 0.25,
            "weight_decay": 0.01,
        },
    )

    assert ok is True, msg
    assert changed == ["channel_scaling_alpha", "weight_decay"]
    parsed = toml.loads(content)
    assert parsed["network"]["channel_scaling_alpha"] == 0.25
    assert parsed["optim"]["weight_decay"] == 0.01
    assert "channel_scaling_alpha" not in parsed
    assert "weight_decay" not in parsed
    saved = toml.loads(spd_path.read_text(encoding="utf-8"))
    assert saved["network"]["channel_scaling_alpha"] == 0.25
    assert saved["optim"]["weight_decay"] == 0.01


def test_raw_patch_removes_retired_config_fields(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    train_rel = "configs/imported/lora.toml"
    retired_keys = [
        "per_channel_scaling",
        "repa_layer",
        "repa_lr_scale",
        "repa_weight",
        "trim_crossattn_kv",
        "use_fei_router",
        "use_hydra",
        "use_repa",
        "use_sigma_router",
    ]
    (configs / "imported" / "lora.toml").write_text(
        "\n".join(
            [
                'output_name = "anima"',
                "per_channel_scaling = true",
                "repa_layer = 8",
                "repa_lr_scale = 1.0",
                "repa_weight = 0.5",
                "trim_crossattn_kv = true",
                "use_fei_router = true",
                "use_hydra = true",
                "use_repa = true",
                "use_sigma_router = true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)

    ok, msg, content, changed = config_service.patch_raw_file_values(
        train_rel,
        {
            "per_channel_scaling": False,
            "repa_layer": 0,
            "repa_lr_scale": 0,
            "repa_weight": 0,
            "trim_crossattn_kv": False,
            "use_fei_router": False,
            "use_hydra": False,
            "use_repa": False,
            "use_sigma_router": False,
            "output_name": "clean",
        },
    )

    assert ok is True, msg
    assert changed == sorted(["output_name", *retired_keys])
    assert 'output_name = "clean"' in content
    for key in retired_keys:
        assert key not in content
    saved = (configs / "imported" / "lora.toml").read_text(encoding="utf-8")
    for key in retired_keys:
        assert key not in saved


def test_raw_patch_auto_fixes_came_two_beta_optimizer_args(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    train_rel = "configs/imported/lora.toml"
    (configs / "imported" / "lora.toml").write_text(
        "\n".join(
            [
                'output_name = "anima"',
                'optimizer_type = "CAME"',
                'optimizer_args = ["weight_decay=0.01", "eps=1e-8", "betas=0.9,0.99"]',
                "",
            ]
        ),
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)

    ok, msg, content, changed = config_service.patch_raw_file_values(
        train_rel,
        {
            "output_name": "clean",
        },
    )

    assert ok is True, msg
    assert changed == ["optimizer_args", "output_name"]
    parsed = toml.loads(content)
    assert parsed["output_name"] == "clean"
    assert parsed["optimizer_args"] == ["weight_decay=0.01", "eps=1e-8", "betas=0.9,0.999,0.9999"]
    saved = toml.loads((configs / "imported" / "lora.toml").read_text(encoding="utf-8"))
    assert saved["optimizer_args"][-1] == "betas=0.9,0.999,0.9999"


def test_save_raw_file_rejects_blank_output_name(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)

    ok, msg = config_service.save_raw_file(
        "configs/imported/blank.toml",
        'output_name = "   "\n',
    )

    assert ok is False
    assert "output_name 不能为空" in msg
    assert not (configs / "imported" / "blank.toml").exists()


def test_save_raw_file_auto_fixes_came_two_beta_optimizer_args(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)

    ok, msg = config_service.save_raw_file(
        "configs/imported/came.toml",
        "\n".join(
            [
                'optimizer_type = "CAME"',
                'optimizer_args = ["weight_decay=0.01", "eps=1e-8", "betas=0.9,0.99"]',
                "",
            ]
        ),
    )

    assert ok is True, msg
    saved = toml.loads((configs / "imported" / "came.toml").read_text(encoding="utf-8"))
    assert saved["optimizer_args"] == ["weight_decay=0.01", "eps=1e-8", "betas=0.9,0.999,0.9999"]


def test_save_raw_file_does_not_touch_non_came_two_beta_optimizer_args(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)

    ok, msg = config_service.save_raw_file(
        "configs/imported/adamw.toml",
        "\n".join(
            [
                'optimizer_type = "AdamW"',
                'optimizer_args = ["weight_decay=0.01", "betas=0.9,0.99"]',
                "",
            ]
        ),
    )

    assert ok is True, msg
    saved_text = (configs / "imported" / "adamw.toml").read_text(encoding="utf-8")
    assert 'optimizer_args = ["weight_decay=0.01", "betas=0.9,0.99"]' in saved_text
    saved = toml.loads(saved_text)
    assert saved["optimizer_args"][-1] == "betas=0.9,0.99"


def test_save_raw_file_keeps_came_three_beta_optimizer_args(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)

    ok, msg = config_service.save_raw_file(
        "configs/imported/came_ready.toml",
        "\n".join(
            [
                'optimizer_type = "CAME"',
                'optimizer_args = ["weight_decay=0.01", "eps=1e-8", "betas=0.9,0.99,0.999"]',
                "",
            ]
        ),
    )

    assert ok is True, msg
    saved_text = (configs / "imported" / "came_ready.toml").read_text(encoding="utf-8")
    assert 'optimizer_args = ["weight_decay=0.01", "eps=1e-8", "betas=0.9,0.99,0.999"]' in saved_text
    saved = toml.loads(saved_text)
    assert saved["optimizer_args"][-1] == "betas=0.9,0.99,0.999"


def test_raw_file_helpers_remain_available_from_legacy_module(tmp_path: Path, monkeypatch):
    import importlib

    raw_files_module = importlib.import_module("web.services.config.raw_files")
    raw_file_shims = legacy_config._RAW_FILES_SHIMS
    assert legacy_config._safe_resolve is not config_service._safe_resolve
    expected_shims = (
        "load_raw_file",
        "save_raw_file",
        "delete_raw_file",
        "patch_raw_file_values",
        "preview_raw_file_patch",
        "_prepare_raw_file_patch",
        "_restore_dataset_config_after_failed_train_patch",
        "_patch_toml_top_level",
        "_is_spd_patch_target",
        "_remove_retired_top_level_fields",
        "_normalize_patch_value",
        "_normalize_saved_raw_config_content",
        "_normalize_saved_raw_config_content_with_changed_keys",
        "_is_blank_output_name",
    )
    assert tuple(legacy_config._RAW_FILES_SHIM_NAMES) == expected_shims
    assert tuple(raw_file_shims) == expected_shims
    for name in expected_shims:
        assert callable(getattr(raw_files_module, name))
        assert name in raw_file_shims
        assert (
            raw_file_shims[name].__doc__
            == f"Compatibility shim forwarding to web.services.config.raw_files.{name}."
        )
        assert getattr(legacy_config, name) is raw_file_shims[name]

    _write_minimal_config_tree(tmp_path)
    configs = tmp_path / "configs"
    _patch_config_service_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(legacy_config, "ROOT", tmp_path)
    monkeypatch.setattr(legacy_config, "CONFIGS_DIR", configs)
    monkeypatch.setattr(legacy_config, "DATASET_PRESETS_DIR", configs / "datasets")
    monkeypatch.setattr(legacy_config, "GUI_METHODS_DIR", configs / "gui-methods")
    monkeypatch.setattr(legacy_config, "IMPORTED_CONFIGS_DIR", configs / "imported")
    monkeypatch.setattr(legacy_config, "PRESETS_FILE", configs / "presets.toml")
    monkeypatch.setattr(legacy_config, "WEB_FILE_GROUPS_FILE", configs / "web-file-groups.toml")
    monkeypatch.setattr(legacy_config, "WEB_USER_LOCKS_FILE", configs / "web-user-locks.toml")

    assert legacy_config.load_raw_file("../outside.toml") == ""
    outside_ok, outside_msg = legacy_config.save_raw_file(
        "../outside.toml",
        'output_name = "outside"\n',
    )
    assert outside_ok is False
    assert "路径不合法" in outside_msg

    outside_patch_ok, outside_patch_msg, outside_path, outside_content, outside_changed = (
        legacy_config._prepare_raw_file_patch(
            "../outside.toml",
            {"output_name": "outside"},
        )
    )
    assert outside_patch_ok is False
    assert "路径不合法" in outside_patch_msg
    assert outside_path is None
    assert outside_content == ""
    assert outside_changed == []

    outside_delete_ok, outside_delete_msg = legacy_config.delete_raw_file("../outside.toml")
    assert outside_delete_ok is False
    assert "路径不合法" in outside_delete_msg

    rel_path = "configs/imported/legacy_raw.toml"
    ok, msg = legacy_config.save_raw_file(
        rel_path,
        "\n".join(
            [
                'output_name = "legacy_raw"',
                'optimizer_type = "CAME"',
                'optimizer_args = ["betas=0.9,0.999"]',
                "use_hydra = true",
                "",
            ]
        ),
    )

    assert ok is True, msg
    for name in expected_shims:
        assert getattr(legacy_config, name) is raw_file_shims[name]
    loaded = legacy_config.load_raw_file(rel_path)
    assert 'output_name = "legacy_raw"' in loaded
    assert 'betas=0.9,0.999,0.9999' in loaded

    preview_ok, preview_msg, path, preview_content, preview_changed = legacy_config._prepare_raw_file_patch(
        rel_path,
        {
            "output_name": "legacy_next",
            "precision_preference": "fp16",
            "use_hydra": False,
        },
    )

    assert preview_ok is True, preview_msg
    assert path is not None
    assert preview_changed == ["output_name", "use_hydra"]
    assert 'output_name = "legacy_next"' in preview_content
    assert "precision_preference" not in preview_content
    assert "use_hydra" not in preview_content

    patched_ok, patched_msg, patched_content, changed = legacy_config.patch_raw_file_values(
        rel_path,
        {
            "output_name": "legacy_next",
            "precision_preference": "fp16",
            "use_hydra": False,
        },
    )

    assert patched_ok is True, patched_msg
    assert changed == ["output_name", "use_hydra"]
    assert patched_content == preview_content
    saved = (configs / "imported" / "legacy_raw.toml").read_text(encoding="utf-8")
    assert saved == patched_content
    assert "precision_preference" not in saved
    assert "use_hydra" not in saved

    public_preview_ok, public_preview_msg, public_preview_content, public_preview_changed = (
        legacy_config.preview_raw_file_patch(
            rel_path,
            {"output_name": "legacy_preview"},
            content=patched_content,
        )
    )

    assert public_preview_ok is True, public_preview_msg
    assert public_preview_changed == ["output_name"]
    assert 'output_name = "legacy_preview"' in public_preview_content
    assert (configs / "imported" / "legacy_raw.toml").read_text(encoding="utf-8") == patched_content

    spd_content = legacy_config._patch_toml_top_level(
        "\n".join(
            [
                'dit_path = "model.safetensors"',
                'data_dir = "image_dataset"',
                "iterations = 1",
                "weight_decay = 0.1",
                "[schedule]",
                "",
            ]
        ),
        {"weight_decay": 0.2},
        rel_path="configs/methods/spd.toml",
    )
    spd_data = toml.loads(spd_content)
    assert "weight_decay" not in spd_data
    assert spd_data["optim"]["weight_decay"] == 0.2

    cleaned_content, removed_keys = legacy_config._remove_retired_top_level_fields(
        'output_name = "legacy_next"\nuse_hydra = true\n'
    )
    assert removed_keys == ["use_hydra"]
    assert "use_hydra" not in cleaned_content

    delete_ok, delete_msg = legacy_config.delete_raw_file(rel_path)
    assert delete_ok is True, delete_msg
    assert not (configs / "imported" / "legacy_raw.toml").exists()
    assert legacy_config.load_raw_file(rel_path) == ""
    for name in expected_shims:
        assert getattr(legacy_config, name) is raw_file_shims[name]


def test_patch_raw_file_rejects_invalid_choice(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    train_rel = "configs/imported/lora.toml"
    original = (configs / "imported" / "lora.toml").read_text(encoding="utf-8")

    ok, msg, content, changed = config_service.patch_raw_file_values(
        train_rel,
        {
            "preprocess_precision_preference": "nope",
        },
    )

    assert ok is False
    assert "preprocess_precision_preference" in msg
    assert content == ""
    assert changed == []
    assert (configs / "imported" / "lora.toml").read_text(encoding="utf-8") == original


def test_patch_raw_file_warns_unknown_key_but_still_saves(tmp_path: Path, monkeypatch):
    # unknown key is warning-only for preflight; raw patch currently allows custom fields
    # unless choices/type fails. Keep custom key writable for compatibility.
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    train_rel = "configs/imported/lora.toml"

    ok, msg, content, changed = config_service.patch_raw_file_values(
        train_rel,
        {
            "custom_user_flag": True,
        },
    )
    assert ok is True, msg
    assert "custom_user_flag" in changed
    assert "custom_user_flag" in content

