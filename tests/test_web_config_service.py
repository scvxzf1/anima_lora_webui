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
from web.services.config import (
    dataset_editor,
    dataset_media,
    dataset_preset_paths,
    dataset_presets_api,
    dataset_rows,
    file_group_runtime,
    file_groups,
    merge,
    output_runs,
    preflight_runtime,
    raw_files,
    sample_prompts,
)
from web.services.config import datasets as config_datasets
from web.services.config import metadata as config_metadata
from web.services.config import paths as config_paths


def _json_response_payload(response) -> dict[str, Any]:
    return json.loads(response.text or "{}")


class _QueryRequest:
    def __init__(self, query: dict[str, str]):
        self.query = query


class _JsonRequest:
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    async def json(self) -> dict[str, Any]:
        return self._payload

def test_config_metadata_exports_remain_available_from_legacy_facade():
    for name in config_metadata.__all__:
        assert getattr(config_service, name) is getattr(config_metadata, name)
        assert getattr(legacy_config, name) is getattr(config_metadata, name)

    assert config_service.get_field_help is config_metadata.get_field_help
    assert config_service.get_groups is config_metadata.get_groups
    assert legacy_config.get_groups is config_metadata.get_groups

    assert config_metadata.get_field_help()["network_dim"]["en"]
    groups = config_metadata.get_groups()
    assert "Architecture" in groups["groups"]
    assert "learning_rate" in groups["basic"]
    performance = groups["groups"]["Performance"]
    assert "compile_dynamic_seq" in performance
    assert "v100_flash_stability" in performance
    assert "debug_finite_checks" in performance

def test_metadata_module_imports_without_facade_cycle():
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    script = (
        "import sys; "
        "import web.services.config.metadata as metadata; "
        "assert callable(metadata.get_field_help); "
        "assert 'configs/base.toml' in metadata.SYSTEM_PRESET_FILES; "
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

def test_paths_module_imports_without_facade_cycle(tmp_path):
    root = (tmp_path / "anima-root").resolve()
    configs = root / "configs"
    configs.mkdir(parents=True)
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "ANIMA_TEST_PATH_ROOT": str(root),
    }
    script = (
        "import os, sys; "
        "from pathlib import Path; "
        "import web.services.config.paths as paths; "
        "root = Path(os.environ['ANIMA_TEST_PATH_ROOT']).resolve(); "
        "configs = (root / 'configs').resolve(); "
        "target = (configs / 'base.toml').resolve(); "
        "assert paths.normalize_config_rel_path('/configs/base.toml') == 'configs/base.toml'; "
        "assert paths.safe_resolve('configs/base.toml', root=root, configs_dir=configs) == target; "
        "assert paths.safe_resolve('configs/../outside.toml', root=root, configs_dir=configs) is None; "
        "assert paths.safe_resolve(str(target), root=root, configs_dir=configs) == target; "
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
    ("module_name", "expected_callable"),
    [
        ("web.services.config.datasets", "_build_dataset_config_doc"),
        ("web.services.config.file_groups", "_is_dataset_preset_readonly"),
    ],
)
def test_high_coupling_config_modules_import_without_facade_cycle(
    module_name: str,
    expected_callable: str,
):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    script = (
        "import importlib, sys; "
        f"module = importlib.import_module({module_name!r}); "
        f"assert callable(getattr(module, {expected_callable!r})); "
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

def test_config_path_helpers_reject_escaping_and_facade_stays_compatible(tmp_path: Path, monkeypatch):
    root = tmp_path
    configs = root / "configs"
    configs.mkdir()
    config_file = configs / "demo.toml"
    config_file.write_text("output_name = 'demo'\n", encoding="utf-8")
    monkeypatch.setattr(config_service, "ROOT", root)
    monkeypatch.setattr(config_service, "CONFIGS_DIR", configs)

    assert config_paths.normalize_config_rel_path("\\configs\\demo.toml") == "configs/demo.toml"
    assert config_paths.safe_resolve("configs/demo.toml", root=root, configs_dir=configs) == config_file.resolve()
    assert config_paths.safe_resolve("../outside.toml", root=root, configs_dir=configs) is None
    assert config_paths.safe_config_subdir("gui-methods", configs_dir=configs) == (configs / "gui-methods").resolve()
    assert config_paths.safe_config_subdir("../outside", configs_dir=configs) is None
    assert config_paths.resolve_project_path(
        "$ROOT/configs/demo.toml",
        root=root,
        expand_env_vars_fn=lambda value: value.replace("$ROOT", str(root)),
    ) == config_file.resolve()
    assert config_paths.resolve_display_path(
        "configs/demo.toml",
        root=root,
        configs_dir=configs,
        expand_env_vars_fn=lambda value: value,
    ) == config_file.resolve()
    assert config_paths.display_path(config_file, root=root) == "configs/demo.toml"
    assert config_service._safe_resolve("configs/demo.toml") == config_file.resolve()
    assert config_service._safe_resolve("../outside.toml") is None

def test_config_path_helpers_support_external_configs_root(tmp_path: Path):
    root = tmp_path / "repo"
    configs = tmp_path / "external-configs"
    root.mkdir()
    configs.mkdir()
    config_file = configs / "gui-methods" / "demo.toml"
    config_file.parent.mkdir()
    config_file.write_text("output_name = 'demo'\n", encoding="utf-8")

    assert config_paths.safe_resolve("configs/gui-methods/demo.toml", root=root, configs_dir=configs) == config_file.resolve()
    assert config_paths.safe_resolve("gui-methods/demo.toml", root=root, configs_dir=configs) == config_file.resolve()
    assert config_paths.safe_resolve(str(config_file), root=root, configs_dir=configs) == config_file.resolve()
    assert config_paths.safe_resolve("configs/../outside.toml", root=root, configs_dir=configs) is None
    assert config_paths.resolve_display_path(
        "configs/gui-methods/demo.toml",
        root=root,
        configs_dir=configs,
        expand_env_vars_fn=lambda value: value,
    ) == config_file.resolve()
    assert config_paths.resolve_display_path(
        "post_image_dataset/demo",
        root=root,
        configs_dir=configs,
        expand_env_vars_fn=lambda value: value,
    ) == (root / "post_image_dataset/demo").resolve()
    assert config_paths.display_path(config_file, root=root, configs_dir=configs) == "configs/gui-methods/demo.toml"

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
    (gui_methods / "loha.toml").write_text(
        '[variant]\nfamily = "lora"\norder = 11\n',
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

    assert config_service.list_variants("lora") == ["lora", "loha", "lokr", "custom/user_variant"]
    assert config_service.list_variants("hydralora") == [
        "hydralora-8gb",
        "hydralora",
        "custom/user_variant",
    ]
    assert config_service.CONFIG_FILE_LABELS_ZH["configs/gui-methods/glora.toml"] == "GLoRA 训练变体"
    assert config_service.CONFIG_FILE_LABELS_ZH["configs/gui-methods/vera.toml"] == "VeRA 训练变体"

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

    ok, msg, content, changed, _warnings = config_service.preview_raw_file_patch(
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

def test_common_config_helpers_import_without_facade_cycle():
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    script = (
        "import sys; "
        "import web.services.config.common as common; "
        "assert 'web.services.config_service' not in sys.modules; "
        "assert 'web.services.config._legacy' not in sys.modules; "
        "assert common._positive_int('3', 1) == 3; "
        "assert common._positive_int('0', 1) == 1; "
        "assert common._positive_int_or_none('2') == 2; "
        "assert common._positive_int_or_none('0') is None; "
        "assert common._nonnegative_int('-1', 5) == 5; "
        "assert common._nonnegative_float('0.5', 1.0) == 0.5; "
        "assert common._positive_float('0', 1.0) == 1.0; "
        "assert common._bool_value('yes') is True; "
        "assert 'web.services.config_service' not in sys.modules; "
        "assert 'web.services.config._legacy' not in sys.modules"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

def test_common_config_path_helpers_work_without_facade_snapshot(tmp_path: Path, monkeypatch):
    from web.services.config import common as common_impl

    configs = tmp_path / "configs"
    configs.mkdir()
    imported = configs / "imported"
    imported.mkdir()
    config_file = imported / "demo.toml"
    config_file.write_text('path = "$HOME/demo"\ncount = 2\n', encoding="utf-8")

    monkeypatch.setattr(common_impl, "ROOT", tmp_path)
    monkeypatch.setattr(common_impl, "CONFIGS_DIR", configs)

    loaded = common_impl._load(config_file)

    assert loaded["count"] == 2
    assert loaded["path"].endswith("/demo")
    assert common_impl._safe_resolve("configs/imported/demo.toml") == config_file.resolve()
    assert common_impl._safe_config_subdir("imported") == imported.resolve()
    assert common_impl._resolve_project_path("image_dataset/hero") == (
        tmp_path / "image_dataset" / "hero"
    ).resolve()
    assert common_impl._display_path(config_file) == "configs/imported/demo.toml"

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

def _patch_config_service_paths(monkeypatch, root: Path) -> None:
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
        for module in (config_service, legacy_config):
            monkeypatch.setattr(module, name, value)
        monkeypatch.setattr(raw_files_impl, name, value, raising=False)

def _patch_external_config_service_paths(monkeypatch, root: Path, configs: Path) -> None:
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
        for module in (config_service, legacy_config):
            monkeypatch.setattr(module, name, value)
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


CONFIG_ROOT_DOMAIN_MODULES = (
    merge,
    preflight_runtime,
    raw_files,
    sample_prompts,
    file_groups,
    file_group_runtime,
    config_datasets,
    dataset_media,
    dataset_editor,
    dataset_rows,
    dataset_preset_paths,
    dataset_presets_api,
    output_runs,
)


def test_set_configs_root_broadcasts_to_all_domain_modules(tmp_path, monkeypatch):
    target = (tmp_path / "broadcast-configs").resolve()
    target.mkdir(parents=True)

    for module in (config_service, *CONFIG_ROOT_DOMAIN_MODULES):
        if hasattr(module, "CONFIGS_DIR"):
            monkeypatch.setattr(
                module, "CONFIGS_DIR", getattr(module, "CONFIGS_DIR"), raising=False
            )

    resolved = config_service.set_configs_root(target)

    assert resolved == target
    assert config_service.CONFIGS_DIR == target
    for module in CONFIG_ROOT_DOMAIN_MODULES:
        assert getattr(module, "CONFIGS_DIR") == target
        if hasattr(module, "DATASET_PRESETS_DIR"):
            assert Path(getattr(module, "DATASET_PRESETS_DIR")) == target / "datasets"
        if hasattr(module, "IMPORTED_CONFIGS_DIR"):
            assert Path(getattr(module, "IMPORTED_CONFIGS_DIR")) == target / "imported"
        if hasattr(module, "GUI_METHODS_DIR"):
            assert Path(getattr(module, "GUI_METHODS_DIR")) == target / "gui-methods"
