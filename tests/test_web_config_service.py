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


def test_config_metadata_exports_remain_available_from_legacy_facade():
    names = [
        "CAPTION_SOURCE_AUTO",
        "CAPTION_SOURCE_TXT",
        "CAPTION_SOURCE_JSON",
        "CAPTION_SOURCE_CAPTIONS_JSON",
        "CONFIG_FILE_LABELS_ZH",
        "SYSTEM_CONFIG_GROUP_IDS",
        "FIXED_SYSTEM_CONFIG_GROUP_IDS",
        "FILE_MOVE_TARGET_GROUPS",
        "USER_LOCKABLE_GROUPS",
        "HIDDEN_CONFIG_FILES",
        "SYSTEM_PRESET_FILES",
        "SYSTEM_DATASET_PRESET_FILES",
        "HIDDEN_DATASET_PRESET_FILES",
        "OUTPUT_RUN_CONFIG_FILES",
        "SUPPORTED_TRAINING_SAMPLE_SAMPLERS",
        "LEGACY_TRAINING_SAMPLE_SAMPLERS",
        "PREPROCESS_ENV_CHECK_KEY",
        "PREPROCESS_ENV_REQUIRED_FILES",
        "UI_ONLY_CONFIG_FIELDS",
        "SPD_NESTED_PATCH_FIELDS",
        "RETIRED_TOP_LEVEL_CONFIG_FIELDS",
        "DATASET_IMAGE_EXTS",
        "DATASET_PREVIEW_LIMIT",
        "DATASET_CAPTION_MAX_CHARS",
        "DEFAULT_RESIZED_IMAGE_DIR",
        "DEFAULT_LORA_CACHE_DIR",
        "PREPROCESS_DATASET_SETTING_KEYS",
        "CAPTION_SOURCE_MODE_LABELS",
    ]

    for name in names:
        assert getattr(config_service, name) is getattr(config_metadata, name)
        assert getattr(legacy_config, name) is getattr(config_metadata, name)

    assert config_service.get_field_help is config_metadata.get_field_help
    assert legacy_config.get_groups is config_metadata.get_groups


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


def test_sample_prompts_module_imports_without_facade_cycle():
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    script = (
        "import sys; "
        "import web.services.config.sample_prompts; "
        "assert 'web.services.config_service' not in sys.modules"
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


def test_merge_module_imports_without_facade_cycle():
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    script = (
        "import sys; "
        "import web.services.config.merge as merge; "
        "assert callable(merge.list_methods); "
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


def test_output_runs_module_imports_without_facade_cycle():
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    script = (
        "import sys; "
        "import web.services.config.output_runs as output_runs; "
        "assert callable(output_runs.list_output_runs); "
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


def test_file_groups_direct_helpers_work_without_facade_cycle():
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    script = (
        "import sys; "
        "import web.services.config.file_groups as file_groups; "
        "assert file_groups._safe_archive_name('我的配置/分组') == '我的配置_分组'; "
        "assert file_groups._place_index('2', 5) == 2; "
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


def test_file_groups_direct_path_helpers_work_without_facade_snapshot(tmp_path: Path):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    script = f"""
import sys
from pathlib import Path

import web.services.config.file_groups as file_groups

root = Path({str(tmp_path)!r})
configs = root / "configs"
(configs / "imported").mkdir(parents=True)
(configs / "imported" / "lora.toml").write_text('output_name = "lora"\\n', encoding="utf-8")
(configs / "gui-methods").mkdir()
(configs / "datasets").mkdir()

file_groups.ROOT = root
file_groups.CONFIGS_DIR = configs
file_groups.GUI_METHODS_DIR = configs / "gui-methods"
file_groups.IMPORTED_CONFIGS_DIR = configs / "imported"
file_groups.PRESETS_FILE = configs / "presets.toml"
file_groups.WEB_FILE_GROUPS_FILE = configs / "web-file-groups.toml"
file_groups.WEB_USER_LOCKS_FILE = configs / "web-user-locks.toml"
file_groups.DATASET_PRESETS_DIR = configs / "datasets"

assert file_groups._load(configs / "imported" / "lora.toml") == dict(output_name="lora")
assert file_groups._safe_resolve("configs/imported/lora.toml") == (configs / "imported" / "lora.toml").resolve()
assert file_groups._display_path(configs / "imported" / "lora.toml") == "configs/imported/lora.toml"
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


def test_merge_helpers_remain_available_from_legacy_module(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    methods = configs / "methods"
    methods.mkdir()
    (methods / "spd.toml").write_text('output_name = "spd"\n', encoding="utf-8")
    gui_methods = configs / "gui-methods"
    gui_methods.mkdir()
    (gui_methods / "lora.toml").write_text(
        "\n".join([
            'output_name = "lora"',
            "[variant]",
            'family = "lora"',
            "order = 1",
            "",
        ]),
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(legacy_config, "ROOT", tmp_path)
    monkeypatch.setattr(legacy_config, "CONFIGS_DIR", configs)
    monkeypatch.setattr(legacy_config, "DATASET_PRESETS_DIR", configs / "datasets")
    monkeypatch.setattr(legacy_config, "GUI_METHODS_DIR", gui_methods)
    monkeypatch.setattr(legacy_config, "IMPORTED_CONFIGS_DIR", configs / "imported")
    monkeypatch.setattr(legacy_config, "PRESETS_FILE", configs / "presets.toml")
    monkeypatch.setattr(legacy_config, "WEB_FILE_GROUPS_FILE", configs / "web-file-groups.toml")
    monkeypatch.setattr(legacy_config, "WEB_USER_LOCKS_FILE", configs / "web-user-locks.toml")

    expected_shims = (
        "list_methods",
        "list_variants",
        "list_all_variants",
        "list_presets",
        "load_merged_config",
        "suggest_data_dirs",
        "suggest_dataset_dirs",
        "apply_auto_data_dirs",
    )
    assert tuple(legacy_config._MERGE_SHIM_NAMES) == expected_shims
    for name in expected_shims:
        assert getattr(legacy_config, name) is legacy_config._MERGE_SHIMS[name]
        assert (
            getattr(legacy_config, name).__doc__
            == f"Compatibility shim forwarding to web.services.config.merge.{name}."
        )
    legacy_config._restore_raw_files_shims()
    raw_file_shims = legacy_config._RAW_FILES_SHIMS
    assert legacy_config.load_raw_file is raw_file_shims["load_raw_file"]

    assert config_service.list_variants("spd") == ["spd"]
    for name, shim in raw_file_shims.items():
        assert getattr(legacy_config, name) is shim

    assert "spd" in legacy_config.list_methods()
    assert legacy_config.list_variants("spd") == ["spd"]
    assert legacy_config.list_variants("lora") == ["lora"]
    assert legacy_config.list_all_variants() == ["lora"]
    assert legacy_config.list_presets() == ["default"]

    merged = legacy_config.load_merged_config("lora", "default")
    assert merged["max_train_steps"] == 0
    assert merged["source_image_dir"] == "image_dataset"
    assert legacy_config.suggest_data_dirs("image_dataset/hero") == {
        "ok": True,
        "source_image_dir": "image_dataset/hero",
        "resized_image_dir": "image_dataset/hero_resized",
        "lora_cache_dir": "image_dataset/hero_lora_cache",
    }
    assert legacy_config.suggest_dataset_dirs(["image_dataset/hero"]) == {
        "ok": True,
        "datasets": [{
            "index": 0,
            "source_dir": "image_dataset/hero",
            "image_dir": "image_dataset/hero_resized",
            "cache_dir": "image_dataset/hero_lora_cache",
        }],
    }

    auto_dirs = legacy_config.apply_auto_data_dirs({"source_image_dir": "image_dataset/hero"})
    assert auto_dirs["resized_image_dir"] == "image_dataset/hero_resized"
    assert auto_dirs["lora_cache_dir"] == "image_dataset/hero_lora_cache"
    for name in expected_shims:
        assert getattr(legacy_config, name) is legacy_config._MERGE_SHIMS[name]


def test_legacy_merge_private_helpers_forward_to_split_module(monkeypatch):
    from web.services.config import merge as merge_impl

    sentinel_path = Path("configs/gui-methods/lora.toml")
    monkeypatch.setattr(
        merge_impl,
        "_builtin_variants_by_family",
        lambda: {"lora": [(7, "sentinel")]},
    )
    monkeypatch.setattr(
        merge_impl,
        "_read_variant_metadata",
        lambda path: {"path": str(path)},
    )
    monkeypatch.setattr(
        merge_impl,
        "_legacy_exact_variant_for_method",
        lambda method: [f"legacy-{method}"],
    )
    monkeypatch.setattr(
        merge_impl,
        "_custom_gui_variants",
        lambda: ["custom/sentinel"],
    )

    assert legacy_config._builtin_variants_by_family() == {"lora": [(7, "sentinel")]}
    assert legacy_config._read_variant_metadata(sentinel_path) == {"path": str(sentinel_path)}
    assert legacy_config._legacy_exact_variant_for_method("lora") == ["legacy-lora"]
    assert legacy_config._custom_gui_variants() == ["custom/sentinel"]


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


def test_raw_patch_ignores_dataset_picker_ui_field(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    train_rel = "configs/imported/lora.toml"

    ok, msg, content, changed = config_service.patch_raw_file_values(
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


def test_legacy_common_private_helpers_forward_to_common_module(monkeypatch):
    from web.services.config import common as common_impl

    calls: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}

    def sentinel(name: str):
        def impl(*args, **kwargs):
            calls[name] = (args, kwargs)
            return {"name": name, "args": args, "kwargs": kwargs}

        return impl

    source_path = Path("image_dataset/hero")
    helper_args = {
        "_load": (Path("configs/base.toml"),),
        "_safe_resolve": ("configs/imported/lora.toml",),
        "_safe_config_subdir": ("imported",),
        "_resolve_project_path": ("image_dataset/hero",),
        "_auto_data_dir_for_key": ("", source_path, "resized"),
        "_derived_data_dir": (source_path, "resized"),
        "_is_builtin_default_data_dir": ("post_image_dataset/resized",),
        "_display_path": (source_path,),
        "_positive_int": ("3", 1),
        "_positive_int_or_none": ("3",),
        "_nonnegative_int": ("0", 1),
        "_nonnegative_float": ("0.5", 1.0),
        "_positive_float": ("0.5", 1.0),
        "_bool_value": ("yes", False),
    }
    for name in helper_args:
        monkeypatch.setattr(common_impl, name, sentinel(name))

    for name, args in helper_args.items():
        result = getattr(legacy_config, name)(*args)
        assert result["name"] == name
        assert calls[name] == (args, {})


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


def test_legacy_raw_file_shim_restores_facade_file_group_export(monkeypatch):
    def sentinel_list_config_file_groups(kind=None):
        return [{"id": "sentinel", "kind": kind}]

    monkeypatch.setattr(config_service, "list_config_file_groups", sentinel_list_config_file_groups)

    assert legacy_config.load_raw_file("../outside.toml") == ""
    assert config_service.list_config_file_groups is sentinel_list_config_file_groups


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


def test_config_file_meta_keeps_nested_variant_method_path(tmp_path: Path, monkeypatch):
    configs = tmp_path / "configs"
    (configs / "gui-methods" / "custom").mkdir(parents=True)
    (configs / "imported" / "batch").mkdir(parents=True)
    (configs / "gui-methods" / "custom" / "hero.toml").write_text('output_name = "hero"\n', encoding="utf-8")
    (configs / "imported" / "batch" / "hero.toml").write_text('output_name = "imported_hero"\n', encoding="utf-8")
    _patch_config_service_paths(monkeypatch, tmp_path)

    gui_meta = config_service.get_config_file_meta("configs/gui-methods/custom/hero.toml")
    imported_meta = config_service.get_config_file_meta("configs/imported/batch/hero.toml")

    assert gui_meta["method"] == "custom/hero"
    assert gui_meta["methods_subdir"] == "gui-methods"
    assert imported_meta["method"] == "batch/hero"
    assert imported_meta["methods_subdir"] == "imported"


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


def test_external_configs_root_keeps_stable_config_paths_and_groups(tmp_path: Path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    configs = tmp_path / "external-configs"
    for subdir in ("gui-methods", "imported", "datasets", "methods"):
        (configs / subdir).mkdir(parents=True, exist_ok=True)
    (configs / "base.toml").write_text('pretrained_model_name_or_path = "model.safetensors"\n', encoding="utf-8")
    (configs / "presets.toml").write_text("[default]\ntrain_batch_size = 1\n", encoding="utf-8")
    (configs / "gui-methods" / "lora.toml").write_text(
        '[variant]\nfamily = "lora"\norder = 1\noutput_name = "lora"\n',
        encoding="utf-8",
    )
    (configs / "imported" / "alpha.toml").write_text('output_name = "alpha"\n', encoding="utf-8")
    (configs / "datasets" / "character.toml").write_text(
        "\n".join(
            [
                "[[datasets]]",
                "",
                "[[datasets.subsets]]",
                'image_dir = "post/character"',
                'custom_attributes = { source_dir = "image_dataset/character" }',
                "num_repeats = 2",
            ]
        ),
        encoding="utf-8",
    )
    (configs / "web-file-groups.toml").write_text(
        "\n".join(
            [
                "[[groups]]",
                'id = "gui_methods"',
                'label = "可训练方法变体"',
                "open = true",
                "locked = false",
                "trainable = true",
                'methods_subdir = "gui-methods"',
                'patterns = ["configs/gui-methods/*.toml"]',
                "",
                "[[groups]]",
                'id = "imported"',
                'label = "导入配置"',
                "open = true",
                "locked = false",
                "trainable = true",
                'methods_subdir = "imported"',
                'patterns = ["configs/imported/*.toml"]',
                "",
                "[[groups]]",
                'id = "datasets"',
                'label = "数据集配置"',
                "open = true",
                "locked = false",
                "trainable = false",
                'kind = "dataset"',
                'patterns = ["configs/datasets/*.toml"]',
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(config_service, "ROOT", root)
    monkeypatch.setattr(config_service, "CONFIGS_DIR", configs)
    monkeypatch.setattr(config_service, "DATASET_PRESETS_DIR", configs / "datasets")
    monkeypatch.setattr(config_service, "GUI_METHODS_DIR", configs / "gui-methods")
    monkeypatch.setattr(config_service, "IMPORTED_CONFIGS_DIR", configs / "imported")
    monkeypatch.setattr(config_service, "PRESETS_FILE", configs / "presets.toml")
    monkeypatch.setattr(config_service, "WEB_FILE_GROUPS_FILE", configs / "web-file-groups.toml")
    monkeypatch.setattr(config_service, "WEB_USER_LOCKS_FILE", configs / "web-user-locks.toml")

    training_groups = config_service.list_config_file_groups(kind="training")
    assert any(
        item["path"] == "configs/gui-methods/lora.toml"
        for group in training_groups
        for item in group["files"]
    )
    assert any(
        item["path"] == "configs/imported/alpha.toml"
        for group in training_groups
        for item in group["files"]
    )

    dataset_presets = config_service.list_dataset_presets()
    assert dataset_presets["ok"] is True
    assert [preset["path"] for preset in dataset_presets["presets"]] == ["configs/datasets/character.toml"]
    datasets_group = next(group for group in dataset_presets["groups"] if group["id"] == "datasets")
    assert datasets_group["files"][0]["path"] == "configs/datasets/character.toml"
    assert datasets_group["files"][0]["summary"]["dataset_count"] == 1


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


def test_place_config_file_in_group_uses_exact_drop_index(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    for name in ["alpha", "beta", "gamma"]:
        (configs / "imported" / f"{name}.toml").write_text(f'output_name = "{name}"\n', encoding="utf-8")
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
                "[[groups]]",
                'id = "custom_group"',
                'label = "自定义分组"',
                "open = true",
                "locked = false",
                "trainable = true",
                'methods_subdir = "imported"',
                "user_managed = true",
            ]
        ),
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)

    ok, message, group = config_service.place_config_file_in_group("configs/imported/alpha.toml", "custom_group", 0)
    assert ok is True, message
    assert [item["path"] for item in group["files"]] == ["configs/imported/alpha.toml"]

    ok, message, group = config_service.place_config_file_in_group("configs/imported/beta.toml", "custom_group", 0)
    assert ok is True, message
    assert [item["path"] for item in group["files"]] == [
        "configs/imported/beta.toml",
        "configs/imported/alpha.toml",
    ]

    ok, message, group = config_service.place_config_file_in_group("configs/imported/gamma.toml", "custom_group", 1)
    assert ok is True, message
    assert [item["path"] for item in group["files"]] == [
        "configs/imported/beta.toml",
        "configs/imported/gamma.toml",
        "configs/imported/alpha.toml",
    ]

    ok, message, group = config_service.place_config_file_in_group("configs/imported/alpha.toml", "custom_group", 0)
    assert ok is True, message
    assert [item["path"] for item in group["files"]] == [
        "configs/imported/alpha.toml",
        "configs/imported/beta.toml",
        "configs/imported/gamma.toml",
    ]

    response = asyncio.run(config_routes.handle_file_group_place(_JsonRequest({
        "target": "file",
        "file": "configs/imported/gamma.toml",
        "group": "custom_group",
        "index": 1,
    })))
    assert response.status == 200
    body = json.loads(response.text)
    assert [item["path"] for item in body["group"]["files"]] == [
        "configs/imported/alpha.toml",
        "configs/imported/gamma.toml",
        "configs/imported/beta.toml",
    ]


def test_place_config_file_rejects_cross_kind_and_locked_targets(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    (configs / "datasets" / "character.toml").write_text(
        "[[datasets]]\n[[datasets.subsets]]\nimage_dir = \"image_dataset/a\"\n",
        encoding="utf-8",
    )
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
                "[[groups]]",
                'id = "locked_imported"',
                'label = "锁定导入"',
                "open = true",
                "locked = true",
                "trainable = true",
                'methods_subdir = "imported"',
                "user_managed = true",
                "",
                "[[groups]]",
                'id = "datasets"',
                'label = "数据集配置"',
                "open = true",
                "locked = false",
                "trainable = false",
                'patterns = ["configs/datasets/*.toml"]',
            ]
        ),
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)

    ok, message, _group = config_service.place_config_file_in_group(
        "configs/datasets/character.toml",
        "imported",
        0,
    )
    assert ok is False
    assert "数据集预设只能移动到数据集分组" in message

    ok, message, _group = config_service.place_config_file_in_group(
        "configs/imported/lora.toml",
        "locked_imported",
        0,
    )
    assert ok is False
    assert "目标分组已锁定" in message


def test_place_config_file_group_sorts_within_scope_only(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
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
                "[[groups]]",
                'id = "alpha"',
                'label = "Alpha"',
                "open = true",
                "locked = false",
                "trainable = true",
                'methods_subdir = "imported"',
                "user_managed = true",
                "",
                "[[groups]]",
                'id = "beta"',
                'label = "Beta"',
                "open = true",
                "locked = false",
                "trainable = true",
                'methods_subdir = "imported"',
                "user_managed = true",
                "",
                "[[groups]]",
                'id = "datasets"',
                'label = "数据集配置"',
                "open = true",
                "locked = false",
                "trainable = false",
                'patterns = ["configs/datasets/*.toml"]',
                "",
                "[[groups]]",
                'id = "dataset_extra"',
                'label = "额外数据集"',
                "open = false",
                "locked = false",
                "trainable = false",
                'kind = "dataset"',
                "user_managed = true",
                "",
                "[[groups]]",
                'id = "locked_custom"',
                'label = "锁定分组"',
                "open = true",
                "locked = true",
                "trainable = true",
                'methods_subdir = "imported"',
                "user_managed = true",
                "",
                "[[groups]]",
                'id = "gui_methods"',
                'label = "可训练方法变体"',
                "open = true",
                "locked = false",
                "trainable = true",
                'methods_subdir = "gui-methods"',
            ]
        ),
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)

    ok, message, group = config_service.place_config_file_group("beta", "training", 0)
    assert ok is True, message
    assert group["id"] == "beta"
    assert [item["id"] for item in config_service.list_config_file_groups(kind="training")[:3]] == [
        "beta",
        "imported",
        "alpha",
    ]
    assert [item["id"] for item in config_service.list_config_file_groups(kind="dataset")] == [
        "datasets",
        "dataset_extra",
    ]

    ok, message, group = config_service.place_config_file_group("dataset_extra", "dataset", 0)
    assert ok is True, message
    assert group["id"] == "dataset_extra"
    assert [item["id"] for item in config_service.list_config_file_groups(kind="dataset")] == [
        "dataset_extra",
        "datasets",
    ]

    ok, message, _group = config_service.place_config_file_group("locked_custom", "training", 0)
    assert ok is False
    assert "不能在当前范围内拖动排序" in message

    ok, message, _group = config_service.place_config_file_group("gui_methods", "training", 0)
    assert ok is False
    assert "不能在当前范围内拖动排序" in message


def test_export_config_file_group_archive_contains_independent_toml_files(tmp_path: Path, monkeypatch):
    configs = tmp_path / "configs"
    imported = configs / "imported"
    imported.mkdir(parents=True)
    (imported / "alpha.toml").write_text('output_name = "alpha"\n', encoding="utf-8")
    (imported / "beta.toml").write_text('output_name = "beta"\n', encoding="utf-8")
    (configs / "web-file-groups.toml").write_text(
        "\n".join(
            [
                "[[groups]]",
                'id = "custom_group"',
                'label = "我的配置/分组"',
                "open = true",
                "trainable = true",
                'files = ["configs/imported/alpha.toml", "configs/imported/beta.toml"]',
            ]
        ),
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)

    archive = config_service.export_config_file_group_archive("custom_group")

    assert archive["filename"] == "我的配置_分组.zip"
    assert archive["count"] == 2
    with zipfile.ZipFile(io.BytesIO(archive["content"])) as zf:
        assert sorted(zf.namelist()) == ["alpha.toml", "beta.toml"]
        assert zf.read("alpha.toml").decode("utf-8") == 'output_name = "alpha"\n'
        assert zf.read("beta.toml").decode("utf-8") == 'output_name = "beta"\n'


def test_handle_file_group_export_returns_zip_response(tmp_path: Path, monkeypatch):
    configs = tmp_path / "configs"
    imported = configs / "imported"
    imported.mkdir(parents=True)
    (imported / "alpha.toml").write_text('output_name = "alpha"\n', encoding="utf-8")
    (configs / "web-file-groups.toml").write_text(
        "\n".join(
            [
                "[[groups]]",
                'id = "custom_group"',
                'label = "导出测试"',
                "open = true",
                "trainable = true",
                'files = ["configs/imported/alpha.toml"]',
            ]
        ),
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)

    class _MatchQueryRequest(_QueryRequest):
        def __init__(self, group_id: str, query: dict[str, str] | None = None) -> None:
            super().__init__(query)
            self.match_info = {"group_id": group_id}

    response = asyncio.run(config_routes.handle_file_group_export(_MatchQueryRequest("custom_group")))

    assert response.status == 200
    assert response.content_type == "application/zip"
    assert "filename*=UTF-8''" in response.headers["Content-Disposition"]
    with zipfile.ZipFile(io.BytesIO(response.body)) as zf:
        assert zf.namelist() == ["alpha.toml"]


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


def test_legacy_dataset_shim_restores_facade_file_group_export(monkeypatch):
    def sentinel_list_config_file_groups(kind=None):
        return [{"id": "sentinel", "kind": kind}]

    monkeypatch.setattr(config_service, "list_config_file_groups", sentinel_list_config_file_groups)

    summary = legacy_config._dataset_summary_from_rows(
        [{"source_dir": "image_dataset/a", "num_repeats": 2}],
        {"resolution": 512, "batch_size": 1},
    )

    assert summary["dataset_count"] == 1
    assert config_service.list_config_file_groups is sentinel_list_config_file_groups


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


def test_legacy_sample_prompts_exports_forward_to_split_module():
    from web.services.config import sample_prompts as sample_prompts_impl

    assert tuple(legacy_config._SAMPLE_PROMPTS_SHIM_NAMES) == tuple(sample_prompts_impl.__all__)
    for name in sample_prompts_impl.__all__:
        exported = getattr(legacy_config, name)
        assert exported is legacy_config._SAMPLE_PROMPTS_SHIMS[name]
        assert (
            exported.__doc__
            == f"Compatibility shim forwarding to web.services.config.sample_prompts.{name}."
        )


def test_legacy_file_group_exports_forward_to_split_file_group_module():
    from web.services.config import file_groups as file_group_impl

    missing = []
    not_forwarded = []
    for name in file_group_impl.__all__:
        exported = getattr(legacy_config, name, None)
        if exported is None:
            missing.append(name)
            continue
        doc = str(getattr(exported, "__doc__", "") or "")
        if "web.services.config.file_groups" not in doc:
            not_forwarded.append(name)

    assert missing == []
    assert not_forwarded == []


def test_legacy_file_group_private_helpers_forward_to_split_module(monkeypatch):
    from web.services.config import file_groups as file_group_impl

    monkeypatch.setattr(file_group_impl, "_get_config_file_group", lambda group_id: {"id": group_id})
    monkeypatch.setattr(file_group_impl, "_config_group_kind", lambda raw: f"kind:{raw['id']}")
    monkeypatch.setattr(file_group_impl, "_normalize_group_id", lambda group_id: f"norm:{group_id}")
    monkeypatch.setattr(file_group_impl, "_safe_archive_name", lambda name: f"safe:{name}")
    monkeypatch.setattr(
        file_group_impl,
        "_unique_archive_member_name",
        lambda name, used_names: used_names.add(f"unique:{name}") or f"unique:{name}",
    )

    used_names: set[str] = set()
    assert legacy_config._get_config_file_group("abc") == {"id": "abc"}
    assert legacy_config._config_group_kind({"id": "datasets"}) == "kind:datasets"
    assert legacy_config._normalize_group_id(" group ") == "norm: group "
    assert legacy_config._safe_archive_name("bad/name") == "safe:bad/name"
    assert legacy_config._unique_archive_member_name("member", used_names) == "unique:member"
    assert used_names == {"unique:member"}


def test_legacy_file_group_group_model_helpers_forward_to_split_module(monkeypatch):
    from web.services.config import file_groups as file_group_impl

    def sentinel(name: str):
        def impl(*args, **kwargs):
            return {"name": name, "args": args, "kwargs": kwargs}

        return impl

    helper_args = {
        "_config_method_name_for_path": ("configs/imported/lora.toml",),
        "_infer_config_file_group": ("configs/imported/lora.toml",),
        "_strip_configs_prefix": ("configs/imported/lora.toml",),
        "_sort_config_file_group_specs_for_display": ([{"id": "a"}],),
        "_build_config_file_group": ({"id": "a"},),
        "_glob_config_files": ("configs/imported/*.toml",),
        "_default_config_file_group_specs": (),
        "_group_defaults": ("a", "A", False, True, "imported", True),
        "_find_config_group_spec": ([{"id": "a"}], "a"),
        "_new_user_config_group_spec": ("a", "A", "training"),
        "_move_orphaned_config_files_to_fallback_groups": ([], ["configs/imported/lora.toml"]),
        "_config_file_is_covered_by_specs": ([], "configs/imported/lora.toml"),
        "_fallback_config_group_spec": ("configs/imported/lora.toml",),
        "_is_user_managed_group": ({"id": "a", "user_managed": True},),
        "_is_fixed_config_group": ({"id": "a"},),
        "_is_deletable_config_group": ({"id": "a"},),
        "_is_renamable_config_group": ({"id": "a"},),
        "_is_move_target_group": ({"id": "a"}, "configs/imported/lora.toml"),
        "_is_sortable_config_group_for_place": ({"id": "a"}, "training"),
        "_place_index": (2, 5),
        "_lockable_group_ids": (),
    }
    for name in helper_args:
        monkeypatch.setattr(file_group_impl, name, sentinel(name))

    for name, args in helper_args.items():
        result = getattr(legacy_config, name)(*args)
        assert result["name"] == name


def test_legacy_file_group_leaf_helpers_forward_to_split_module(monkeypatch):
    from web.services.config import file_groups as file_group_impl

    def sentinel(name: str):
        def impl(*args, **kwargs):
            return {"name": name, "args": args, "kwargs": kwargs}

        return impl

    helper_args = {
        "_unique_group_id": ("custom", [{"id": "custom"}]),
        "_slugify_group_label": ("My Group",),
        "_normalize_group_label": ("  My   Group  ",),
        "_group_patterns_include_file": ({"patterns": ["configs/imported/*.toml"]}, "configs/imported/a.toml"),
        "_is_system_preset_path": ("configs/base.toml",),
        "_is_system_locked_path": ("configs/base.toml",),
        "_list_system_preset_files": (),
        "_read_git_head_file": ("configs/base.toml",),
        "_backup_relative_path": ("configs/imported/a.toml",),
        "_string_list": (["a", "b"],),
        "_config_group_path_list": (["configs/imported/a.toml"],),
    }
    for name in helper_args:
        monkeypatch.setattr(file_group_impl, name, sentinel(name))

    for name, args in helper_args.items():
        result = getattr(legacy_config, name)(*args)
        assert result["name"] == name


def test_legacy_file_group_helpers_use_split_file_group_module(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(legacy_config, "ROOT", tmp_path)
    monkeypatch.setattr(legacy_config, "CONFIGS_DIR", configs)
    monkeypatch.setattr(legacy_config, "DATASET_PRESETS_DIR", configs / "datasets")
    monkeypatch.setattr(legacy_config, "GUI_METHODS_DIR", configs / "gui-methods")
    monkeypatch.setattr(legacy_config, "IMPORTED_CONFIGS_DIR", configs / "imported")
    monkeypatch.setattr(legacy_config, "PRESETS_FILE", configs / "presets.toml")
    monkeypatch.setattr(legacy_config, "WEB_FILE_GROUPS_FILE", configs / "web-file-groups.toml")
    monkeypatch.setattr(legacy_config, "WEB_USER_LOCKS_FILE", configs / "web-user-locks.toml")

    groups = legacy_config.list_config_file_groups(kind="dataset")
    meta = legacy_config.get_config_file_meta("configs/datasets/lora.toml")

    assert [group["id"] for group in groups] == ["datasets"]
    assert meta["path"] == "configs/datasets/lora.toml"
    assert meta["group"] == "datasets"
    assert meta["group_label"] == "数据集配置"


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


def test_handle_merged_uses_selected_training_config_file(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    (configs / "imported" / "selected.toml").write_text(
        "\n".join(
            [
                'output_name = "selected_form"',
                "max_train_epochs = 9",
                "max_train_steps = 0",
                "train_batch_size = 3",
                "gradient_accumulation_steps = 2",
                "sample_ratio = 0.5",
            ]
        ),
        encoding="utf-8",
    )

    response = asyncio.run(config_routes.handle_merged(_QueryRequest({
        "variant": "lora",
        "preset": "default",
        "methods_subdir": "imported",
        "config_file": "configs/imported/selected.toml",
    })))
    body = json.loads(response.text)

    assert response.status == 200
    assert body["output_name"] == "selected_form"
    assert body["max_train_epochs"] == 9
    assert body["max_train_steps"] == 0
    assert body["train_batch_size"] == 3
    assert body["gradient_accumulation_steps"] == 2
    assert body["sample_ratio"] == 0.5


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


def test_output_run_helpers_remain_available_from_legacy_module(tmp_path: Path, monkeypatch):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    output_root = tmp_path / "output" / "runs"
    _patch_output_root(monkeypatch, output_root)
    monkeypatch.setattr(legacy_config, "ROOT", tmp_path)
    monkeypatch.setattr(legacy_config, "CONFIGS_DIR", tmp_path / "configs")
    monkeypatch.setattr(legacy_config, "DATASET_PRESETS_DIR", tmp_path / "configs" / "datasets")
    monkeypatch.setattr(legacy_config, "GUI_METHODS_DIR", tmp_path / "configs" / "gui-methods")
    monkeypatch.setattr(legacy_config, "IMPORTED_CONFIGS_DIR", tmp_path / "configs" / "imported")
    monkeypatch.setattr(legacy_config, "PRESETS_FILE", tmp_path / "configs" / "presets.toml")
    monkeypatch.setattr(legacy_config, "WEB_FILE_GROUPS_FILE", tmp_path / "configs" / "web-file-groups.toml")
    monkeypatch.setattr(legacy_config, "WEB_USER_LOCKS_FILE", tmp_path / "configs" / "web-user-locks.toml")
    monkeypatch.setattr(legacy_config, "resolve_output_root", lambda: output_root.resolve())
    monkeypatch.setattr(
        legacy_config,
        "_display_settings_path",
        lambda path: _display_test_path(Path(path), output_root.parents[1]),
    )
    run = output_root / "legacy-20260523-114514"
    run.mkdir(parents=True)
    (run / "config.original.toml").write_text('output_name = "legacy_original"\n', encoding="utf-8")
    (run / "config.runtime.toml").write_text('output_name = "legacy_runtime"\n', encoding="utf-8")

    expected_shims = (
        "list_output_runs",
        "load_output_run_config",
        "save_output_run_config_as",
        "_resolve_output_run_dir",
        "_normalize_output_run_name",
    )
    assert tuple(legacy_config._OUTPUT_RUNS_SHIM_NAMES) == expected_shims
    for name in expected_shims:
        assert getattr(legacy_config, name) is legacy_config._OUTPUT_RUNS_SHIMS[name]
        assert (
            getattr(legacy_config, name).__doc__
            == f"Compatibility shim forwarding to web.services.config.output_runs.{name}."
        )
    legacy_config._restore_raw_files_shims()
    raw_file_shims = legacy_config._RAW_FILES_SHIMS
    assert legacy_config.load_raw_file is raw_file_shims["load_raw_file"]

    assert config_service.list_output_runs()["ok"] is True
    for name, shim in raw_file_shims.items():
        assert getattr(legacy_config, name) is shim

    listed = legacy_config.list_output_runs()
    assert listed["ok"] is True
    assert listed["output_root"] == "output/runs"
    assert [item["name"] for item in listed["runs"]] == ["legacy-20260523-114514"]

    original = legacy_config.load_output_run_config("legacy-20260523-114514", "original")
    assert original["content"] == 'output_name = "legacy_original"\n'
    assert legacy_config._normalize_output_run_name("legacy-20260523-114514") == "legacy-20260523-114514"
    assert legacy_config._resolve_output_run_dir("legacy-20260523-114514") == run.resolve()

    saved = legacy_config.save_output_run_config_as(
        "legacy-20260523-114514",
        "copied_from_legacy",
        "imported",
    )
    assert saved["ok"] is True
    assert saved["file"] == "configs/imported/copied_from_legacy.toml"
    copied_path = tmp_path / "configs" / "imported" / "copied_from_legacy.toml"
    assert copied_path.read_text(encoding="utf-8") == 'output_name = "legacy_original"\n'
    with pytest.raises(ValueError, match="已存在"):
        legacy_config.save_output_run_config_as("legacy-20260523-114514", "copied_from_legacy", "imported")
    for name in expected_shims:
        assert getattr(legacy_config, name) is legacy_config._OUTPUT_RUNS_SHIMS[name]


def test_legacy_output_run_private_helpers_forward_to_split_module(monkeypatch):
    from web.services.config import output_runs as output_runs_impl

    run_dir = Path("output/runs/sentinel")
    config_path = run_dir / "config.original.toml"
    monkeypatch.setattr(output_runs_impl, "_output_run_summary", lambda path: {"path": str(path)})
    monkeypatch.setattr(output_runs_impl, "_output_run_config_path", lambda path, kind: path / f"config.{kind}.toml")
    monkeypatch.setattr(
        output_runs_impl,
        "_normalize_output_run_save_as_path",
        lambda value, *, fallback_stem: f"configs/imported/{value or fallback_stem}.toml",
    )
    monkeypatch.setattr(output_runs_impl, "_safe_mtime", lambda path: 123.0)
    monkeypatch.setattr(output_runs_impl, "_format_file_time", lambda value: f"formatted:{value}")

    assert legacy_config._output_run_summary(run_dir) == {"path": str(run_dir)}
    assert legacy_config._output_run_config_path(run_dir, "original") == config_path
    assert (
        legacy_config._normalize_output_run_save_as_path("", fallback_stem="fallback")
        == "configs/imported/fallback.toml"
    )
    assert legacy_config._safe_mtime(config_path) == 123.0
    assert legacy_config._format_file_time(123.0) == "formatted:123.0"


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


def _patch_external_config_service_paths(monkeypatch, root: Path, configs: Path) -> None:
    project_configs = root / "configs"
    if project_configs.exists():
        project_configs.rename(configs)
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


class _QueryRequest:
    def __init__(self, query: dict[str, str] | None = None) -> None:
        self.query = query or {}
