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


def test_file_groups_direct_path_helpers_support_external_configs_root_without_facade_snapshot(
    tmp_path: Path,
):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    script = f"""
import sys
from pathlib import Path

import web.services.config.file_groups as file_groups

root = Path({str(tmp_path / "project")!r})
configs = Path({str(tmp_path / "external-configs")!r})
(configs / "imported").mkdir(parents=True)
(configs / "imported" / "external.toml").write_text('output_name = "external"\\n', encoding="utf-8")
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

external = configs / "imported" / "external.toml"
assert file_groups._load(external) == dict(output_name="external")
assert file_groups._safe_resolve("configs/imported/external.toml") == external.resolve()
assert file_groups._safe_resolve("../outside.toml") is None
assert file_groups._display_path(external) == "configs/imported/external.toml"
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


def test_file_groups_glob_uses_synced_common_paths_under_external_configs_root(
    tmp_path: Path,
):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    script = f"""
import sys
from pathlib import Path

from web.services import config_service
import web.services.config.file_groups as file_groups

root = Path({str(tmp_path / "project")!r})
configs = Path({str(tmp_path / "external-configs")!r})
(configs / "imported").mkdir(parents=True)
(configs / "imported" / "external.toml").write_text('output_name = "external"\\n', encoding="utf-8")
(configs / "imported" / "skip.txt").write_text("nope\\n", encoding="utf-8")
(configs / "gui-methods").mkdir()
(configs / "datasets").mkdir()

for module in (config_service, file_groups):
    module.ROOT = root
    module.CONFIGS_DIR = configs
    module.GUI_METHODS_DIR = configs / "gui-methods"
    module.IMPORTED_CONFIGS_DIR = configs / "imported"
    module.PRESETS_FILE = configs / "presets.toml"
    module.WEB_FILE_GROUPS_FILE = configs / "web-file-groups.toml"
    module.WEB_USER_LOCKS_FILE = configs / "web-user-locks.toml"
    module.DATASET_PRESETS_DIR = configs / "datasets"

assert file_groups._glob_config_files("configs/imported/*.toml") == [
    "configs/imported/external.toml"
]
assert file_groups._glob_config_files("../*.toml") == []
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


def test_file_group_common_path_helpers_forward_to_common_module(monkeypatch):
    from web.services.config import common as common_impl
    from web.services.config import file_groups as file_group_impl

    root = Path("/tmp/anima-root")
    configs_dir = Path("/tmp/anima-configs")
    monkeypatch.setattr(file_group_impl, "ROOT", root)
    monkeypatch.setattr(file_group_impl, "CONFIGS_DIR", configs_dir)

    def assert_synced() -> None:
        assert common_impl.ROOT == root
        assert common_impl.CONFIGS_DIR == configs_dir

    def fake_load(path: Path) -> dict[str, Any]:
        assert_synced()
        return {"loaded": path.as_posix()}

    def fake_safe_resolve(rel_path: str) -> Path:
        assert_synced()
        return configs_dir / rel_path

    def fake_display_path(path: Path) -> str:
        assert_synced()
        return f"display:{path.as_posix()}"

    monkeypatch.setattr(common_impl, "_load", fake_load)
    monkeypatch.setattr(common_impl, "_safe_resolve", fake_safe_resolve)
    monkeypatch.setattr(common_impl, "_display_path", fake_display_path)

    assert file_group_impl._load(Path("configs/base.toml")) == {
        "loaded": "configs/base.toml"
    }
    assert file_group_impl._safe_resolve("configs/base.toml") == (
        configs_dir / "configs/base.toml"
    )
    assert file_group_impl._display_path(root / "configs/base.toml") == (
        "display:/tmp/anima-root/configs/base.toml"
    )


def test_legacy_raw_file_shim_restores_facade_file_group_export(monkeypatch):
    def sentinel_list_config_file_groups(kind=None):
        return [{"id": "sentinel", "kind": kind}]

    monkeypatch.setattr(config_service, "list_config_file_groups", sentinel_list_config_file_groups)

    assert legacy_config.load_raw_file("../outside.toml") == ""
    assert config_service.list_config_file_groups is sentinel_list_config_file_groups


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

