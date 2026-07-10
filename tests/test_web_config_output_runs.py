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

def test_output_runs_direct_private_helpers_work_without_facade_cycle():
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    script = """
import sys
import tempfile
from pathlib import Path

import web.services.config.output_runs as output_runs

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    output_runs.ROOT = root
    output_runs.CONFIGS_DIR = root / "configs"
    assert output_runs._normalize_output_run_name.__wrapped__("run-1") == "run-1"
    assert (
        output_runs._output_run_config_path(
            root / "output" / "runs" / "run-1", "original"
        ).name
        == "config.original.toml"
    )
    assert (
        output_runs._normalize_output_run_save_as_path("copy", fallback_stem="run-1")
        == "configs/imported/copy.toml"
    )
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

    assert [item["name"] for item in config_service.list_output_runs(limit=1)["runs"]] == [
        "newer-20260523-120000",
    ]
    assert [item["name"] for item in config_service.list_output_runs(limit=0)["runs"]] == [
        "newer-20260523-120000",
        "older-20260523-110000",
    ]

def test_output_runs_list_handles_missing_or_file_output_root(
    tmp_path: Path,
    monkeypatch,
):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    missing_root = tmp_path / "output" / "runs"
    _patch_output_root(monkeypatch, missing_root)

    result = config_service.list_output_runs()

    assert result == {
        "ok": True,
        "output_root": "output/runs",
        "output_root_abs": str(missing_root.resolve()),
        "runs": [],
    }

    file_root = tmp_path / "output" / "file-root"
    file_root.parent.mkdir(parents=True)
    file_root.write_text("not a directory\n", encoding="utf-8")
    _patch_output_root(monkeypatch, file_root)

    with pytest.raises(ValueError, match="输出文件夹不是目录"):
        config_service.list_output_runs()

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

def test_output_run_read_reports_missing_fixed_config_file(
    tmp_path: Path,
    monkeypatch,
):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    _patch_output_root(monkeypatch, tmp_path / "output" / "runs")
    run = tmp_path / "output" / "runs" / "522-20260523-114514"
    run.mkdir(parents=True)
    (run / "config.original.toml").write_text('output_name = "original"\n', encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="运行配置不存在"):
        config_service.load_output_run_config("522-20260523-114514", "runtime")
    with pytest.raises(FileNotFoundError, match="运行配置不存在"):
        config_service.load_output_run_config("522-20260523-114514", "dataset")

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

def test_output_run_save_as_rejects_paths_outside_imported_configs(tmp_path: Path, monkeypatch):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    _patch_output_root(monkeypatch, tmp_path / "output" / "runs")
    run = tmp_path / "output" / "runs" / "522-20260523-114514"
    run.mkdir(parents=True)
    (run / "config.original.toml").write_text('output_name = "original"\n', encoding="utf-8")

    unsafe_names = [
        "../escape",
        "configs/imported/../escape",
        "configs/other/escape",
        (tmp_path / "configs" / "other" / "escape.toml").as_posix(),
        (tmp_path.parent / "escape.toml").as_posix(),
    ]

    for unsafe_name in unsafe_names:
        with pytest.raises(ValueError, match="新项目预设"):
            config_service.save_output_run_config_as(
                "522-20260523-114514",
                unsafe_name,
                "imported",
            )

    assert not (tmp_path / "configs" / "imported" / "escape.toml").exists()
    assert not (tmp_path / "configs" / "other" / "escape.toml").exists()
    assert not (tmp_path.parent / "escape.toml").exists()

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

