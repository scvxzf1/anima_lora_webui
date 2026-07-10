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

def test_config_service_common_private_wrappers_sync_facade_state_to_legacy(
    monkeypatch,
):
    root = Path("/tmp/facade-root")
    configs = root / "configs"
    monkeypatch.setattr(config_service, "ROOT", root)
    monkeypatch.setattr(config_service, "CONFIGS_DIR", configs)

    def assert_synced() -> None:
        assert legacy_config.ROOT == root
        assert legacy_config.CONFIGS_DIR == configs

    def fake_load(path: Path) -> dict[str, str]:
        assert_synced()
        return {"path": path.as_posix()}

    def fake_safe_config_subdir(subdir: str) -> Path:
        assert_synced()
        return configs / subdir

    def fake_display_path(path: Path) -> str:
        assert_synced()
        return f"display:{path.as_posix()}"

    monkeypatch.setattr(legacy_config, "_load", fake_load)
    monkeypatch.setattr(legacy_config, "_safe_config_subdir", fake_safe_config_subdir)
    monkeypatch.setattr(legacy_config, "_display_path", fake_display_path)

    assert config_service._load(Path("configs/base.toml")) == {
        "path": "configs/base.toml"
    }
    assert config_service._safe_config_subdir("imported") == configs / "imported"
    assert config_service._display_path(root / "configs/base.toml") == (
        "display:/tmp/facade-root/configs/base.toml"
    )


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

