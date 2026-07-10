"""Configuration loading, merging, and saving."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from library.env import load_dotenv, get_configs_root
# Re-export config metadata constants/helpers for the historical facade surface.
from web.services.config.metadata import *  # noqa: F403
from web.services.settings_service import display_path as _display_settings_path  # noqa: F401 - synced into split modules
from web.services.settings_service import resolve_output_root  # noqa: F401 - synced into split modules

ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = get_configs_root()
GUI_METHODS_DIR = CONFIGS_DIR / "gui-methods"
IMPORTED_CONFIGS_DIR = CONFIGS_DIR / "imported"
PRESETS_FILE = CONFIGS_DIR / "presets.toml"
WEB_FILE_GROUPS_FILE = CONFIGS_DIR / "web-file-groups.toml"
WEB_USER_LOCKS_FILE = CONFIGS_DIR / "web-user-locks.toml"
DEFAULT_SAMPLE_PROMPTS_FILE = str(CONFIGS_DIR / "sample_prompts.txt")
DEFAULT_MAX_TRAIN_STEPS = 0
DATASET_PRESETS_DIR = CONFIGS_DIR / "datasets"
_DELETE_TOML_KEY = object()

load_dotenv()

LOGGER = logging.getLogger(__name__)

from web.services.config.legacy_names import (
    _COMMON_SYNC_NAMES,
    _COMMON_SHIM_NAMES,
    _PREFLIGHT_SHIM_SYNC_NAMES,
    _PREFLIGHT_SHIM_NAMES,
    _PREFLIGHT_PRIVATE_HELPER_NAMES,
    _MERGE_SHIM_SYNC_NAMES,
    _MERGE_SHIM_NAMES,
    _MERGE_PRIVATE_HELPER_NAMES,
    _OUTPUT_RUNS_SHIM_SYNC_NAMES,
    _OUTPUT_RUNS_LEGACY_HELPER_NAMES,
    _OUTPUT_RUNS_SHIM_NAMES,
    _ESTIMATION_SHIM_SYNC_NAMES,
    _ESTIMATION_SHIM_NAMES,
    _DATASET_SHIM_SYNC_NAMES,
    _DATASET_SHIM_NAMES,
    _SAMPLE_PROMPTS_SHIM_SYNC_NAMES,
    _SAMPLE_PROMPTS_SHIM_NAMES,
    _FILE_GROUPS_SHIM_SYNC_NAMES,
    _FILE_GROUPS_SHIM_NAMES,
    _RAW_FILES_SHIM_SYNC_NAMES,
    _RAW_FILES_LEGACY_HELPER_NAMES,
    _RAW_FILES_FACADE_HELPER_NAMES,
    _RAW_FILES_SHIM_NAMES,
)


_SHIM_BUCKETS: dict[str, dict[str, Any]] = {
    "common": {},
    "preflight": {},
    "merge": {},
    "output_runs": {},
    "estimation": {},
    "dataset": {},
    "sample_prompts": {},
    "file_groups": {},
    "raw_files": {},
}

# Keep historical bucket names as aliases for monkeypatch / installer restore paths.
_COMMON_SHIMS = _SHIM_BUCKETS["common"]
_PREFLIGHT_SHIMS = _SHIM_BUCKETS["preflight"]
_MERGE_SHIMS = _SHIM_BUCKETS["merge"]
_OUTPUT_RUNS_SHIMS = _SHIM_BUCKETS["output_runs"]
_ESTIMATION_SHIMS = _SHIM_BUCKETS["estimation"]
_DATASET_SHIMS = _SHIM_BUCKETS["dataset"]
_SAMPLE_PROMPTS_SHIMS = _SHIM_BUCKETS["sample_prompts"]
_FILE_GROUPS_SHIMS = _SHIM_BUCKETS["file_groups"]
_RAW_FILES_SHIMS = _SHIM_BUCKETS["raw_files"]


def _inspect_network_weight(
    path: str,
    *,
    variant: str,
    preset: str,
    methods_subdir: str,
    config_file: str | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _call_preflight_impl(
        "_inspect_network_weight_impl",
        path,
        variant=variant,
        preset=preset,
        methods_subdir=methods_subdir,
        config_file=config_file,
        cfg=cfg,
    )


def _install_shim_bucket(bucket: str, installer_name: str) -> None:
    from web.services.config import legacy_shims as _legacy_shims

    installer = getattr(_legacy_shims, installer_name)
    installed = installer(globals())
    target = _SHIM_BUCKETS[bucket]
    target.clear()
    target.update(installed)


def _install_common_public_shims() -> None:
    _install_shim_bucket("common", "install_common_public_shims")


def _install_preflight_public_shims() -> None:
    _install_shim_bucket("preflight", "install_preflight_public_shims")


def _install_merge_public_shims() -> None:
    _install_shim_bucket("merge", "install_merge_public_shims")


def _install_output_runs_public_shims() -> None:
    _install_shim_bucket("output_runs", "install_output_runs_public_shims")


def _install_estimation_public_shims() -> None:
    _install_shim_bucket("estimation", "install_estimation_public_shims")


def _install_dataset_public_shims() -> None:
    _install_shim_bucket("dataset", "install_dataset_public_shims")


def _restore_raw_files_shims() -> None:
    from web.services.config.legacy_shims import restore_raw_files_shims

    restore_raw_files_shims(globals())


def _install_sample_prompts_public_shims() -> None:
    _install_shim_bucket("sample_prompts", "install_sample_prompts_public_shims")


def _install_file_groups_public_shims() -> None:
    _install_shim_bucket("file_groups", "install_file_groups_public_shims")


def _install_raw_files_public_shims() -> None:
    _install_shim_bucket("raw_files", "install_raw_files_public_shims")


_SHIM_INSTALL_ORDER = (
    _install_common_public_shims,
    _install_preflight_public_shims,
    _install_merge_public_shims,
    _install_output_runs_public_shims,
    _install_estimation_public_shims,
    _install_dataset_public_shims,
    _install_sample_prompts_public_shims,
    _install_file_groups_public_shims,
    _install_raw_files_public_shims,
)

for _install in _SHIM_INSTALL_ORDER:
    _install()
