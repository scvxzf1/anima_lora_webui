"""Shared runtime for config file-group helpers.

Owns path roots, owner lookup, and facade monkeypatch sync so domain modules
can stay thin without importing the public facade at module import time.
"""

from __future__ import annotations

from functools import wraps
from pathlib import Path

from library.env import get_configs_root, load_dotenv
from web.services.config import common as _config_common

ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = get_configs_root()
GUI_METHODS_DIR = CONFIGS_DIR / "gui-methods"
IMPORTED_CONFIGS_DIR = CONFIGS_DIR / "imported"
PRESETS_FILE = CONFIGS_DIR / "presets.toml"
WEB_FILE_GROUPS_FILE = CONFIGS_DIR / "web-file-groups.toml"
WEB_USER_LOCKS_FILE = CONFIGS_DIR / "web-user-locks.toml"
DATASET_PRESETS_DIR = CONFIGS_DIR / "datasets"

load_dotenv()

_SYNC_NAMES = (
    "ROOT",
    "CONFIGS_DIR",
    "WEB_FILE_GROUPS_FILE",
    "WEB_USER_LOCKS_FILE",
    "save_raw_file",
    "load_raw_file",
    "delete_raw_file",
    "patch_raw_file_values",
    "preview_raw_file_patch",
)


def _owner():
    """Prefer the public file_groups module so tests can monkeypatch paths there."""
    from web.services.config import file_groups as owner

    return owner


def _owner_attr(name: str, default=None):
    owner = _owner()
    return getattr(owner, name, default)


def _sync_from_facade() -> None:
    from web.services import config_service as _facade

    owner = _owner()
    _exported_names = set(getattr(owner, "__all__", ()) or ())
    # Path constants must always follow the public facade, even when tests only
    # monkeypatch config_service.ROOT / CONFIGS_DIR for external configs roots.
    _force_sync_names = {
        "ROOT",
        "CONFIGS_DIR",
        "WEB_FILE_GROUPS_FILE",
        "WEB_USER_LOCKS_FILE",
        "GUI_METHODS_DIR",
        "IMPORTED_CONFIGS_DIR",
        "PRESETS_FILE",
        "DATASET_PRESETS_DIR",
    }
    for _name in (*_SYNC_NAMES, "GUI_METHODS_DIR", "IMPORTED_CONFIGS_DIR", "PRESETS_FILE", "DATASET_PRESETS_DIR"):
        if not hasattr(_facade, _name):
            continue
        _value = getattr(_facade, _name)
        if _name in _force_sync_names or _name not in _exported_names:
            setattr(owner, _name, _value)
            globals()[_name] = _value
            # Keep the compatibility re-export module in sync for direct imports/tests.
            import sys

            core_mod = sys.modules.get("web.services.config.file_group_core")
            if core_mod is not None:
                setattr(core_mod, _name, _value)
    _sync_common_paths()


def _exported(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        _sync_from_facade()
        return fn(*args, **kwargs)

    return wrapper


def _sync_common_paths() -> None:
    owner = _owner()
    _config_common.ROOT = owner.ROOT
    _config_common.CONFIGS_DIR = owner.CONFIGS_DIR


def _load(p: Path) -> dict:
    _sync_common_paths()
    return _config_common._load(p)


def _safe_resolve(rel_path: str) -> Path | None:
    _sync_common_paths()
    return _config_common._safe_resolve(rel_path)


def _display_path(path: Path) -> str:
    _sync_common_paths()
    return _config_common._display_path(path)
