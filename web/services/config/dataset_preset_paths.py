"""Dataset preset path helpers extracted from datasets service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from library.env import expand_env_vars, get_configs_root
from web.services.config import paths as _config_paths
from web.services.config.metadata import SYSTEM_DATASET_PRESET_FILES

ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = get_configs_root()


def _config_facade():
    from web.services import config_service as _facade
    return _facade


def _sync_from_facade() -> None:
    """Keep path roots aligned with the config_service facade / test patches."""
    import sys

    global ROOT, CONFIGS_DIR
    facade = sys.modules.get("web.services.config_service")
    if facade is None:
        return
    if hasattr(facade, "ROOT"):
        ROOT = facade.ROOT
    if hasattr(facade, "CONFIGS_DIR"):
        CONFIGS_DIR = facade.CONFIGS_DIR


def get_config_file_meta(*args, **kwargs):
    return _config_facade().get_config_file_meta(*args, **kwargs)


def _normalize_config_rel_path(rel_path: str) -> str:
    return _config_paths.normalize_config_rel_path(rel_path)


def _safe_resolve(rel_path: str) -> Path | None:
    _sync_from_facade()
    return _config_paths.safe_resolve(rel_path, root=ROOT, configs_dir=CONFIGS_DIR)


def _normalize_dataset_preset_path(rel_path: str, *, must_exist: bool) -> str:
    raw = str(rel_path or "").replace("\\", "/").strip()
    if not raw:
        raise ValueError("缺少数据集预设路径")
    path = Path(raw)
    if path.is_absolute():
        try:
            raw = path.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError("数据集预设必须在项目目录内") from exc
        path = Path(raw)
    if ".." in path.parts:
        raise ValueError("数据集预设路径不能包含 ..")
    if path.suffix.lower() != ".toml":
        path = path.with_suffix(".toml")
    if len(path.parts) == 1:
        path = Path("configs") / "datasets" / path
    normalized = path.as_posix().lstrip("/")
    if not normalized.startswith("configs/datasets/"):
        raise ValueError("数据集预设必须保存在 configs/datasets/ 下")
    safe_path = _safe_resolve(normalized)
    if safe_path is None:
        raise ValueError("数据集预设路径不合法")
    if must_exist and not safe_path.exists():
        raise ValueError("数据集预设不存在")
    return normalized


def _is_dataset_preset_readonly(rel_path: str) -> bool:
    normalized = _normalize_config_rel_path(rel_path)
    if normalized in SYSTEM_DATASET_PRESET_FILES:
        return True
    try:
        return bool(get_config_file_meta(normalized).get("locked", False))
    except RuntimeError:
        return False


def _lock_reason_message(meta: dict[str, Any]) -> str:
    reason = str(meta.get("lock_reason") or meta.get("readonly_reason") or "").strip()
    return reason or "该配置由系统管理"
