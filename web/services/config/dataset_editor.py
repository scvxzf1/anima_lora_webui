"""Dataset editor load/save API extracted from datasets service.

Keeps facade access lazy so the module can be imported without pulling the
legacy config facade. Shared row/document helpers live in dataset_rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import toml

from library.env import expand_env_vars_in_obj, get_configs_root, load_dotenv
from web.services.config import paths as _config_paths
from web.services.config.dataset_rows import (
    _build_dataset_config_doc,
    _stage_schedule_fields_from_dataset_data,
    _normalize_stage_schedule_list,
    _dataset_defaults_from_config,
    _dataset_rows_from_config,
    _dataset_summary_from_rows,
    _dataset_training_defaults,
    _ensure_training_dataset_rows,
    _fill_missing_dataset_row_settings,
    _first_training_dataset_row,
    _normalize_dataset_defaults,
    _normalize_dataset_rows,
    _safe_file_stem,
    _single_dataset_config_from_cfg,
)
from web.services.config.preflight_stage_schedule import validate_stage_schedule_or_raise
from web.services.config.file_groups import _lock_reason_message
from web.services.settings_service import resolve_output_root

ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = get_configs_root()

load_dotenv()


def _config_facade():
    from web.services import config_service as _facade

    return _facade


def _sync_from_facade() -> None:
    """Keep path roots aligned with the config_service facade / test patches.

    Prefer an already-imported facade so pure helper callers stay facade-free.
    """
    import sys

    global ROOT, CONFIGS_DIR
    facade = sys.modules.get("web.services.config_service")
    if facade is None:
        return
    if hasattr(facade, "ROOT"):
        ROOT = facade.ROOT
    if hasattr(facade, "CONFIGS_DIR"):
        CONFIGS_DIR = facade.CONFIGS_DIR


def save_raw_file(*args, **kwargs):
    return _config_facade().save_raw_file(*args, **kwargs)


def get_config_file_meta(*args, **kwargs):
    return _config_facade().get_config_file_meta(*args, **kwargs)


def load_merged_config(*args, **kwargs):
    return _config_facade().load_merged_config(*args, **kwargs)


def apply_auto_data_dirs(*args, **kwargs):
    return _config_facade().apply_auto_data_dirs(*args, **kwargs)


def _prepare_raw_file_patch(*args, **kwargs):
    return _config_facade()._prepare_raw_file_patch(*args, **kwargs)


def _load_training_config_for_web_run(*args, **kwargs):
    return _config_facade()._load_training_config_for_web_run(*args, **kwargs)


def _normalize_config_rel_path(rel_path: str) -> str:
    return _config_paths.normalize_config_rel_path(rel_path)


def _safe_resolve(rel_path: str) -> Path | None:
    _sync_from_facade()
    return _config_paths.safe_resolve(rel_path, root=ROOT, configs_dir=CONFIGS_DIR)


def _safe_config_subdir(subdir: str) -> Path | None:
    _sync_from_facade()
    return _config_paths.safe_config_subdir(subdir, configs_dir=CONFIGS_DIR)


def _resolve_project_path(value: str) -> Path:
    from library.env import expand_env_vars

    _sync_from_facade()
    return _config_paths.resolve_display_path(
        value,
        root=ROOT,
        configs_dir=CONFIGS_DIR,
        expand_env_vars_fn=expand_env_vars,
    )


def _display_path(path: Path) -> str:
    _sync_from_facade()
    return _config_paths.display_path(path, root=ROOT, configs_dir=CONFIGS_DIR)


def _restore_dataset_config_after_failed_train_patch(path: Path, existed: bool, previous_content: str) -> None:
    if existed:
        path.write_text(previous_content, encoding="utf-8")
        return
    try:
        path.unlink()
    except FileNotFoundError:
        return

def load_dataset_editor(
    variant: str,
    preset: str,
    methods_subdir: str = "gui-methods",
    *,
    config_file: str | None = None,
    dataset_config: str | None = None,
) -> dict[str, Any]:
    _sync_from_facade()
    cfg = _load_training_config_for_web_run(
        variant,
        preset,
        methods_subdir,
        config_file=config_file,
    )
    if dataset_config is not None:
        dataset_rel = _normalize_config_rel_path(str(dataset_config or ""))
        if dataset_rel:
            cfg["dataset_config"] = dataset_rel
        else:
            cfg.pop("dataset_config", None)
    dataset_path = _dataset_config_path_from_cfg(cfg)
    if dataset_path and dataset_path.exists():
        data = toml.loads(dataset_path.read_text(encoding="utf-8"))
    else:
        data = _single_dataset_config_from_cfg(cfg)
    rows = _dataset_rows_from_config(data, cfg)
    result = {
        "ok": True,
        "dataset_config": _display_path(dataset_path) if dataset_path else "",
        "datasets": rows,
        "defaults": _dataset_defaults_from_config(data),
    }
    result.update(_stage_schedule_fields_from_dataset_data(data))
    return result

def save_dataset_editor(
    variant: str,
    preset: str,
    methods_subdir: str,
    rows: list[dict[str, Any]],
    defaults: dict[str, Any] | None = None,
    config_values: dict[str, Any] | None = None,
    train_file: str | None = None,
    train_content: str | None = None,
    prefer_existing_dataset_config: bool = True,
) -> dict[str, Any]:
    _sync_from_facade()
    raw_file_saver = save_raw_file
    cfg = apply_auto_data_dirs(load_merged_config(variant, preset, methods_subdir))
    if defaults:
        cfg.update(_normalize_dataset_defaults(defaults))
    clean_rows = _fill_missing_dataset_row_settings(_normalize_dataset_rows(rows), _normalize_dataset_defaults(cfg))
    if not clean_rows:
        raise ValueError("请至少填写一个数据集路径")
    _ensure_training_dataset_rows(clean_rows)

    train_rel = _normalize_config_rel_path(train_file) if train_file else _training_config_rel_path(variant, methods_subdir)
    dataset_variant = Path(train_rel).stem if train_rel else variant
    dataset_rel = _dataset_config_rel_path(
        cfg,
        dataset_variant,
        methods_subdir,
        prefer_existing=prefer_existing_dataset_config,
    )
    dataset_path = _safe_resolve(dataset_rel)
    if dataset_path is None:
        raise ValueError("数据集配置路径不合法")

    if train_rel and get_config_file_meta(train_rel).get("locked"):
        raise ValueError(f"{_lock_reason_message(get_config_file_meta(train_rel))}，请使用新名称保存新配置后编辑")

    next_content = ""
    if train_rel:
        first = _first_training_dataset_row(clean_rows)
        compatibility_defaults = _dataset_training_defaults(clean_rows, cfg)
        values = {
            "dataset_config": dataset_rel,
            "source_image_dir": first["source_dir"],
            "resized_image_dir": first["image_dir"],
            "lora_cache_dir": first["cache_dir"],
            "prior_loss_weight": compatibility_defaults["prior_loss_weight"],
        }
        ok, msg, _train_path, next_content, _changed, _warnings = _prepare_raw_file_patch(train_rel, values, content=train_content)
        if not ok:
            raise ValueError(msg)

    doc_cfg = dict(cfg)
    if train_rel:
        try:
            doc_cfg.update(expand_env_vars_in_obj(toml.loads(next_content)))
        except toml.TomlDecodeError as exc:
            raise ValueError(f"训练配置 TOML 解析失败: {exc}") from exc
    elif train_content is not None:
        try:
            doc_cfg.update(expand_env_vars_in_obj(toml.loads(str(train_content or ""))))
        except toml.TomlDecodeError as exc:
            raise ValueError(f"训练配置 TOML 解析失败: {exc}") from exc
    if config_values:
        doc_cfg.update(expand_env_vars_in_obj(dict(config_values)))
    if defaults:
        doc_cfg.update(_normalize_dataset_defaults(defaults))
    doc_cfg = apply_auto_data_dirs(doc_cfg)

    # Preserve stage schedule owned by the dataset config unless caller overrides.
    if dataset_path.exists():
        try:
            import toml as _toml
            existing_stage = _stage_schedule_fields_from_dataset_data(
                _toml.loads(dataset_path.read_text(encoding="utf-8"))
            )
        except Exception:
            existing_stage = {}
        if "stage_schedule_enabled" in existing_stage:
            doc_cfg["stage_schedule_enabled"] = existing_stage["stage_schedule_enabled"]
        if "stage_schedule" in existing_stage:
            doc_cfg["stage_schedule"] = list(existing_stage["stage_schedule"])
    if config_values:
        if "stage_schedule_enabled" in config_values:
            doc_cfg["stage_schedule_enabled"] = bool(config_values.get("stage_schedule_enabled"))
        if "stage_schedule" in config_values:
            doc_cfg["stage_schedule"] = _normalize_stage_schedule_list(config_values.get("stage_schedule"))
    validate_stage_schedule_or_raise(doc_cfg, dataset_rows=clean_rows)

    dataset_doc = _build_dataset_config_doc(
        clean_rows,
        doc_cfg,
        prefer_train_batch_size=True,
    )
    dataset_existed = dataset_path.exists()
    previous_dataset_doc = dataset_path.read_text(encoding="utf-8") if dataset_existed else ""
    ok, msg, *_rest = raw_file_saver(dataset_rel, dataset_doc, overwrite=True)
    if not ok:
        raise ValueError(msg)
    if train_rel:
        ok, msg, *_rest = raw_file_saver(train_rel, next_content, overwrite=True)
        if not ok:
            _restore_dataset_config_after_failed_train_patch(dataset_path, dataset_existed, previous_dataset_doc)
            raise ValueError(msg)

    saved_defaults = _normalize_dataset_defaults(doc_cfg)
    result = {
        "ok": True,
        "message": f"已保存 {len(clean_rows)} 个数据集路径",
        "dataset_config": dataset_rel,
        "datasets": clean_rows,
        "defaults": saved_defaults,
        "summary": _dataset_summary_from_rows(clean_rows, saved_defaults),
        "train_content": next_content,
    }
    if "stage_schedule_enabled" in doc_cfg:
        result["stage_schedule_enabled"] = bool(doc_cfg.get("stage_schedule_enabled"))
    if "stage_schedule" in doc_cfg:
        result["stage_schedule"] = _normalize_stage_schedule_list(doc_cfg.get("stage_schedule"))
    return result

def _dataset_config_path_from_cfg(cfg: dict[str, Any]) -> Path | None:
    rel_path = str(cfg.get("dataset_config") or "").strip()
    if not rel_path:
        return None
    path = _resolve_project_path(rel_path)
    if path.suffix.lower() != ".toml":
        return None
    if not _is_allowed_dataset_config_path(path):
        return None
    return path

def _is_allowed_dataset_config_path(path: Path) -> bool:
    resolved = path.resolve()
    for root in (ROOT.resolve(), CONFIGS_DIR.resolve(), resolve_output_root().resolve()):
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    if resolved.name == "dataset.runtime.toml":
        run_dir = resolved.parent
        return (
            resolved.is_file()
            and (run_dir / "config.runtime.toml").is_file()
            and (run_dir / "model_cache").is_dir()
            and (run_dir / "dataset_cache").is_dir()
            and (run_dir / "training_output").is_dir()
        )
    return False

def _dataset_config_rel_path(
    cfg: dict[str, Any],
    variant: str,
    methods_subdir: str,
    *,
    prefer_existing: bool = True,
) -> str:
    existing = str(cfg.get("dataset_config") or "").strip()
    if prefer_existing and existing:
        normalized = _normalize_config_rel_path(existing)
        path = _safe_resolve(normalized)
        if path is not None and normalized.startswith("configs/datasets/"):
            return normalized
    stem = _safe_file_stem(variant or methods_subdir or "dataset")
    return f"configs/datasets/{stem}.toml"

def _training_config_rel_path(variant: str, methods_subdir: str) -> str:
    methods_dir = _safe_config_subdir(methods_subdir)
    if methods_dir is None:
        return ""
    stem = _safe_file_stem(variant)
    path = methods_dir / f"{stem}.toml"
    if not path.exists():
        return ""
    return _display_path(path)
