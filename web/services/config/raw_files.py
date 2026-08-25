"""Raw TOML read, write, patch, and delete helpers.

This module is loaded by ``web.services.config_service`` as part of the
compatibility facade.  It keeps facade access lazy so the module can also be
imported directly without pulling in the legacy facade.
"""

from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any

from library.env import get_configs_root, load_dotenv
from web.services.atomic_io import atomic_write_text
from web.services.config import file_groups as _file_groups
from web.services.config import paths as _config_paths
from web.services.config.schema_gate import (
    normalize_patch_values,
    validate_config_mapping,
)

_DELETE_TOML_KEY = object()

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
)

_LOCAL_HELPER_NAMES = {
    "get_config_file_meta",
}

_FILE_GROUP_SYNC_NAMES = (
    "ROOT",
    "CONFIGS_DIR",
    "WEB_FILE_GROUPS_FILE",
    "WEB_USER_LOCKS_FILE",
)


def _sync_from_facade() -> None:
    from web.services import config_service as _facade

    if hasattr(_facade, "_sync_legacy_from_facade"):
        _facade._sync_legacy_from_facade()
    _exported_names = set(globals().get("__all__", ()))
    for _name in _SYNC_NAMES:
        if not hasattr(_facade, _name):
            continue
        _value = getattr(_facade, _name)
        if _name not in _exported_names and _name not in _LOCAL_HELPER_NAMES:
            globals()[_name] = _value


def _exported(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        _sync_from_facade()
        return fn(*args, **kwargs)

    return wrapper


def _toml_module():
    import toml

    return toml


def _tomlkit_module():
    import tomlkit

    return tomlkit


def _metadata_value(name: str):
    from web.services.config import metadata as _metadata

    return getattr(_metadata, name)


def _normalize_config_rel_path(rel_path: str) -> str:
    return _config_paths.normalize_config_rel_path(rel_path)


def _safe_resolve(rel_path: str) -> Path | None:
    return _config_paths.safe_resolve(rel_path, root=ROOT, configs_dir=CONFIGS_DIR)


def _sync_file_groups_from_local() -> None:
    for _name in _FILE_GROUP_SYNC_NAMES:
        if _name in globals():
            setattr(_file_groups, _name, globals()[_name])


def _call_file_groups_impl(name: str, *args, **kwargs):
    _sync_file_groups_from_local()
    exported = getattr(_file_groups, name)
    impl = getattr(exported, "__wrapped__", exported)
    return impl(*args, **kwargs)


def get_config_file_meta(
    rel_path: str,
    group_id: str | None = None,
    group_label: str | None = None,
    locked: bool | None = None,
    trainable: bool | None = None,
    methods_subdir: str | None = None,
) -> dict[str, Any]:
    return _call_file_groups_impl(
        "get_config_file_meta",
        rel_path,
        group_id,
        group_label,
        locked,
        trainable,
        methods_subdir,
    )


def _load_user_locks() -> tuple[set[str], set[str]]:
    return _call_file_groups_impl("_load_user_locks")


def _save_user_locks(file_locks: set[str], group_locks: set[str]) -> None:
    return _call_file_groups_impl("_save_user_locks", file_locks, group_locks)


def _lock_reason_message(meta: dict[str, Any]) -> str:
    return _call_file_groups_impl("_lock_reason_message", meta)


__all__ = ['load_raw_file', 'save_raw_file', 'delete_raw_file', 'patch_raw_file_values', 'preview_raw_file_patch', '_prepare_raw_file_patch', '_restore_dataset_config_after_failed_train_patch', '_patch_toml_top_level', '_normalize_patch_value', '_normalize_saved_raw_config_content', '_normalize_saved_raw_config_content_with_changed_keys', '_is_blank_output_name']

def load_raw_file(rel_path: str) -> str:
    path = _safe_resolve(_normalize_config_rel_path(rel_path))
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def save_raw_file(
    rel_path: str,
    content: str,
    *,
    allow_locked: bool = False,
    overwrite: bool = True,
) -> tuple[bool, str, list[str]]:
    """Save a raw TOML config file.

    Returns ``(ok, message, warnings)``. ``warnings`` carries schema unknown-key
    notices and never blocks a successful save (invalid choices still error).
    """
    normalized = _normalize_config_rel_path(rel_path)
    path = _safe_resolve(normalized)
    if path is None:
        return False, "路径不合法", []
    if path.exists() and not overwrite:
        return False, "配置文件已存在，请换一个新的名称", []
    meta = get_config_file_meta(normalized)
    if meta.get("locked") and not allow_locked:
        return False, f"{_lock_reason_message(meta)}，请使用新名称保存新配置后编辑", []
    toml = _toml_module()
    tomlkit = _tomlkit_module()
    schema_warnings: list[str] = []
    try:
        parsed = toml.loads(content)
        content = _normalize_saved_raw_config_content(content)
        parsed = toml.loads(content)
    except (toml.TomlDecodeError, tomlkit.exceptions.TOMLKitError) as e:
        return False, f"TOML 语法错误: {e}", []
    except ValueError as e:
        return False, str(e), []
    if isinstance(parsed, dict):
        schema_errors, schema_warnings = validate_config_mapping(parsed)
        if schema_errors:
            return False, "; ".join(schema_errors), list(schema_warnings or [])
        schema_warnings = [str(item) for item in (schema_warnings or [])]
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, content)
    message = "保存成功"
    if schema_warnings:
        message = f"保存成功（警告: {'; '.join(schema_warnings)}）"
    return True, message, schema_warnings


def delete_raw_file(rel_path: str) -> tuple[bool, str]:
    normalized = _normalize_config_rel_path(rel_path)
    path = _safe_resolve(normalized)
    if path is None or path.suffix.lower() != ".toml":
        return False, "路径不合法，只能删除 configs/ 下的 TOML 文件"
    if not path.exists():
        return False, "配置文件不存在或已被删除"
    if not path.is_file():
        return False, "目标不是文件，已拒绝删除"

    meta = get_config_file_meta(normalized)
    if meta.get("locked"):
        return False, f"{_lock_reason_message(meta)}，不能删除"

    try:
        path.unlink()
    except OSError as e:
        return False, f"删除失败: {e}"

    user_locks, user_group_locks = _load_user_locks()
    if normalized in user_locks:
        user_locks.discard(normalized)
        _save_user_locks(user_locks, user_group_locks)

    return True, "删除成功"


def patch_raw_file_values(
    rel_path: str,
    values: dict[str, Any],
    *,
    content: str | None = None,
) -> tuple[bool, str, str, list[str], list[str]]:
    ok, msg, path, next_content, changed, warnings = _prepare_raw_file_patch(
        rel_path, values, content=content
    )
    if not ok or path is None:
        return False, msg, "", [], list(warnings or [])
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, next_content)
    return True, msg or "保存成功", next_content, changed, list(warnings or [])


def preview_raw_file_patch(
    rel_path: str,
    values: dict[str, Any],
    *,
    content: str | None = None,
) -> tuple[bool, str, str, list[str], list[str]]:
    ok, msg, _path, next_content, changed, warnings = _prepare_raw_file_patch(
        rel_path, values, content=content
    )
    if not ok:
        return False, msg, "", [], list(warnings or [])
    return True, msg or "预览成功", next_content, changed, list(warnings or [])


def _prepare_raw_file_patch(
    rel_path: str,
    values: dict[str, Any],
    *,
    content: str | None = None,
) -> tuple[bool, str, Path | None, str, list[str], list[str]]:
    normalized = _normalize_config_rel_path(rel_path)
    path = _safe_resolve(normalized)
    if path is None:
        return False, "路径不合法", None, "", [], []
    meta = get_config_file_meta(normalized)
    if meta.get("locked"):
        return False, f"{_lock_reason_message(meta)}，请使用新名称保存新配置后编辑", None, "", [], []
    if not isinstance(values, dict):
        return False, "字段补丁格式不合法", None, "", [], []
    ui_only_fields = _metadata_value("UI_ONLY_CONFIG_FIELDS")
    retired_fields = _metadata_value("RETIRED_TOP_LEVEL_CONFIG_FIELDS")
    values = {
        key: value
        for key, value in values.items()
        if key not in ui_only_fields and key not in retired_fields
    }
    values, schema_errors, schema_warnings = normalize_patch_values(values)
    schema_warnings = [str(item) for item in (schema_warnings or [])]
    if schema_errors:
        return False, "; ".join(schema_errors), None, "", [], schema_warnings

    source = content if content is not None else load_raw_file(rel_path)
    try:
        next_content = _patch_toml_top_level(source, values, rel_path=normalized)
        next_content, removed_keys = _remove_retired_top_level_fields(next_content)
        next_content, compatibility_keys = _normalize_saved_raw_config_content_with_changed_keys(next_content)
        _toml_module().loads(next_content)
    except Exception as e:
        return False, f"TOML 更新失败: {e}", None, "", [], schema_warnings

    changed_keys = {*values.keys(), *removed_keys, *compatibility_keys}
    ok_msg = "保存成功"
    if schema_warnings:
        ok_msg = f"保存成功（警告: {'; '.join(schema_warnings)}）"
    return True, ok_msg, path, next_content, sorted(changed_keys), schema_warnings


def _restore_dataset_config_after_failed_train_patch(path: Path, existed: bool, previous_content: str) -> None:
    if existed:
        atomic_write_text(path, previous_content)
        return
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _patch_toml_top_level(content: str, values: dict[str, Any], *, rel_path: str = "") -> str:
    tomlkit = _tomlkit_module()
    doc = tomlkit.parse(content or "")
    nested_patch_fields = _metadata_value("SPD_NESTED_PATCH_FIELDS") if _is_spd_patch_target(rel_path, doc) else {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            continue
        if "." in key or key in {"general", "datasets"}:
            raise ValueError(f"不支持写入嵌套字段: {key}")
        normalized = _normalize_patch_value(key, value)
        if key in nested_patch_fields:
            table_key, nested_key = nested_patch_fields[key]
            table = doc.get(table_key)
            if not isinstance(table, dict):
                table = tomlkit.table()
                doc[table_key] = table
            if key in doc:
                del doc[key]
            if normalized is _DELETE_TOML_KEY:
                if nested_key in table:
                    del table[nested_key]
                continue
            table[nested_key] = normalized
            continue
        if normalized is _DELETE_TOML_KEY:
            if key in doc:
                del doc[key]
            continue
        doc[key] = normalized
    return tomlkit.dumps(doc)


def _is_spd_patch_target(rel_path: str, doc: dict[str, Any]) -> bool:
    normalized = _normalize_config_rel_path(rel_path) if rel_path else ""
    if normalized == "configs/methods/spd.toml" or Path(normalized).stem == "spd":
        return True
    return all(key in doc for key in ("dit_path", "data_dir", "iterations")) and (
        "schedule" in doc or "network" in doc or "optim" in doc
    )


def _remove_retired_top_level_fields(content: str) -> tuple[str, list[str]]:
    tomlkit = _tomlkit_module()
    doc = tomlkit.parse(content or "")
    removed: list[str] = []
    for key in sorted(_metadata_value("RETIRED_TOP_LEVEL_CONFIG_FIELDS")):
        if key in doc:
            del doc[key]
            removed.append(key)
    if not removed:
        return content, []
    return tomlkit.dumps(doc), removed


# 适配器专属字段：从非对应变体切回时前端会发 null，表示从 TOML 顶层删除。
_ADAPTER_SCOPED_OPTIONAL_KEYS = frozenset(
    {
        "lokr_factor",
        "lokr_use_einsum",
        "lokr_decompose_w2",
        "lokr_full_factor",
        "lokr_allow_legacy_dim",
        "lokr_factor_group_size",
        "lokr_project_chunk_bytes",
        "vera_projection_prng_key",
        "vera_d_initial",
        "vera_save_projection",
    }
)


def _normalize_patch_value(key: str, value: Any) -> Any:
    if key == "output_name":
        if _is_blank_output_name(value):
            raise ValueError("output_name 不能为空")
        return str(value).strip()
    if key in _ADAPTER_SCOPED_OPTIONAL_KEYS and value is None:
        # 普通 LoRA 等变体不应保留 LoKr/VeRA 专属键。
        return _DELETE_TOML_KEY
    if key in {"sample_every_n_epochs", "sample_every_n_steps", "max_train_epochs"}:
        if value in ("", None):
            # TOML 没有 null。WebUI 留空表示禁用该可选数值，
            # 因此删除顶层键，让训练端按缺省 None 处理。
            return _DELETE_TOML_KEY
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} 必须是整数") from exc
    if key == "convrot_large_layer_mode" and value in ("", None):
        # ConvRot 大层模式留空表示“未启用大层特化”；schema choices 不含空串，
        # 因此删除顶层键，训练端会按 None 处理（与显式空串语义一致）。
        return _DELETE_TOML_KEY
    if key == "sample_at_first":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if key == "stage_schedule_enabled":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if key == "stage_schedule":
        if value in ("", None):
            return _DELETE_TOML_KEY
        if isinstance(value, str):
            import json

            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("stage_schedule 必须是数组") from exc
        if not isinstance(value, list):
            raise ValueError("stage_schedule 必须是数组")
        normalized_stages: list[dict[str, Any]] = []
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise ValueError(f"stage_schedule[{index}] 必须是对象")
            start = item.get("start_pct", item.get("startPct", 0))
            end = item.get("end_pct", item.get("endPct", 1))
            try:
                start_f = float(start)
                end_f = float(end)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"stage_schedule[{index}] 百分比无效") from exc
            if start_f > 1.0 or end_f > 1.0:
                start_f /= 100.0
                end_f /= 100.0
            subset_index = item.get("subset_index", item.get("subsetIndex", index))
            try:
                subset_i = int(subset_index)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"stage_schedule[{index}].subset_index 必须是整数") from exc
            normalized_stages.append(
                {
                    "name": str(item.get("name") or f"阶段{index + 1}").strip()
                    or f"阶段{index + 1}",
                    "subset_index": max(0, subset_i),
                    "start_pct": max(0.0, min(1.0, start_f)),
                    "end_pct": max(0.0, min(1.0, end_f)),
                }
            )
        return normalized_stages
    return value


def _is_blank_output_name(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def _normalize_saved_raw_config_content(content: str) -> str:
    normalized, _changed_keys = _normalize_saved_raw_config_content_with_changed_keys(content)
    return normalized


def _same_scalar_type(left: Any, right: Any) -> bool:
    if isinstance(right, bool):
        return isinstance(left, bool)
    if isinstance(right, int):
        return isinstance(left, int) and not isinstance(left, bool)
    if isinstance(right, float):
        return isinstance(left, float)
    if isinstance(right, str):
        return isinstance(left, str)
    return type(left) is type(right)


def _normalize_saved_raw_config_content_with_changed_keys(content: str) -> tuple[str, list[str]]:
    tomlkit = _tomlkit_module()
    doc = tomlkit.parse(content or "")
    if "output_name" in doc and _is_blank_output_name(doc["output_name"]):
        raise ValueError("output_name 不能为空")

    scalar_values = {
        key: value
        for key, value in doc.items()
        if isinstance(key, str) and not isinstance(value, (dict, list))
    }
    normalized_values, schema_errors, _schema_warnings = normalize_patch_values(
        scalar_values
    )
    if schema_errors:
        raise ValueError("; ".join(schema_errors))

    changed_keys: list[str] = []
    for key, normalized in normalized_values.items():
        original = scalar_values.get(key)
        if _same_scalar_type(original, normalized) and original == normalized:
            continue
        doc[key] = normalized
        changed_keys.append(key)

    optimizer_type = str(doc.get("optimizer_type") or "").strip().lower()
    if optimizer_type != "came" or "optimizer_args" not in doc:
        return (tomlkit.dumps(doc), changed_keys) if changed_keys else (content, [])
    raw_args = doc["optimizer_args"]
    if not isinstance(raw_args, list):
        return (tomlkit.dumps(doc), changed_keys) if changed_keys else (content, [])
    for index, arg in enumerate(raw_args):
        text = str(arg).strip()
        if not text.lower().startswith("betas="):
            continue
        raw_betas = text.split("=", 1)[1].strip()
        parts = [item.strip() for item in raw_betas.split(",") if item.strip()]
        if len(parts) == 2:
            raw_args[index] = "betas=0.9,0.999,0.9999"
            if "optimizer_args" not in changed_keys:
                changed_keys.append("optimizer_args")
            return tomlkit.dumps(doc), changed_keys
        return (tomlkit.dumps(doc), changed_keys) if changed_keys else (content, [])
    return (tomlkit.dumps(doc), changed_keys) if changed_keys else (content, [])

_SYNC_WRAPPED_EXPORTS = {
    "load_raw_file",
    "save_raw_file",
    "delete_raw_file",
    "patch_raw_file_values",
    "preview_raw_file_patch",
    "_prepare_raw_file_patch",
}


for _name in __all__:
    if _name in _SYNC_WRAPPED_EXPORTS:
        globals()[_name] = _exported(globals()[_name])
