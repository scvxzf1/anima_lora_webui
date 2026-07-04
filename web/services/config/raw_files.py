"""Raw TOML read, write, patch, and delete helpers.

This module is loaded by ``web.services.config_service`` as part of the
compatibility facade.  It keeps facade access lazy so the module can also be
imported directly without pulling in the legacy facade.
"""

from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any

from web.services.config import paths as _config_paths
from web.services.config.file_groups import (
    _load_user_locks,
    _lock_reason_message,
    _safe_resolve,
    _save_user_locks,
    get_config_file_meta,
)

_DELETE_TOML_KEY = object()

_SYNC_NAMES = (
    "ROOT",
    "CONFIGS_DIR",
    "GUI_METHODS_DIR",
    "IMPORTED_CONFIGS_DIR",
    "PRESETS_FILE",
    "WEB_FILE_GROUPS_FILE",
    "WEB_USER_LOCKS_FILE",
    "DATASET_PRESETS_DIR",
    "resolve_output_root",
    "_display_settings_path",
    "save_raw_file",
    "load_raw_file",
    "delete_raw_file",
    "patch_raw_file_values",
    "preview_raw_file_patch",
    "get_config_file_meta",
    "list_config_file_groups",
    "move_config_file_to_group",
    "_inspect_network_weight",
    "LOGGER",
)

_LEGACY_HELPER_NAMES = (
    "_safe_resolve",
    "_normalize_config_rel_path",
    "_load_user_locks",
    "_save_user_locks",
    "_lock_reason_message",
)
_LEGACY_FILE_GROUP_SHIM_NAMES = {
    "get_config_file_meta",
    "list_config_file_groups",
    "move_config_file_to_group",
}


def _sync_from_facade() -> None:
    from web.services import config_service as _facade

    _exported_names = set(globals().get("__all__", ()))
    _legacy_module = getattr(_facade, "_legacy", None)
    for _name in _SYNC_NAMES:
        if not hasattr(_facade, _name):
            continue
        _value = getattr(_facade, _name)
        if _name not in _exported_names:
            globals()[_name] = _value
        if (
            _legacy_module is not None
            and _name not in _exported_names
            and _name not in _LEGACY_FILE_GROUP_SHIM_NAMES
        ):
            setattr(_legacy_module, _name, _value)
    for _name in _LEGACY_HELPER_NAMES:
        if _legacy_module is not None and hasattr(_legacy_module, _name):
            globals()[_name] = getattr(_legacy_module, _name)
        elif hasattr(_facade, _name):
            globals()[_name] = getattr(_facade, _name)


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
) -> tuple[bool, str]:
    normalized = _normalize_config_rel_path(rel_path)
    path = _safe_resolve(normalized)
    if path is None:
        return False, "路径不合法"
    if path.exists() and not overwrite:
        return False, "配置文件已存在，请换一个新的名称"
    meta = get_config_file_meta(normalized)
    if meta.get("locked") and not allow_locked:
        return False, f"{_lock_reason_message(meta)}，请使用新名称保存新配置后编辑"
    toml = _toml_module()
    tomlkit = _tomlkit_module()
    try:
        toml.loads(content)
        content = _normalize_saved_raw_config_content(content)
    except (toml.TomlDecodeError, tomlkit.exceptions.TOMLKitError) as e:
        return False, f"TOML 语法错误: {e}"
    except ValueError as e:
        return False, str(e)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True, "保存成功"


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
) -> tuple[bool, str, str, list[str]]:
    ok, msg, path, next_content, changed = _prepare_raw_file_patch(rel_path, values, content=content)
    if not ok or path is None:
        return False, msg, "", []
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(next_content, encoding="utf-8")
    return True, "保存成功", next_content, changed


def preview_raw_file_patch(
    rel_path: str,
    values: dict[str, Any],
    *,
    content: str | None = None,
) -> tuple[bool, str, str, list[str]]:
    ok, msg, _path, next_content, changed = _prepare_raw_file_patch(rel_path, values, content=content)
    if not ok:
        return False, msg, "", []
    return True, "预览成功", next_content, changed


def _prepare_raw_file_patch(
    rel_path: str,
    values: dict[str, Any],
    *,
    content: str | None = None,
) -> tuple[bool, str, Path | None, str, list[str]]:
    normalized = _normalize_config_rel_path(rel_path)
    path = _safe_resolve(normalized)
    if path is None:
        return False, "路径不合法", None, "", []
    meta = get_config_file_meta(normalized)
    if meta.get("locked"):
        return False, f"{_lock_reason_message(meta)}，请使用新名称保存新配置后编辑", None, "", []
    if not isinstance(values, dict):
        return False, "字段补丁格式不合法", None, "", []
    ui_only_fields = _metadata_value("UI_ONLY_CONFIG_FIELDS")
    retired_fields = _metadata_value("RETIRED_TOP_LEVEL_CONFIG_FIELDS")
    values = {
        key: value
        for key, value in values.items()
        if key not in ui_only_fields and key not in retired_fields
    }

    source = content if content is not None else load_raw_file(rel_path)
    try:
        next_content = _patch_toml_top_level(source, values, rel_path=normalized)
        next_content, removed_keys = _remove_retired_top_level_fields(next_content)
        next_content, compatibility_keys = _normalize_saved_raw_config_content_with_changed_keys(next_content)
        _toml_module().loads(next_content)
    except Exception as e:
        return False, f"TOML 更新失败: {e}", None, "", []

    changed_keys = {*values.keys(), *removed_keys, *compatibility_keys}
    return True, "保存成功", path, next_content, sorted(changed_keys)


def _restore_dataset_config_after_failed_train_patch(path: Path, existed: bool, previous_content: str) -> None:
    if existed:
        path.write_text(previous_content, encoding="utf-8")
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


def _normalize_patch_value(key: str, value: Any) -> Any:
    if key == "output_name":
        if _is_blank_output_name(value):
            raise ValueError("output_name 不能为空")
        return str(value).strip()
    if key in {"sample_every_n_epochs", "sample_every_n_steps", "max_train_epochs"}:
        if value in ("", None):
            # TOML 没有 null。WebUI 留空表示禁用该可选数值，
            # 因此删除顶层键，让训练端按缺省 None 处理。
            return _DELETE_TOML_KEY
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} 必须是整数") from exc
    if key == "sample_at_first":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    return value


def _is_blank_output_name(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def _normalize_saved_raw_config_content(content: str) -> str:
    normalized, _changed_keys = _normalize_saved_raw_config_content_with_changed_keys(content)
    return normalized


def _normalize_saved_raw_config_content_with_changed_keys(content: str) -> tuple[str, list[str]]:
    tomlkit = _tomlkit_module()
    doc = tomlkit.parse(content or "")
    if "output_name" in doc and _is_blank_output_name(doc["output_name"]):
        raise ValueError("output_name 不能为空")
    optimizer_type = str(doc.get("optimizer_type") or "").strip().lower()
    if optimizer_type != "came" or "optimizer_args" not in doc:
        return content, []
    raw_args = doc["optimizer_args"]
    if not isinstance(raw_args, list):
        return content, []
    for index, arg in enumerate(raw_args):
        text = str(arg).strip()
        if not text.lower().startswith("betas="):
            continue
        raw_betas = text.split("=", 1)[1].strip()
        parts = [item.strip() for item in raw_betas.split(",") if item.strip()]
        if len(parts) == 2:
            raw_args[index] = "betas=0.9,0.999,0.9999"
            return tomlkit.dumps(doc), ["optimizer_args"]
        return content, []
    return content, []




for _name in __all__:
    globals()[_name] = _exported(globals()[_name])
