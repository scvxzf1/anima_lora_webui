"""Ordered global base-model configurations for the Web UI."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

import toml

from library.env import resolve_model_family
from library.models.family_registry import (
    MODEL_FAMILY_REGISTRY,
    get_model_family_spec,
    normalize_registered_family,
)
from web.services import settings_service
from web.services.atomic_io import atomic_write_text

MODEL_CONFIG_SECTION = "model_config_library"
MODEL_PATH_KEYS = settings_service.GLOBAL_MODEL_PATH_KEYS
MODEL_CONFIG_FIELDS = (
    "id",
    "name",
    "model_family",
    *MODEL_PATH_KEYS,
)
MODEL_CONFIG_GROUP_FIELDS = ("id", "label", "item_ids")
DEFAULT_MODEL_CONFIG_GROUP_ID = "ungrouped"
DEFAULT_MODEL_CONFIG_GROUP_LABEL = "未分组"
MODEL_CONFIG_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


class ModelConfigConflictError(ValueError):
    """Raised when the settings file changed after the client loaded it."""


class ModelConfigFileError(ValueError):
    """Raised when an existing settings file cannot be parsed safely."""


def get_model_configs() -> dict[str, Any]:
    settings_file = Path(settings_service.SETTINGS_FILE)
    raw, revision = _load_raw_settings_strict(settings_file)
    section = raw.get(MODEL_CONFIG_SECTION)
    if section is None:
        items, default_id = _legacy_model_configs(raw, settings_file)
        groups, _ = _normalize_groups(None, items, allow_repair=True)
        return _response(
            items,
            default_id,
            groups,
            revision=revision,
            migrated=True,
            groups_migrated=False,
        )
    items, default_id = _normalize_library(section, allow_generated_ids=False)
    groups, groups_migrated = _normalize_groups(
        section.get("groups"),
        items,
        allow_repair=False,
    )
    items = _order_items_by_groups(items, groups)
    return _response(
        items,
        default_id,
        groups,
        revision=revision,
        migrated=False,
        groups_migrated=groups_migrated,
    )


def save_model_configs(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("模型配置请求必须是对象")

    settings_file = Path(settings_service.SETTINGS_FILE)
    raw, current_revision = _load_raw_settings_strict(settings_file)
    expected_revision = data.get("revision")
    if not isinstance(expected_revision, str):
        raise ValueError("缺少模型配置 revision")
    if expected_revision != current_revision:
        raise ModelConfigConflictError("模型配置已被其他页面修改，请刷新后重试")

    items, default_id = _normalize_library(data, allow_generated_ids=True)
    groups_supplied = "groups" in data
    raw_groups = data.get("groups")
    if not groups_supplied:
        current_section = raw.get(MODEL_CONFIG_SECTION)
        if isinstance(current_section, dict):
            raw_groups = current_section.get("groups")
    groups, _ = _normalize_groups(
        raw_groups,
        items,
        allow_repair=not groups_supplied,
    )
    if not groups_supplied:
        groups = _sort_group_items_by_flat_order(groups, items)
    items = _order_items_by_groups(items, groups)
    default_item = next(item for item in items if item["id"] == default_id)

    next_raw = dict(raw)
    next_raw[MODEL_CONFIG_SECTION] = {
        "default_id": default_id,
        "items": [{key: item[key] for key in MODEL_CONFIG_FIELDS} for item in items],
        "groups": [{key: group[key] for key in MODEL_CONFIG_GROUP_FIELDS} for group in groups],
    }
    global_section = raw.get("global") if isinstance(raw.get("global"), dict) else {}
    next_global = dict(global_section)
    for key in MODEL_PATH_KEYS:
        next_global[key] = default_item[key]
    if default_item["model_family"] == "anima":
        next_global.pop("model_family", None)
    else:
        next_global["model_family"] = default_item["model_family"]
    next_raw["global"] = next_global

    settings_file.parent.mkdir(parents=True, exist_ok=True)
    serialized = toml.dumps(next_raw)
    atomic_write_text(settings_file, serialized)
    return {
        **_response(
            items,
            default_id,
            groups,
            revision=_revision_for_text(serialized),
            migrated=False,
            groups_migrated=False,
        ),
        "message": "全局模型配置已保存",
    }


def _load_raw_settings_strict(settings_file: Path) -> tuple[dict[str, Any], str]:
    if not settings_file.exists():
        return {}, _revision_for_text("")
    try:
        text = settings_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise ModelConfigFileError(f"无法读取全局设置文件: {exc}") from exc
    try:
        raw = toml.loads(text)
    except toml.TomlDecodeError as exc:
        raise ModelConfigFileError("全局设置 TOML 已损坏，已拒绝覆盖") from exc
    if not isinstance(raw, dict):
        raise ModelConfigFileError("全局设置 TOML 顶层必须是对象")
    return raw, _revision_for_text(text)


def _revision_for_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _legacy_model_configs(
    raw: dict[str, Any],
    settings_file: Path,
) -> tuple[list[dict[str, str]], str]:
    global_section = raw.get("global") if isinstance(raw.get("global"), dict) else {}
    base = _load_base_settings(settings_file.parent / "base.toml")
    raw_family = str(
        global_section.get("model_family")
        or base.get("model_family")
        or resolve_model_family()
    )
    family = _normalize_family(raw_family)
    item = {
        "id": "legacy-default",
        "name": f"{get_model_family_spec(family).display_name} 默认配置",
        "model_family": family,
    }
    for key in MODEL_PATH_KEYS:
        item[key] = _clean_path(global_section.get(key) or base.get(key) or "")
    return [item], item["id"]


def _load_base_settings(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = toml.loads(path.read_text(encoding="utf-8"))
    except (OSError, toml.TomlDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _normalize_library(
    data: Any,
    *,
    allow_generated_ids: bool,
) -> tuple[list[dict[str, str]], str]:
    if not isinstance(data, dict):
        raise ValueError("模型配置库必须是对象")
    raw_items = data.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("至少需要保留一个全局模型配置")

    items: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for index, raw_item in enumerate(raw_items, start=1):
        if not isinstance(raw_item, dict):
            raise ValueError(f"第 {index} 个模型配置必须是对象")
        item_id = str(raw_item.get("id") or "").strip()
        if not item_id and allow_generated_ids:
            item_id = f"model-{uuid4().hex[:12]}"
        if not MODEL_CONFIG_ID_PATTERN.fullmatch(item_id):
            raise ValueError(f"第 {index} 个模型配置 ID 无效")
        if item_id in seen_ids:
            raise ValueError("模型配置 ID 不能重复")
        seen_ids.add(item_id)

        name = str(raw_item.get("name") or "").strip()
        if not name:
            raise ValueError(f"第 {index} 个模型配置缺少名称")
        if len(name) > 80:
            raise ValueError("模型配置名称不能超过 80 个字符")
        normalized_name = name.casefold()
        if normalized_name in seen_names:
            raise ValueError("模型配置名称不能重复")
        seen_names.add(normalized_name)

        item = {
            "id": item_id,
            "name": name,
            "model_family": _normalize_family(raw_item.get("model_family")),
        }
        for key in MODEL_PATH_KEYS:
            value = _clean_path(raw_item.get(key))
            if not value:
                raise ValueError(f"模型配置“{name}”缺少 {key}")
            item[key] = value
        items.append(item)

    default_id = str(data.get("default_id") or "").strip()
    if default_id not in seen_ids:
        raise ValueError("默认模型配置必须指向现有配置")
    return items, default_id


def _normalize_groups(
    raw_groups: Any,
    items: list[dict[str, str]],
    *,
    allow_repair: bool,
) -> tuple[list[dict[str, Any]], bool]:
    item_ids = [item["id"] for item in items]
    known_ids = set(item_ids)
    if raw_groups is None:
        return [
            {
                "id": DEFAULT_MODEL_CONFIG_GROUP_ID,
                "label": DEFAULT_MODEL_CONFIG_GROUP_LABEL,
                "item_ids": item_ids,
            }
        ], True
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError("至少需要保留一个模型配置分组")

    groups: list[dict[str, Any]] = []
    seen_group_ids: set[str] = set()
    seen_labels: set[str] = set()
    assigned_ids: set[str] = set()
    for index, raw_group in enumerate(raw_groups, start=1):
        if not isinstance(raw_group, dict):
            raise ValueError(f"第 {index} 个模型配置分组必须是对象")
        group_id = str(raw_group.get("id") or "").strip()
        if not MODEL_CONFIG_ID_PATTERN.fullmatch(group_id):
            raise ValueError(f"第 {index} 个模型配置分组 ID 无效")
        if group_id in seen_group_ids:
            raise ValueError("模型配置分组 ID 不能重复")
        seen_group_ids.add(group_id)

        label = str(raw_group.get("label") or "").strip()
        if not label:
            raise ValueError(f"第 {index} 个模型配置分组缺少名称")
        if len(label) > 80:
            raise ValueError("模型配置分组名称不能超过 80 个字符")
        normalized_label = label.casefold()
        if normalized_label in seen_labels:
            raise ValueError("模型配置分组名称不能重复")
        seen_labels.add(normalized_label)

        raw_item_ids = raw_group.get("item_ids")
        if not isinstance(raw_item_ids, list):
            raise ValueError(f"模型配置分组“{label}”的 item_ids 必须是列表")
        group_item_ids: list[str] = []
        for raw_item_id in raw_item_ids:
            item_id = str(raw_item_id or "").strip()
            if item_id not in known_ids:
                if allow_repair:
                    continue
                raise ValueError(f"模型配置分组“{label}”引用了不存在的配置")
            if item_id in assigned_ids:
                raise ValueError("每个模型配置只能属于一个分组")
            assigned_ids.add(item_id)
            group_item_ids.append(item_id)
        groups.append({"id": group_id, "label": label, "item_ids": group_item_ids})

    missing_ids = [item_id for item_id in item_ids if item_id not in assigned_ids]
    if missing_ids and not allow_repair:
        raise ValueError("每个模型配置都必须属于一个分组")
    if missing_ids:
        groups[0]["item_ids"].extend(missing_ids)
    return groups, False


def _order_items_by_groups(
    items: list[dict[str, str]],
    groups: list[dict[str, Any]],
) -> list[dict[str, str]]:
    items_by_id = {item["id"]: item for item in items}
    return [items_by_id[item_id] for group in groups for item_id in group["item_ids"]]


def _sort_group_items_by_flat_order(
    groups: list[dict[str, Any]],
    items: list[dict[str, str]],
) -> list[dict[str, Any]]:
    order = {item["id"]: index for index, item in enumerate(items)}
    return [
        {
            **group,
            "item_ids": sorted(group["item_ids"], key=lambda item_id: order[item_id]),
        }
        for group in groups
    ]


def _normalize_family(value: Any) -> str:
    try:
        return normalize_registered_family(
            value,
            source="WebUI model config family",
            allow_aliases=True,
        )
    except ValueError as exc:
        raise ValueError(
            f"模型格式仅支持 anima、krea2_raw 或 z_image；注册值: "
            f"{', '.join(MODEL_FAMILY_REGISTRY)}: {exc}"
        ) from exc


def _clean_path(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _response(
    items: list[dict[str, str]],
    default_id: str,
    groups: list[dict[str, Any]],
    *,
    revision: str,
    migrated: bool,
    groups_migrated: bool,
) -> dict[str, Any]:
    return {
        "ok": True,
        "items": [
            {
                **item,
                "complete": all(bool(item.get(key)) for key in MODEL_PATH_KEYS),
            }
            for item in items
        ],
        "default_id": default_id,
        "groups": [
            {key: group[key] for key in MODEL_CONFIG_GROUP_FIELDS}
            for group in groups
        ],
        "revision": revision,
        "migrated": migrated,
        "groups_migrated": groups_migrated,
    }
