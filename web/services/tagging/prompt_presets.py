"""Persistent system/user prompt presets for the tagging workbench."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any
import uuid

import toml

from web.services.atomic_io import atomic_write_text

from . import settings as tagging_settings

MAX_PRESETS = 100
MAX_PRESET_NAME_LENGTH = 80
MAX_PROMPT_LENGTH = 10_000


def list_prompt_presets() -> dict[str, Any]:
    return {"ok": True, "presets": _load_presets()}


def create_prompt_preset(payload: dict[str, Any] | None) -> dict[str, Any]:
    values = _normalize_payload(payload, require_all=True)
    presets = _load_presets()
    if len(presets) >= MAX_PRESETS:
        raise ValueError(f"最多保存 {MAX_PRESETS} 个提示词预设")
    now = time.time()
    preset = {
        "id": uuid.uuid4().hex[:12],
        **values,
        "created_at": now,
        "updated_at": now,
    }
    presets.insert(0, preset)
    _save_presets(presets)
    return {"ok": True, "preset": dict(preset), "presets": presets}


def update_prompt_preset(preset_id: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    target_id = str(preset_id or "").strip()
    presets = _load_presets()
    index = next((idx for idx, item in enumerate(presets) if item["id"] == target_id), None)
    if index is None:
        raise KeyError("提示词预设不存在")
    current = presets[index]
    merged = {**current, **(payload if isinstance(payload, dict) else {})}
    values = _normalize_payload(merged, require_all=True)
    updated = {
        **current,
        **values,
        "updated_at": time.time(),
    }
    presets[index] = updated
    _save_presets(presets)
    return {"ok": True, "preset": dict(updated), "presets": presets}


def delete_prompt_preset(preset_id: str) -> dict[str, Any]:
    target_id = str(preset_id or "").strip()
    presets = _load_presets()
    remaining = [item for item in presets if item["id"] != target_id]
    if len(remaining) == len(presets):
        raise KeyError("提示词预设不存在")
    _save_presets(remaining)
    return {"ok": True, "deleted": target_id, "presets": remaining}


def _normalize_payload(payload: dict[str, Any] | None, *, require_all: bool) -> dict[str, str]:
    source = payload if isinstance(payload, dict) else {}
    name = str(source.get("name") or "").strip()
    system_prompt = str(source.get("system_prompt") or "").strip()
    user_prompt = str(source.get("user_prompt") or source.get("prompt") or "").strip()
    if require_all and not name:
        raise ValueError("请输入预设名称")
    if require_all and not system_prompt:
        raise ValueError("请输入系统提示词")
    if require_all and not user_prompt:
        raise ValueError("请输入用户提示词")
    if len(name) > MAX_PRESET_NAME_LENGTH:
        raise ValueError(f"预设名称最多 {MAX_PRESET_NAME_LENGTH} 个字符")
    if len(system_prompt) > MAX_PROMPT_LENGTH or len(user_prompt) > MAX_PROMPT_LENGTH:
        raise ValueError(f"单条提示词最多 {MAX_PROMPT_LENGTH} 个字符")
    return {"name": name, "system_prompt": system_prompt, "user_prompt": user_prompt}


def _load_presets() -> list[dict[str, Any]]:
    path = _presets_file()
    try:
        raw = toml.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, toml.TomlDecodeError):
        raw = {}
    values = raw.get("presets") if isinstance(raw, dict) else []
    if not isinstance(values, list):
        return []
    presets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_item in values[:MAX_PRESETS]:
        if not isinstance(raw_item, dict):
            continue
        preset_id = str(raw_item.get("id") or "").strip()
        if not preset_id or preset_id in seen:
            continue
        try:
            normalized = _normalize_payload(raw_item, require_all=True)
        except ValueError:
            continue
        seen.add(preset_id)
        presets.append(
            {
                "id": preset_id[:64],
                **normalized,
                "created_at": _timestamp(raw_item.get("created_at")),
                "updated_at": _timestamp(raw_item.get("updated_at")),
            }
        )
    return presets


def _save_presets(presets: list[dict[str, Any]]) -> None:
    path = _presets_file()
    atomic_write_text(path, toml.dumps({"presets": presets}))
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _presets_file() -> Path:
    return Path(tagging_settings.SETTINGS_FILE).with_name("prompt-presets.toml")


def _timestamp(value: Any) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0
