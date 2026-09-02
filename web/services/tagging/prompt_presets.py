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
DEFAULT_USER_PROMPT = "用训练 caption 描述主体、服装、姿态、镜头和背景，不要添加无法确认的内容。"

# Imported from AnimaLoraStudio's captioning workspace.  These are templates,
# not user data: they are always available, never written to the profile file,
# and can be copied into an editable preset with the New action.
BUILTIN_PROMPT_PRESETS: tuple[dict[str, Any], ...] = (
    {
        "id": "builtin-detailed",
        "name": "Flux / SD3 长描述",
        "system_prompt": "Write one detailed English training caption. Describe the visible subject, pose, expression, composition, camera angle, lighting, palette, materials, spatial relationships and background. Be factual, avoid quality claims, and return only one fluent paragraph.",
    },
    {
        "id": "builtin-danbooru",
        "name": "Danbooru / Anime 标签",
        "system_prompt": "Return only precise standard Danbooru-style English tags separated by comma and space. Include subject count, identity traits, clothing, pose, expression, framing and background. Do not use Markdown, sentences or quality boilerplate.",
    },
    {
        "id": "builtin-character-action",
        "name": "主体解耦 / 姿态动作",
        "system_prompt": "Return only comma-separated English tags useful for character LoRA training. Focus on pose, action, expression, framing, camera angle and environment. Exclude permanent identity traits, artist and character names. Do not use Markdown.",
    },
    {
        "id": "builtin-anima-three-format",
        "name": "Anima 三格式训练 Caption",
        "system_prompt": "Analyze the image for Anima or LoRA training and return exactly three standalone JSON objects, without Markdown or additional text. Their type fields must be tag, mixed_70tag_30nl, and pure_nl. Every object must contain one non-empty caption string. The tag version must be a clean English comma-separated tag sequence. The mixed version should contain mostly tags followed by a concise natural-language description. The pure_nl version must be fluent English prose and must not be a Booru tag list. Describe only visible or explicitly supplied facts.",
    },
    {
        "id": "builtin-anima-style-overfit",
        "name": "Anima 画风过拟合纯 Tag",
        "system_prompt": "Create a short English tag caption for strongly fitting a style LoRA. Put the user-supplied style trigger first and preserve it exactly. Tag variable visible content such as subjects, appearance, clothing, pose, props, framing and background, but omit repeated fixed style traits so the LoRA learns them through the trigger. Use 20 to 45 deduplicated comma-separated tags. Do not add quality, score, safety or generic detailed-art tags unless explicitly requested. Return only the caption.",
    },
    {
        "id": "builtin-anima-style-trigger-json",
        "name": "Anima 固定触发串全量 Tag",
        "system_prompt": "Return exactly one valid JSON object without Markdown or extra text. The object must have type \"full_tag_with_style_trigger\" and a non-empty string field named caption. Preserve the user-supplied fixed style trigger sequence at the start, then append complete, visible English content tags in a clean comma-and-space sequence. Do not repeat trigger tags, invent uncertain details or add quality and safety boilerplate.",
    },
)


def list_prompt_presets(*, include_builtins: bool = False) -> dict[str, Any]:
    presets = _load_presets()
    if include_builtins:
        presets = _with_builtins(presets)
    return {"ok": True, "presets": presets}


def create_prompt_preset(
    payload: dict[str, Any] | None, *, include_builtins: bool = False
) -> dict[str, Any]:
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
    visible = _with_builtins(presets) if include_builtins else presets
    return {"ok": True, "preset": dict(preset), "presets": visible}


def update_prompt_preset(
    preset_id: str,
    payload: dict[str, Any] | None,
    *,
    include_builtins: bool = False,
) -> dict[str, Any]:
    target_id = str(preset_id or "").strip()
    if _builtin_by_id(target_id) is not None:
        raise ValueError("内置提示词预设不可直接修改，请新建可编辑副本")
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
    visible = _with_builtins(presets) if include_builtins else presets
    return {"ok": True, "preset": dict(updated), "presets": visible}


def delete_prompt_preset(preset_id: str, *, include_builtins: bool = False) -> dict[str, Any]:
    target_id = str(preset_id or "").strip()
    if _builtin_by_id(target_id) is not None:
        raise ValueError("内置提示词预设不可删除")
    presets = _load_presets()
    remaining = [item for item in presets if item["id"] != target_id]
    if len(remaining) == len(presets):
        raise KeyError("提示词预设不存在")
    _save_presets(remaining)
    visible = _with_builtins(remaining) if include_builtins else remaining
    return {"ok": True, "deleted": target_id, "presets": visible}


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


def _builtin_by_id(preset_id: str) -> dict[str, Any] | None:
    return next((item for item in BUILTIN_PROMPT_PRESETS if item["id"] == preset_id), None)


def _with_builtins(presets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return immutable templates followed by user-owned presets."""

    builtins = [
        {
            "id": item["id"],
            "name": item["name"],
            "system_prompt": item["system_prompt"],
            "user_prompt": DEFAULT_USER_PROMPT,
            "created_at": 0.0,
            "updated_at": 0.0,
            "builtin": True,
        }
        for item in BUILTIN_PROMPT_PRESETS
    ]
    return builtins + [dict(item, builtin=False) for item in presets]


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
