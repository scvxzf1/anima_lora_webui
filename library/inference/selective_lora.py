"""Selective LoRA controls for Anima inference."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

import torch

ANIMA_MAIN_BLOCKS = tuple(f"block_{i}" for i in range(28))
ANIMA_LLM_ADAPTER_BLOCKS = tuple(f"llm_adapter_{i}" for i in range(6))
ANIMA_SPECIAL_BLOCKS = (
    "llm_adapter_io",
    "final_layer",
    "t_embedder",
    "x_embedder",
    "other_weights",
)
ANIMA_SELECTIVE_BLOCKS = (
    *ANIMA_MAIN_BLOCKS,
    *ANIMA_LLM_ADAPTER_BLOCKS,
    *ANIMA_SPECIAL_BLOCKS,
)
ANIMA_SELECTIVE_STRENGTH_STEP = 0.05
ANIMA_SELECTIVE_STRENGTH_MIN = 0.0
ANIMA_SELECTIVE_STRENGTH_MAX = 2.0

ANIMA_SELECTIVE_PRESETS = {
    "default": {
        "label": "默认",
        "blocks": ANIMA_SELECTIVE_BLOCKS,
        "strength": 1.0,
    },
    "all_off": {
        "label": "全关",
        "blocks": (),
        "strength": 0.0,
    },
    "half_strength": {
        "label": "半强度",
        "blocks": ANIMA_SELECTIVE_BLOCKS,
        "strength": 0.5,
    },
    "main_blocks_only": {
        "label": "仅主干",
        "blocks": (*ANIMA_MAIN_BLOCKS, "final_layer", "t_embedder", "x_embedder", "other_weights"),
        "strength": 1.0,
    },
    "llm_adapter_only": {
        "label": "仅 LLM Adapter",
        "blocks": (*ANIMA_LLM_ADAPTER_BLOCKS, "llm_adapter_io", "other_weights"),
        "strength": 1.0,
    },
    "late_main": {
        "label": "晚段主干 (20-27)",
        "blocks": tuple(f"block_{i}" for i in range(20, 28)) + ("final_layer", "t_embedder", "x_embedder", "other_weights"),
        "strength": 1.0,
    },
    "mid_late_main": {
        "label": "中后段主干 (14-27)",
        "blocks": tuple(f"block_{i}" for i in range(14, 28)) + ("final_layer", "t_embedder", "x_embedder", "other_weights"),
        "strength": 1.0,
    },
    "evens_only": {
        "label": "偶数层",
        "blocks": tuple(f"block_{i}" for i in range(0, 28, 2)) + tuple(f"llm_adapter_{i}" for i in range(0, 6, 2)),
        "strength": 1.0,
    },
    "odds_only": {
        "label": "奇数层",
        "blocks": tuple(f"block_{i}" for i in range(1, 28, 2)) + tuple(f"llm_adapter_{i}" for i in range(1, 6, 2)),
        "strength": 1.0,
    },
    "custom": {
        "label": "自定义",
        "blocks": ANIMA_SELECTIVE_BLOCKS,
        "strength": 1.0,
    },
}

_ANIMA_SELECTIVE_BLOCK_SET = frozenset(ANIMA_SELECTIVE_BLOCKS)
_ANIMA_SELECTIVE_PRESET_IDS = frozenset(ANIMA_SELECTIVE_PRESETS.keys())
_ANIMA_LLM_ADAPTER_BLOCK_RE = re.compile(r"(?:^|_)llm_adapter_blocks_(\d+)_")
_ANIMA_MAIN_BLOCK_RE = re.compile(r"(?:^|_)blocks_(\d+)_")


def normalize_anima_selective_preset(value: Any, *, default: str = "default") -> str:
    normalized = str(value or "").strip().lower() or default
    if normalized not in _ANIMA_SELECTIVE_PRESET_IDS:
        return default
    return normalized


def preset_blocks_for_anima_selective(preset: Any) -> list[str]:
    normalized = normalize_anima_selective_preset(preset)
    return list(ANIMA_SELECTIVE_PRESETS[normalized]["blocks"])


def preset_strength_for_anima_selective(preset: Any) -> float:
    normalized = normalize_anima_selective_preset(preset)
    return float(ANIMA_SELECTIVE_PRESETS[normalized]["strength"])


def _normalize_anima_selective_strength_value(
    value: Any,
    *,
    default: float = 0.0,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    parsed = max(ANIMA_SELECTIVE_STRENGTH_MIN, min(ANIMA_SELECTIVE_STRENGTH_MAX, parsed))
    stepped = round(parsed / ANIMA_SELECTIVE_STRENGTH_STEP) * ANIMA_SELECTIVE_STRENGTH_STEP
    return round(stepped + 1e-8, 2)


def normalize_anima_selective_blocks(
    values: Iterable[Any] | str | None,
    *,
    preset: Any = "default",
) -> list[str]:
    if values is None:
        return preset_blocks_for_anima_selective(preset)

    raw_items: list[str] = []
    if isinstance(values, str):
        raw_items = [item.strip() for item in values.split(",")]
    else:
        raw_items = [str(item or "").strip() for item in values]

    selected = {item for item in raw_items if item in _ANIMA_SELECTIVE_BLOCK_SET}
    return [block for block in ANIMA_SELECTIVE_BLOCKS if block in selected]


def preset_block_strengths_for_anima_selective(preset: Any) -> dict[str, float]:
    normalized = normalize_anima_selective_preset(preset)
    preset_blocks = set(preset_blocks_for_anima_selective(normalized))
    preset_strength = preset_strength_for_anima_selective(normalized)
    return {
        block: preset_strength if block in preset_blocks else 0.0
        for block in ANIMA_SELECTIVE_BLOCKS
    }


def normalize_anima_selective_block_strengths(
    values: Mapping[str, Any] | Iterable[Any] | str | None,
    *,
    preset: Any = "default",
) -> dict[str, float]:
    if values is None:
        return preset_block_strengths_for_anima_selective(preset)

    normalized = {block: 0.0 for block in ANIMA_SELECTIVE_BLOCKS}
    items: Iterable[tuple[Any, Any]]
    if isinstance(values, Mapping):
        items = values.items()
    else:
        raw_items: list[str]
        if isinstance(values, str):
            raw_items = [item.strip() for item in values.split(",")]
        else:
            raw_items = [str(item or "").strip() for item in values]
        parsed_items: list[tuple[str, str]] = []
        for item in raw_items:
            if "=" not in item:
                continue
            key, raw_value = item.split("=", 1)
            parsed_items.append((key.strip(), raw_value.strip()))
        items = parsed_items

    for raw_key, raw_value in items:
        key = str(raw_key or "").strip()
        if key not in _ANIMA_SELECTIVE_BLOCK_SET:
            continue
        normalized[key] = _normalize_anima_selective_strength_value(raw_value)
    return normalized


def enabled_blocks_from_anima_selective_strengths(
    values: Mapping[str, Any] | Iterable[Any] | str | None,
    *,
    preset: Any = "default",
) -> list[str]:
    strengths = normalize_anima_selective_block_strengths(values, preset=preset)
    return [block for block in ANIMA_SELECTIVE_BLOCKS if strengths.get(block, 0.0) > 0.0]


def classify_anima_lora_key(key: str) -> str:
    value = str(key or "").strip().lower()

    match = _ANIMA_LLM_ADAPTER_BLOCK_RE.search(value)
    if match:
        return f"llm_adapter_{match.group(1)}"

    match = _ANIMA_MAIN_BLOCK_RE.search(value)
    if match:
        return f"block_{match.group(1)}"

    if "llm_adapter_embed_" in value or "llm_adapter_norm_" in value or "llm_adapter_out_proj_" in value:
        return "llm_adapter_io"
    if "final_layer_" in value:
        return "final_layer"
    if "t_embedder_" in value or "t_embedding_norm_" in value:
        return "t_embedder"
    if "x_embedder_" in value:
        return "x_embedder"
    return "other_weights"


def apply_anima_selective_lora(
    weights_sd: Mapping[str, torch.Tensor],
    enabled_blocks: Iterable[Any] | str | None,
    *,
    strength: float = 1.0,
    preset: Any = "custom",
    block_strengths: Mapping[str, Any] | Iterable[Any] | str | None = None,
) -> dict[str, torch.Tensor]:
    scale = float(strength)
    if scale == 0.0:
        return {}

    if block_strengths is not None:
        normalized_strengths = normalize_anima_selective_block_strengths(
            block_strengths,
            preset=preset,
        )
    else:
        selected = set(normalize_anima_selective_blocks(enabled_blocks, preset=preset))
        if not selected:
            return {}
        normalized_strengths = {
            block: (scale if block in selected else 0.0)
            for block in ANIMA_SELECTIVE_BLOCKS
        }
        scale = 1.0

    filtered: dict[str, torch.Tensor] = {}
    for key, value in weights_sd.items():
        block_id = classify_anima_lora_key(key)
        block_scale = normalized_strengths.get(block_id, 0.0) * scale
        if block_scale <= 0.0:
            continue
        filtered[key] = value * block_scale if block_scale != 1.0 else value
    return filtered
