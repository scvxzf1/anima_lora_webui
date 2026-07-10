"""Summary, candidate ranking and heatmap aggregation."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from web.services.weight_analysis.constants import TOP_LAYER_LIMIT


def _summary(
    file_name: str,
    adapter_type: str,
    metadata: Mapping[str, str],
    layers: list[Mapping[str, Any]],
) -> dict[str, Any]:
    total_fro = sum(float(layer.get("fro_norm") or 0.0) for layer in layers)
    total_energy = sum(float(layer.get("fro_norm") or 0.0) ** 2 for layer in layers)
    total_params = sum(int(layer.get("param_count") or 0) for layer in layers)
    blocks = sorted({int(layer["block"]) for layer in layers if isinstance(layer.get("block"), int)})
    components = sorted({str(layer.get("component") or "") for layer in layers if layer.get("component")})
    early = _average_fro_for_blocks(layers, range(0, 9))
    mid_late = _average_fro_for_blocks(layers, range(13, 27))
    ratio = (mid_late / early) if early > 0 else None
    top_layer = layers[0] if layers else {}
    return {
        "file_name": file_name,
        "adapter_type": adapter_type,
        "output_name": str(metadata.get("ss_output_name") or ""),
        "layer_count": len(layers),
        "component_count": len(components),
        "block_count": len(blocks),
        "blocks": blocks,
        "components": components,
        "total_fro_norm": total_fro,
        "total_energy": total_energy,
        "total_param_count": total_params,
        "top_layer": top_layer.get("name", ""),
        "top_component": top_layer.get("component", ""),
        "top_block": top_layer.get("block"),
        "early_avg_fro_norm": early,
        "mid_late_avg_fro_norm": mid_late,
        "mid_late_vs_early_ratio": ratio,
        "conclusion": [
            "这是基于权重能量的静态推断，不是实际 prompt 激活。",
            "中后段 block 与高能层更可能承载风格信息，但仍需 prompt probe / heatmap / ablation 验证。",
        ],
    }


def _average_fro_for_blocks(layers: Iterable[Mapping[str, Any]], block_range: range) -> float:
    values = [float(layer.get("fro_norm") or 0.0) for layer in layers if layer.get("block") in block_range]
    return sum(values) / len(values) if values else 0.0


def _component_summary(layers: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for layer in layers:
        groups.setdefault(str(layer.get("component") or "unknown"), []).append(layer)
    rows = [_aggregate_group(component, items, key="component") for component, items in groups.items()]
    return sorted(rows, key=lambda item: float(item.get("fro_norm") or 0.0), reverse=True)


def _block_summary(layers: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for layer in layers:
        block = layer.get("block")
        label = str(block) if isinstance(block, int) else "其他"
        groups.setdefault(label, []).append(layer)
    rows = [_aggregate_group(label, items, key="block") for label, items in groups.items()]
    return sorted(rows, key=lambda item: (item.get("block") is None, -(float(item.get("fro_norm") or 0.0))))


def _aggregate_group(label: str, items: list[Mapping[str, Any]], *, key: str) -> dict[str, Any]:
    fro = sum(float(item.get("fro_norm") or 0.0) for item in items)
    energy = sum(float(item.get("fro_norm") or 0.0) ** 2 for item in items)
    params = sum(int(item.get("param_count") or 0) for item in items)
    max_item = max(items, key=lambda item: float(item.get("fro_norm") or 0.0))
    row = {
        key: None if key == "block" and label == "其他" else (_int_or_none(label) if key == "block" else label),
        "label": label,
        "layer_count": len(items),
        "fro_norm": fro,
        "energy": energy,
        "mean_abs": sum(float(item.get("mean_abs") or 0.0) for item in items) / len(items),
        "max_abs": max(float(item.get("max_abs") or 0.0) for item in items),
        "param_count": params,
        "top_layer": max_item.get("name", ""),
        "top_component": max_item.get("component", ""),
        "contribution": sum(float(item.get("contribution") or 0.0) for item in items),
        "energy_contribution": sum(float(item.get("energy_contribution") or 0.0) for item in items),
    }
    return row


def _top_candidates(layers: list[Mapping[str, Any]], *, kind: str) -> list[dict[str, Any]]:
    score_key = "style_score" if kind == "style" else "character_score"
    sorted_layers = sorted(layers, key=lambda item: float(item.get(score_key) or 0.0), reverse=True)
    return [
        {
            "rank": index + 1,
            "name": layer.get("name", ""),
            "block": layer.get("block"),
            "component": layer.get("component", ""),
            "fro_norm": layer.get("fro_norm", 0.0),
            "mean_abs": layer.get("mean_abs", 0.0),
            "max_abs": layer.get("max_abs", 0.0),
            "contribution": layer.get("contribution", 0.0),
            "score": layer.get(score_key, 0.0),
            "reason": _candidate_reason(layer, kind=kind),
        }
        for index, layer in enumerate(sorted_layers[:TOP_LAYER_LIMIT])
    ]


def _candidate_reason(layer: Mapping[str, Any], *, kind: str) -> str:
    component = str(layer.get("component") or "")
    block = layer.get("block")
    parts: list[str] = []
    if kind == "style":
        if component == "mlp_layer1":
            parts.append("mlp_layer1 更像风格染色器")
        elif component in {"cross_attn_k_proj", "cross_attn_v_proj"}:
            parts.append("cross-attn k/v 更像提示词到风格的绑定")
        elif component == "self_attn_output_proj":
            parts.append("self-attn output 影响整体气质和一致性")
        else:
            parts.append("按 ΔW 范数与层类型权重排序")
        if isinstance(block, int) and (13 <= block <= 18 or 25 <= block <= 26):
            parts.append("处于中后段重点 block")
    else:
        if component in {"self_attn_q_proj", "self_attn_v_proj"}:
            parts.append("self-attn q/v 更偏结构塑形候选")
        elif component in {"cross_attn_q_proj", "cross_attn_k_proj", "cross_attn_v_proj"}:
            parts.append("cross-attn q/k/v 可能影响角色提示绑定")
        else:
            parts.append("按早中段与注意力层启发式排序")
        if isinstance(block, int) and 0 <= block <= 8:
            parts.append("低 block 高能，偏角色/结构候选")
    return "；".join(parts)


def _heatmap(layers: list[Mapping[str, Any]]) -> dict[str, Any]:
    block_values = sorted({int(layer["block"]) for layer in layers if isinstance(layer.get("block"), int)})
    component_values = _ordered_components(str(layer.get("component") or "unknown") for layer in layers)
    matrix: list[list[float]] = []
    cells: list[dict[str, Any]] = []
    max_value = 0.0
    lookup: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for layer in layers:
        block = layer.get("block")
        component = str(layer.get("component") or "unknown")
        if not isinstance(block, int):
            continue
        lookup.setdefault((block, component), []).append(layer)
    for block in block_values:
        row: list[float] = []
        for component in component_values:
            items = lookup.get((block, component), [])
            value = sum(float(item.get("fro_norm") or 0.0) for item in items)
            row.append(value)
            max_value = max(max_value, value)
            if value > 0:
                cells.append(
                    {
                        "block": block,
                        "component": component,
                        "fro_norm": value,
                        "layer_count": len(items),
                        "top_layer": max(items, key=lambda item: float(item.get("fro_norm") or 0.0)).get("name", ""),
                    }
                )
        matrix.append(row)
    for cell in cells:
        cell["intensity"] = float(cell["fro_norm"] or 0.0) / max_value if max_value > 0 else 0.0
    return {
        "blocks": block_values,
        "components": component_values,
        "matrix": matrix,
        "max_value": max_value,
        "cells": cells,
    }


def _ordered_components(components: Iterable[str]) -> list[str]:
    preferred = [
        "self_attn_q_proj",
        "self_attn_k_proj",
        "self_attn_v_proj",
        "self_attn_output_proj",
        "cross_attn_q_proj",
        "cross_attn_k_proj",
        "cross_attn_v_proj",
        "cross_attn_output_proj",
        "mlp_layer1",
        "mlp_layer2",
    ]
    unique = sorted(set(components))
    return [item for item in preferred if item in unique] + [item for item in unique if item not in preferred]


def _empty_summary() -> dict[str, Any]:
    return {
        "file_name": "",
        "adapter_type": "",
        "output_name": "",
        "layer_count": 0,
        "component_count": 0,
        "block_count": 0,
        "blocks": [],
        "components": [],
        "total_fro_norm": 0.0,
        "total_energy": 0.0,
        "total_param_count": 0,
        "top_layer": "",
        "top_component": "",
        "top_block": None,
        "early_avg_fro_norm": 0.0,
        "mid_late_avg_fro_norm": 0.0,
        "mid_late_vs_early_ratio": None,
        "conclusion": [],
    }


def _empty_heatmap() -> dict[str, Any]:
    return {"blocks": [], "components": [], "matrix": [], "max_value": 0.0, "cells": []}




def _int_or_none(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None
