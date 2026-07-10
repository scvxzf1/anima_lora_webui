"""Per-layer ΔW reconstruction for LoRA / LoHa / LoKr."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import torch

from web.services.weight_analysis.constants import (
    BLOCK_RE,
    CHARACTER_PRIORITY,
    LOHA_SUFFIX,
    LOKR_SUFFIX,
    LORA_SUFFIX,
    STYLE_PRIORITY,
)


def _compute_layers(
    state: Mapping[str, torch.Tensor],
    adapter_type: str,
    metadata: Mapping[str, str],
) -> list[dict[str, Any]]:
    if adapter_type == "LoRA":
        return _compute_lora_layers(state, metadata)
    if adapter_type == "LoHa":
        return _compute_loha_layers(state, metadata)
    if adapter_type == "LoKr":
        return _compute_lokr_layers(state, metadata)
    return []


def _compute_lora_layers(state: Mapping[str, torch.Tensor], metadata: Mapping[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(state):
        if not key.endswith(LORA_SUFFIX):
            continue
        prefix = key.removesuffix(LORA_SUFFIX)
        down = state.get(key)
        up = state.get(f"{prefix}.lora_up.weight")
        if down is None or up is None:
            continue
        alpha = _alpha_value(state, prefix, default=_rank_from_down(down, metadata))
        rank = max(1, _rank_from_down(down, metadata))
        try:
            delta = _lora_delta(up, down, alpha, rank)
        except ValueError as exc:
            rows.append(_layer_error(prefix, "LoRA", str(exc), [down, up]))
            continue
        rows.append(_layer_stats(prefix, "LoRA", delta, [down, up], alpha=alpha, rank=rank))
        del delta
    return rows


def _compute_loha_layers(state: Mapping[str, torch.Tensor], metadata: Mapping[str, str]) -> list[dict[str, Any]]:
    del metadata
    rows: list[dict[str, Any]] = []
    for key in sorted(state):
        if not key.endswith(LOHA_SUFFIX):
            continue
        prefix = key.removesuffix(LOHA_SUFFIX)
        w1_a = state.get(f"{prefix}.hada_w1_a")
        w1_b = state.get(f"{prefix}.hada_w1_b")
        w2_a = state.get(f"{prefix}.hada_w2_a")
        w2_b = state.get(f"{prefix}.hada_w2_b")
        tensors = [tensor for tensor in (w1_a, w1_b, w2_a, w2_b) if tensor is not None]
        if len(tensors) != 4:
            continue
        alpha = _alpha_value(state, prefix, default=_rank_from_loha(w1_a, w1_b))
        rank = max(1, _rank_from_loha(w1_a, w1_b))
        try:
            delta = ((w1_a.float() @ w1_b.float()) * (w2_a.float() @ w2_b.float())) * (alpha / rank)
        except RuntimeError as exc:
            rows.append(_layer_error(prefix, "LoHa", str(exc), tensors))
            continue
        rows.append(_layer_stats(prefix, "LoHa", delta, tensors, alpha=alpha, rank=rank))
        del delta
    return rows


def _compute_lokr_layers(state: Mapping[str, torch.Tensor], metadata: Mapping[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(state):
        if not key.endswith(LOKR_SUFFIX):
            continue
        prefix = key.removesuffix(LOKR_SUFFIX)
        w1 = state.get(f"{prefix}.lokr_w1")
        w2 = state.get(f"{prefix}.lokr_w2")
        if w1 is None or w2 is None:
            continue
        network_dim = _metadata_network_dim(metadata)
        rank_guess = max(1, int(w1.shape[0]) if w1.dim() >= 1 else 1)
        alpha = _alpha_value(state, prefix, default=network_dim or rank_guess)
        # LoKr 的实际 Kronecker factor 不是 LoRA rank。训练加载器在没有
        # ss_network_dim 时会退回用 alpha 作为 modules_dim，因此这里同样
        # 优先使用元数据 rank，其次使用 alpha，最后才退回 factor 大小。
        rank = max(1, int(network_dim or round(alpha) or rank_guess))
        try:
            delta = torch.kron(w1.float(), w2.float()) * (alpha / rank)
        except RuntimeError as exc:
            rows.append(_layer_error(prefix, "LoKr", str(exc), [w1, w2]))
            continue
        rows.append(_layer_stats(prefix, "LoKr", delta, [w1, w2], alpha=alpha, rank=rank))
        del delta
    return rows


def _lora_delta(up: torch.Tensor, down: torch.Tensor, alpha: float, rank: int) -> torch.Tensor:
    up_f = up.float()
    down_f = down.float()
    if down_f.dim() == 2 and up_f.dim() == 2:
        return (up_f @ down_f) * (alpha / max(rank, 1))
    if down_f.dim() == 4 and up_f.dim() == 4:
        if tuple(down_f.shape[2:4]) == (1, 1):
            return (
                up_f.squeeze(3).squeeze(2) @ down_f.squeeze(3).squeeze(2)
            ).unsqueeze(2).unsqueeze(3) * (alpha / max(rank, 1))
        return torch.nn.functional.conv2d(down_f.permute(1, 0, 2, 3), up_f).permute(1, 0, 2, 3) * (
            alpha / max(rank, 1)
        )
    raise ValueError(f"暂不支持的 LoRA 张量维度: up={tuple(up.shape)}, down={tuple(down.shape)}")


def _alpha_value(state: Mapping[str, torch.Tensor], prefix: str, *, default: int | float) -> float:
    tensor = state.get(f"{prefix}.alpha")
    if tensor is None:
        return float(default or 1)
    try:
        return float(tensor.detach().float().cpu().reshape(-1)[0].item())
    except Exception:
        return float(default or 1)


def _rank_from_down(down: torch.Tensor, metadata: Mapping[str, str]) -> int:
    if down.dim() >= 1:
        return int(down.shape[0])
    try:
        return int(float(metadata.get("ss_network_dim") or 1))
    except (TypeError, ValueError):
        return 1


def _rank_from_loha(w1_a: torch.Tensor, w1_b: torch.Tensor) -> int:
    if w1_a.dim() == 2:
        return int(w1_a.shape[1])
    if w1_b.dim() == 2:
        return int(w1_b.shape[0])
    return 1


def _metadata_network_dim(metadata: Mapping[str, str]) -> int | None:
    try:
        value = int(float(metadata.get("ss_network_dim") or ""))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _layer_stats(
    name: str,
    adapter_type: str,
    delta: torch.Tensor,
    source_tensors: Iterable[torch.Tensor],
    *,
    alpha: float,
    rank: int,
) -> dict[str, Any]:
    delta_f = delta.detach().float()
    abs_delta = delta_f.abs()
    fro_sq = float(delta_f.pow(2).sum().item())
    fro_norm = float(fro_sq ** 0.5)
    mean_abs = float(abs_delta.mean().item()) if delta_f.numel() else 0.0
    max_abs = float(abs_delta.max().item()) if delta_f.numel() else 0.0
    parsed = parse_layer_name(name)
    return {
        "name": name,
        "adapter_type": adapter_type,
        "block": parsed["block"],
        "block_label": parsed["block_label"],
        "component": parsed["component"],
        "component_label": parsed["component_label"],
        "fro_norm": fro_norm,
        "fro_norm_sq": fro_sq,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "param_count": int(sum(int(t.numel()) for t in source_tensors)),
        "delta_param_count": int(delta_f.numel()),
        "shape": list(delta_f.shape),
        "alpha": float(alpha),
        "rank": int(rank),
        "contribution": 0.0,
        "energy_contribution": 0.0,
        "style_score": 0.0,
        "character_score": 0.0,
        "notes": [],
    }


def _layer_error(name: str, adapter_type: str, error: str, source_tensors: Iterable[torch.Tensor]) -> dict[str, Any]:
    parsed = parse_layer_name(name)
    return {
        "name": name,
        "adapter_type": adapter_type,
        "block": parsed["block"],
        "block_label": parsed["block_label"],
        "component": parsed["component"],
        "component_label": parsed["component_label"],
        "fro_norm": 0.0,
        "fro_norm_sq": 0.0,
        "mean_abs": 0.0,
        "max_abs": 0.0,
        "param_count": int(sum(int(t.numel()) for t in source_tensors)),
        "delta_param_count": 0,
        "shape": [],
        "alpha": 0.0,
        "rank": 0,
        "contribution": 0.0,
        "energy_contribution": 0.0,
        "style_score": 0.0,
        "character_score": 0.0,
        "notes": [f"该层 ΔW 重建失败: {error}"],
    }


def parse_layer_name(name: str) -> dict[str, Any]:
    match = BLOCK_RE.search(name)
    if match:
        component = _normalize_component(match.group("component"))
        block = int(match.group("block"))
        return {
            "block": block,
            "block_label": str(block),
            "component": component,
            "component_label": component,
        }
    return {
        "block": None,
        "block_label": "其他",
        "component": _normalize_component(name),
        "component_label": _normalize_component(name),
    }


def _normalize_component(component: str) -> str:
    value = str(component or "").strip("_")
    for prefix in ("lora_unet_", "unet_"):
        if value.startswith(prefix):
            value = value.removeprefix(prefix)
    return value or "unknown"


def _finalize_layer_contributions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total_fro = sum(float(row.get("fro_norm") or 0.0) for row in rows) or 0.0
    total_energy = sum(float(row.get("fro_norm_sq") or 0.0) for row in rows) or 0.0
    for row in rows:
        fro = float(row.get("fro_norm") or 0.0)
        energy = float(row.get("fro_norm_sq") or 0.0)
        row["contribution"] = fro / total_fro if total_fro > 0 else 0.0
        row["energy_contribution"] = energy / total_energy if total_energy > 0 else 0.0
        row["style_score"] = _candidate_score(row, kind="style")
        row["character_score"] = _candidate_score(row, kind="character")
        row["notes"] = _layer_notes(row)
        row.pop("fro_norm_sq", None)
    return sorted(rows, key=lambda item: float(item.get("fro_norm") or 0.0), reverse=True)


def _candidate_score(row: Mapping[str, Any], *, kind: str) -> float:
    component = str(row.get("component") or "")
    block = row.get("block")
    block_int = int(block) if isinstance(block, int) else None
    fro = float(row.get("fro_norm") or 0.0)
    if kind == "style":
        weight = STYLE_PRIORITY.get(component, 0.72)
        if block_int is not None:
            if 13 <= block_int <= 18:
                weight *= 1.24
            elif 25 <= block_int <= 26:
                weight *= 1.20
            elif 9 <= block_int <= 24:
                weight *= 1.08
            elif 0 <= block_int <= 8:
                weight *= 0.84
        return fro * weight
    weight = CHARACTER_PRIORITY.get(component, 0.76)
    if block_int is not None:
        if 0 <= block_int <= 8:
            weight *= 1.24
        elif 9 <= block_int <= 18:
            weight *= 1.12
        elif block_int >= 25:
            weight *= 0.86
    return fro * weight


def _layer_notes(row: Mapping[str, Any]) -> list[str]:
    notes: list[str] = []
    component = str(row.get("component") or "")
    block = row.get("block")
    if component in {"mlp_layer1", "cross_attn_k_proj", "cross_attn_v_proj", "self_attn_output_proj"}:
        notes.append("风格候选高权重层类型")
    if component in {"self_attn_q_proj", "self_attn_v_proj", "cross_attn_q_proj", "cross_attn_k_proj", "cross_attn_v_proj"}:
        notes.append("角色/结构候选注意力层")
    if isinstance(block, int) and (13 <= block <= 18 or 25 <= block <= 26):
        notes.append("中后段重点 block")
    if isinstance(block, int) and 0 <= block <= 8:
        notes.append("早期 block，通常更偏结构基础")
    return notes

