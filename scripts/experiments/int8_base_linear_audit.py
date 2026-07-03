#!/usr/bin/env python
"""Offline int8 audit for frozen Anima base Linear weights.

This is a Phase-0 experiment for selective int8 storage:

* quantize only frozen DiT block Linear weights,
* keep AdaLN/modulation, final layer, timestep embedding, router/guidance,
  normalization, scale, bias, and adapter weights out of scope,
* use per-output-channel int8 + fp32 scale,
* dequantize back to fp32 only for measuring error.

The script does not alter training or checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open

INT8_MAX = 127.0
SCALE_EPS = 1e-12

BLOCK_WEIGHT_RE = re.compile(r"^(?:net\.)?blocks\.(?P<block>\d+)\.(?P<name>.+)\.weight$")

MLP_LINEAR_NAMES = {
    "mlp.layer1",
    "mlp.layer2",
}

ATTENTION_LINEAR_SUFFIXES = {
    "self_attn.qkv_proj",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.kv_proj",
    "self_attn.output_proj",
    "cross_attn.q_proj",
    "cross_attn.k_proj",
    "cross_attn.v_proj",
    "cross_attn.kv_proj",
    "cross_attn.output_proj",
}

SENSITIVE_NAME_FRAGMENTS = (
    "adaln",
    "final_layer",
    "t_embedder",
    "t_embedding",
    "pooled_text_proj",
    "mod_guidance",
    "router",
    "norm",
    "layer_norm",
    "rms",
    "bias",
    "scale",
    "lora",
    "adapter",
)


@dataclass(frozen=True)
class Candidate:
    key: str
    block_idx: int
    module_name: str
    family: str


@dataclass
class TensorAudit:
    key: str
    block_idx: int
    module_name: str
    family: str
    shape: list[int]
    numel: int
    rows: int
    source_dtype: str
    bf16_bytes: int
    payload_bytes: int
    payload_ratio_vs_bf16: float
    relative_l2: float
    mean_abs_error: float
    max_abs_error: float
    cosine: float
    scale_min: float
    scale_max: float
    zero_rows: int
    saturated_values: int


def _canonical_scope(scope: str) -> set[str]:
    normalized = {item.strip().lower() for item in scope.split(",") if item.strip()}
    if not normalized:
        return {"mlp"}
    if "all" in normalized:
        return {"mlp", "attention"}
    aliases = {
        "attn": "attention",
        "attention": "attention",
        "mlp": "mlp",
    }
    unknown = normalized - set(aliases)
    if unknown:
        raise ValueError(f"unknown audit scope: {', '.join(sorted(unknown))}")
    return {aliases[item] for item in normalized}


def classify_candidate_key(key: str, *, scope: str = "mlp") -> Candidate | None:
    """Return the int8 audit candidate represented by ``key``, or None."""
    lowered = key.lower()
    if not lowered.endswith(".weight"):
        return None
    if any(fragment in lowered for fragment in SENSITIVE_NAME_FRAGMENTS):
        return None

    match = BLOCK_WEIGHT_RE.match(key)
    if match is None:
        return None

    module_name = match.group("name")
    selected = _canonical_scope(scope)
    if module_name in MLP_LINEAR_NAMES and "mlp" in selected:
        return Candidate(
            key=key,
            block_idx=int(match.group("block")),
            module_name=module_name,
            family="mlp",
        )
    if module_name in ATTENTION_LINEAR_SUFFIXES and "attention" in selected:
        return Candidate(
            key=key,
            block_idx=int(match.group("block")),
            module_name=module_name,
            family="attention",
        )
    return None


def iter_candidate_keys(keys: Iterable[str], *, scope: str = "mlp") -> list[Candidate]:
    candidates = [
        candidate
        for key in keys
        if (candidate := classify_candidate_key(key, scope=scope)) is not None
    ]
    return sorted(candidates, key=lambda item: (item.block_idx, item.module_name, item.key))


def quantize_per_channel_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D+ tensor by output channel and return int8 values + scales."""
    if weight.dim() < 2:
        raise ValueError("per-channel int8 audit requires a 2D+ tensor")
    rows = weight.to(torch.float32).reshape(weight.shape[0], -1)
    amax = rows.abs().amax(dim=1)
    scale = (amax / INT8_MAX).clamp_min(SCALE_EPS)
    quantized = (rows / scale[:, None]).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    return quantized.reshape(weight.shape), scale


def dequantize_per_channel_int8(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    rows = quantized.reshape(quantized.shape[0], -1).to(torch.float32)
    dequantized = rows * scale.to(torch.float32)[:, None]
    return dequantized.reshape(quantized.shape)


def audit_tensor(candidate: Candidate, weight: torch.Tensor) -> TensorAudit:
    if weight.dim() < 2:
        raise ValueError(f"{candidate.key} is not a Linear matrix: shape={tuple(weight.shape)}")

    source = weight.detach().to(torch.float32)
    quantized, scale = quantize_per_channel_int8(source)
    dequantized = dequantize_per_channel_int8(quantized, scale)
    diff = dequantized - source
    denom = float(source.norm().item())
    relative_l2 = float(diff.norm().item()) / denom if denom > 0 else 0.0
    cosine = float(torch.nn.functional.cosine_similarity(
        source.reshape(1, -1),
        dequantized.reshape(1, -1),
        dim=1,
    ).item()) if source.numel() else 1.0
    bf16_bytes = int(source.numel()) * 2
    payload_bytes = int(source.numel()) + int(scale.numel()) * 4
    rows = source.reshape(source.shape[0], -1)

    return TensorAudit(
        key=candidate.key,
        block_idx=candidate.block_idx,
        module_name=candidate.module_name,
        family=candidate.family,
        shape=[int(dim) for dim in source.shape],
        numel=int(source.numel()),
        rows=int(source.shape[0]),
        source_dtype=str(weight.dtype).replace("torch.", ""),
        bf16_bytes=bf16_bytes,
        payload_bytes=payload_bytes,
        payload_ratio_vs_bf16=payload_bytes / bf16_bytes if bf16_bytes else 0.0,
        relative_l2=relative_l2,
        mean_abs_error=float(diff.abs().mean().item()) if diff.numel() else 0.0,
        max_abs_error=float(diff.abs().max().item()) if diff.numel() else 0.0,
        cosine=cosine,
        scale_min=float(scale.min().item()) if scale.numel() else 0.0,
        scale_max=float(scale.max().item()) if scale.numel() else 0.0,
        zero_rows=int((rows.abs().amax(dim=1) == 0).sum().item()),
        saturated_values=int((quantized.abs().to(torch.int16) == 127).sum().item()),
    )


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(q / 100.0 * len(ordered)) - 1))
    return ordered[idx]


def summarize(audits: list[TensorAudit], *, model: str, scope: str, gate_pct: float, elapsed_s: float) -> dict:
    rel = [item.relative_l2 for item in audits]
    payload = sum(item.payload_bytes for item in audits)
    bf16 = sum(item.bf16_bytes for item in audits)
    by_family: dict[str, list[TensorAudit]] = {}
    for item in audits:
        by_family.setdefault(item.family, []).append(item)

    return {
        "model": model,
        "scope": scope,
        "gate_pct": gate_pct,
        "elapsed_s": elapsed_s,
        "tensor_count": len(audits),
        "families": {
            family: {
                "tensor_count": len(items),
                "relative_l2_p50": percentile([item.relative_l2 for item in items], 50),
                "relative_l2_p95": percentile([item.relative_l2 for item in items], 95),
                "relative_l2_max": max((item.relative_l2 for item in items), default=0.0),
                "payload_ratio_vs_bf16": (
                    sum(item.payload_bytes for item in items) / sum(item.bf16_bytes for item in items)
                    if sum(item.bf16_bytes for item in items)
                    else 0.0
                ),
            }
            for family, items in sorted(by_family.items())
        },
        "relative_l2_p50": percentile(rel, 50),
        "relative_l2_p95": percentile(rel, 95),
        "relative_l2_max": max(rel) if rel else 0.0,
        "mean_abs_error_p95": percentile([item.mean_abs_error for item in audits], 95),
        "max_abs_error_p95": percentile([item.max_abs_error for item in audits], 95),
        "cosine_min": min((item.cosine for item in audits), default=1.0),
        "payload_bytes": payload,
        "bf16_bytes": bf16,
        "payload_ratio_vs_bf16": payload / bf16 if bf16 else 0.0,
        "gate_pass": percentile(rel, 95) < (gate_pct / 100.0) if audits else False,
        "worst": [
            asdict(item)
            for item in sorted(audits, key=lambda entry: entry.relative_l2, reverse=True)[:10]
        ],
    }


def run_audit(
    model: str | Path,
    *,
    out_dir: str | Path,
    scope: str = "mlp",
    gate_pct: float = 2.0,
    max_tensors: int | None = None,
) -> dict:
    model_path = Path(model)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    detail_path = out_path / "int8_base_linear_audit.jsonl"
    summary_path = out_path / "int8_base_linear_audit_summary.json"

    t0 = time.perf_counter()
    audits: list[TensorAudit] = []
    with safe_open(str(model_path), framework="pt", device="cpu") as f:
        candidates = iter_candidate_keys(f.keys(), scope=scope)
        if max_tensors is not None:
            candidates = candidates[: max(0, int(max_tensors))]
        with detail_path.open("w", encoding="utf-8") as detail:
            for candidate in candidates:
                audit = audit_tensor(candidate, f.get_tensor(candidate.key))
                audits.append(audit)
                detail.write(json.dumps(asdict(audit), ensure_ascii=False) + "\n")

    summary = summarize(
        audits,
        model=str(model_path),
        scope=scope,
        gate_pct=gate_pct,
        elapsed_s=time.perf_counter() - t0,
    )
    summary["detail_path"] = str(detail_path)
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _print_summary(summary: dict) -> None:
    print(f"model: {summary['model']}")
    print(f"scope: {summary['scope']}")
    print(
        f"tensors: {summary['tensor_count']} | "
        f"payload ratio vs bf16: {summary['payload_ratio_vs_bf16']:.4f} | "
        f"elapsed: {summary['elapsed_s']:.2f}s"
    )
    print(
        "relative L2: "
        f"p50={summary['relative_l2_p50'] * 100:.4f}% "
        f"p95={summary['relative_l2_p95'] * 100:.4f}% "
        f"max={summary['relative_l2_max'] * 100:.4f}% "
        f"gate={'PASS' if summary['gate_pass'] else 'FAIL'}"
    )
    for family, family_summary in summary["families"].items():
        print(
            f"  {family}: n={family_summary['tensor_count']} "
            f"p95={family_summary['relative_l2_p95'] * 100:.4f}% "
            f"max={family_summary['relative_l2_max'] * 100:.4f}% "
            f"ratio={family_summary['payload_ratio_vs_bf16']:.4f}"
        )
    print(f"detail: {summary['detail_path']}")
    print(f"summary: {summary['summary_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="models/diffusion_models/anima-base-v1.0.safetensors",
        help="Anima DiT safetensors checkpoint to audit.",
    )
    parser.add_argument("--out-dir", default="/tmp/anima-int8-base-linear-audit")
    parser.add_argument(
        "--scope",
        default="mlp",
        help="Comma-separated scopes: mlp, attention, all. Default: mlp.",
    )
    parser.add_argument("--gate-pct", type=float, default=2.0)
    parser.add_argument("--max-tensors", type=int, default=None)
    args = parser.parse_args()

    summary = run_audit(
        args.model,
        out_dir=args.out_dir,
        scope=args.scope,
        gate_pct=args.gate_pct,
        max_tensors=args.max_tensors,
    )
    _print_summary(summary)


if __name__ == "__main__":
    main()
