#!/usr/bin/env python
"""Small-batch equivalence probe for frozen int8 base Linear storage.

This is a Phase-1 smoke test for the selective int8 route. It does not load a
real Anima checkpoint and does not alter the training entrypoint. The default
probe uses an Anima-shaped module surface; ``--model-kind anima`` uses the real
``library.anima.models.Anima`` tiny forward path:

* frozen ``blocks.*.mlp.layer{1,2}`` and optional attention projections,
* a trainable adapter outside the frozen base path,
* bf16 baseline vs int8-stored frozen base Linear weights,
* output, loss, and trainable-adapter gradient-norm comparison.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn import functional as F

from library.runtime.int8_linear import replace_frozen_base_linears_with_int8


class ProbeBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(dim, dim, bias=False)
        self.self_attn.output_proj = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Module()
        self.mlp.layer1 = nn.Linear(dim, hidden_dim, bias=False)
        self.mlp.layer2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.self_attn.output_proj(torch.tanh(self.self_attn.q_proj(x)))
        hidden = self.mlp.layer1(x + attn)
        return self.mlp.layer2(F.gelu(hidden))


class ProbeModel(nn.Module):
    def __init__(self, *, dim: int, hidden_dim: int, num_blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [ProbeBlock(dim, hidden_dim) for _ in range(num_blocks)]
        )
        self.adapter = nn.Linear(dim, dim, bias=False)
        self.final_layer = nn.Linear(dim, dim, bias=False)

        for block in self.blocks:
            block.requires_grad_(False)
        self.final_layer.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_layer(hidden) + self.adapter(x)


class TinyAnimaProbe(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        hidden_dim: int,
        num_blocks: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        from library.anima.models import Anima

        num_heads = 2
        if dim % num_heads != 0:
            raise ValueError("tiny Anima dim must be divisible by 2 heads")
        head_dim = dim // num_heads
        dim_h = head_dim // 6 * 2
        dim_t = head_dim - 2 * dim_h
        if dim_h <= 2 or dim_t <= 2:
            raise ValueError("tiny Anima dim is too small for 3D RoPE; use dim>=24")
        self.anima = Anima(
            max_img_h=4,
            max_img_w=4,
            max_frames=1,
            in_channels=4,
            out_channels=4,
            patch_spatial=2,
            patch_temporal=1,
            concat_padding_mask=False,
            model_channels=dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            mlp_ratio=hidden_dim / dim,
            crossattn_emb_channels=dim,
            pos_emb_learnable=True,
            use_adaln_lora=True,
            adaln_lora_dim=max(4, dim // 3),
            use_llm_adapter=False,
            attn_mode="torch",
        )
        self.adapter = nn.Conv3d(4, 4, kernel_size=1, bias=False)
        self.anima.requires_grad_(False)
        self.to(dtype=dtype)

    @property
    def blocks(self) -> nn.ModuleList:
        return self.anima.blocks

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        return self.anima(x + self.adapter(x), timesteps, context)


def _dtype_from_name(name: str) -> torch.dtype:
    normalized = name.strip().lower()
    if normalized == "bf16":
        return torch.bfloat16
    if normalized == "fp16":
        return torch.float16
    if normalized == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _relative_delta(next_value: float, base_value: float) -> float:
    denom = abs(base_value)
    if denom <= 1e-12:
        return 0.0 if abs(next_value) <= 1e-12 else math.inf
    return abs(next_value - base_value) / denom


def _grad_norm(model: nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if not param.requires_grad or param.grad is None:
            continue
        grad = param.grad.detach().float()
        total += float(torch.sum(grad * grad).item())
    return math.sqrt(total)


def _run_step(
    model: nn.Module,
    model_inputs: tuple[torch.Tensor, ...],
    target: torch.Tensor,
    *,
    capture_blocks: bool = False,
) -> dict[str, Any]:
    captured: dict[str, torch.Tensor] = {}
    hooks = []
    if capture_blocks:
        blocks = getattr(model, "blocks", None)
        if blocks is not None:
            for idx, block in enumerate(blocks):
                hooks.append(
                    block.register_forward_hook(
                        lambda _module, _inputs, output, idx=idx: captured.__setitem__(
                            f"blocks.{idx}", output.detach().float()
                        )
                    )
                )
    model.zero_grad(set_to_none=True)
    try:
        output = model(*model_inputs)
        loss = F.mse_loss(output.float(), target.float())
        loss.backward()
        return {
            "output": output.detach().float(),
            "loss": float(loss.detach().item()),
            "grad_norm": _grad_norm(model),
            "block_outputs": captured,
        }
    finally:
        for hook in hooks:
            hook.remove()


def _tensor_rel_l2_and_cosine(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
) -> tuple[float, float]:
    diff = candidate - baseline
    output_norm = float(baseline.norm().item())
    rel_l2 = float(diff.norm().item()) / output_norm if output_norm > 0 else 0.0
    cosine = (
        float(F.cosine_similarity(baseline.reshape(1, -1), candidate.reshape(1, -1), dim=1).item())
        if baseline.numel()
        else 1.0
    )
    return rel_l2, cosine


def _block_output_deltas(
    baseline: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    out = {}
    for key, base in sorted(baseline.items()):
        if key not in candidate:
            continue
        rel_l2, cosine = _tensor_rel_l2_and_cosine(base, candidate[key])
        out[key] = {"relative_l2": rel_l2, "cosine": cosine}
    return out


def _finish_result(
    *,
    model_kind: str,
    scope: str,
    seed: int,
    dtype: torch.dtype,
    batch_size: int,
    dim: int,
    hidden_dim: int,
    num_blocks: int,
    replacements,
    baseline_step: dict[str, Any],
    int8_step: dict[str, Any],
    max_output_rel_l2: float,
    max_loss_rel_delta: float,
    max_grad_norm_rel_delta: float,
) -> dict[str, Any]:
    output_rel_l2, output_cosine = _tensor_rel_l2_and_cosine(
        baseline_step["output"],
        int8_step["output"],
    )
    block_output_deltas = _block_output_deltas(
        baseline_step.get("block_outputs", {}),
        int8_step.get("block_outputs", {}),
    )
    block_output_rel_l2_max = max(
        (item["relative_l2"] for item in block_output_deltas.values()),
        default=0.0,
    )
    loss_rel_delta = _relative_delta(int8_step["loss"], baseline_step["loss"])
    grad_norm_rel_delta = _relative_delta(
        int8_step["grad_norm"], baseline_step["grad_norm"]
    )
    payload_bytes = sum(item.payload_bytes for item in replacements)
    bf16_bytes = sum(item.bf16_bytes for item in replacements)
    gate_pass = (
        output_rel_l2 <= max_output_rel_l2
        and loss_rel_delta <= max_loss_rel_delta
        and grad_norm_rel_delta <= max_grad_norm_rel_delta
    )
    return {
        "model_kind": model_kind,
        "scope": scope,
        "seed": seed,
        "dtype": str(dtype).replace("torch.", ""),
        "batch_size": batch_size,
        "dim": dim,
        "hidden_dim": hidden_dim,
        "num_blocks": num_blocks,
        "replacement_count": len(replacements),
        "replacements": [item.__dict__ for item in replacements],
        "payload_bytes": payload_bytes,
        "bf16_bytes": bf16_bytes,
        "payload_ratio_vs_bf16": payload_bytes / bf16_bytes if bf16_bytes else 0.0,
        "baseline_loss": baseline_step["loss"],
        "int8_loss": int8_step["loss"],
        "loss_rel_delta": loss_rel_delta,
        "baseline_grad_norm": baseline_step["grad_norm"],
        "int8_grad_norm": int8_step["grad_norm"],
        "grad_norm_rel_delta": grad_norm_rel_delta,
        "output_rel_l2": output_rel_l2,
        "output_cosine": output_cosine,
        "block_output_rel_l2_max": block_output_rel_l2_max,
        "block_output_deltas": block_output_deltas,
        "thresholds": {
            "max_output_rel_l2": max_output_rel_l2,
            "max_loss_rel_delta": max_loss_rel_delta,
            "max_grad_norm_rel_delta": max_grad_norm_rel_delta,
        },
        "gate_pass": gate_pass,
    }


def run_probe(
    *,
    scope: str = "mlp",
    seed: int = 0,
    dtype: torch.dtype = torch.bfloat16,
    batch_size: int = 4,
    dim: int = 16,
    hidden_dim: int = 64,
    num_blocks: int = 2,
    max_output_rel_l2: float = 0.03,
    max_loss_rel_delta: float = 0.05,
    max_grad_norm_rel_delta: float = 0.05,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    baseline = ProbeModel(dim=dim, hidden_dim=hidden_dim, num_blocks=num_blocks).to(dtype=dtype)
    int8_model = copy.deepcopy(baseline)

    replacements = replace_frozen_base_linears_with_int8(int8_model, scope=scope)

    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    x = torch.randn(batch_size, dim, generator=generator, dtype=dtype)
    target = torch.randn(batch_size, dim, generator=generator, dtype=dtype)

    baseline_step = _run_step(baseline, (x,), target)
    int8_step = _run_step(int8_model, (x,), target)

    return _finish_result(
        model_kind="toy",
        scope=scope,
        seed=seed,
        dtype=dtype,
        batch_size=batch_size,
        dim=dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        replacements=replacements,
        baseline_step=baseline_step,
        int8_step=int8_step,
        max_output_rel_l2=max_output_rel_l2,
        max_loss_rel_delta=max_loss_rel_delta,
        max_grad_norm_rel_delta=max_grad_norm_rel_delta,
    )


def run_anima_probe(
    *,
    scope: str = "mlp",
    seed: int = 0,
    dtype: torch.dtype = torch.bfloat16,
    batch_size: int = 2,
    dim: int = 24,
    hidden_dim: int = 48,
    num_blocks: int = 2,
    max_output_rel_l2: float = 0.03,
    max_loss_rel_delta: float = 0.05,
    max_grad_norm_rel_delta: float = 0.05,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    baseline = TinyAnimaProbe(
        dim=dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dtype=dtype,
    )
    int8_model = copy.deepcopy(baseline)
    baseline.train()
    int8_model.train()

    replacements = replace_frozen_base_linears_with_int8(int8_model.anima, scope=scope)

    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    x = torch.randn(batch_size, 4, 1, 4, 4, generator=generator, dtype=dtype)
    context = torch.randn(batch_size, 5, dim, generator=generator, dtype=dtype)
    timesteps = torch.rand(batch_size, generator=generator, dtype=dtype)
    target = torch.randn(batch_size, 4, 1, 4, 4, generator=generator, dtype=dtype)

    model_inputs = (x, timesteps, context)
    baseline_step = _run_step(baseline, model_inputs, target, capture_blocks=True)
    int8_step = _run_step(int8_model, model_inputs, target, capture_blocks=True)

    return _finish_result(
        model_kind="anima",
        scope=scope,
        seed=seed,
        dtype=dtype,
        batch_size=batch_size,
        dim=dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        replacements=replacements,
        baseline_step=baseline_step,
        int8_step=int8_step,
        max_output_rel_l2=max_output_rel_l2,
        max_loss_rel_delta=max_loss_rel_delta,
        max_grad_norm_rel_delta=max_grad_norm_rel_delta,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", choices=("toy", "anima"), default="toy")
    parser.add_argument("--scope", default="mlp", help="mlp, attention, or all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--max-output-rel-l2", type=float, default=0.03)
    parser.add_argument("--max-loss-rel-delta", type=float, default=0.05)
    parser.add_argument("--max-grad-norm-rel-delta", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = run_anima_probe if args.model_kind == "anima" else run_probe
    dim = args.dim if args.dim is not None else (24 if args.model_kind == "anima" else 16)
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else (dim * 2 if args.model_kind == "anima" else 64)
    result = runner(
        scope=args.scope,
        seed=args.seed,
        dtype=_dtype_from_name(args.dtype),
        batch_size=args.batch_size,
        dim=dim,
        hidden_dim=hidden_dim,
        num_blocks=args.num_blocks,
        max_output_rel_l2=args.max_output_rel_l2,
        max_loss_rel_delta=args.max_loss_rel_delta,
        max_grad_norm_rel_delta=args.max_grad_norm_rel_delta,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
