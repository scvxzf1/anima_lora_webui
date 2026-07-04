#!/usr/bin/env python
"""Small-batch probe for int8 block-swap CPU masters.

This is a Phase-2 smoke test for the selective int8 route. It compares the
existing bf16 block-swap CPU master path with ``block_swap_transfer_dtype=int8``
on an Anima-shaped block surface. It does not run a real Anima checkpoint or a
full training job.
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

from library.runtime.offloading import ModelOffloader


class BlockSwapProbeBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.self_attn.output_proj = nn.Linear(dim, dim, bias=False)
        self.cross_attn = nn.Module()
        self.cross_attn.q_proj = nn.Linear(dim, dim, bias=False)
        self.cross_attn.kv_proj = nn.Linear(dim, dim * 2, bias=False)
        self.cross_attn.output_proj = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Module()
        self.mlp.layer1 = nn.Linear(dim, hidden_dim, bias=False)
        self.mlp.layer2 = nn.Linear(hidden_dim, dim, bias=False)
        self.adaln_up_mlp = nn.Linear(dim, dim, bias=False)
        self.adapter = nn.Linear(dim, dim, bias=False)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.requires_grad_(False)
        self.adapter.weight.requires_grad_(True)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q, k, v = self.self_attn.qkv_proj(x).chunk(3, dim=-1)
        self_attn = self.self_attn.output_proj(torch.tanh(q + k + v))

        cross_q = self.cross_attn.q_proj(x)
        cross_k, cross_v = self.cross_attn.kv_proj(context).chunk(2, dim=-1)
        cross = self.cross_attn.output_proj(torch.tanh(cross_q + cross_k + cross_v))

        gate = torch.sigmoid(self.adaln_up_mlp(context))
        hidden = self.mlp.layer1(x + self_attn + cross)
        base = self.mlp.layer2(F.gelu(hidden)) * gate
        return x + base + self.adapter(x)


class BlockSwapProbeModel(nn.Module):
    def __init__(self, *, dim: int, hidden_dim: int, num_blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [BlockSwapProbeBlock(dim, hidden_dim) for _ in range(num_blocks)]
        )
        self.final_layer = nn.Linear(dim, dim, bias=False)
        self.final_layer.weight.requires_grad_(False)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        *,
        offloader: ModelOffloader | None = None,
    ) -> torch.Tensor:
        hidden = x
        for block_idx, block in enumerate(self.blocks):
            if offloader is not None:
                offloader.wait_for_block(block_idx)
            hidden = block(hidden, context)
            if offloader is not None:
                offloader.submit_move_blocks(self.blocks, block_idx)
        return self.final_layer(hidden)


def _dtype_from_name(name: str) -> torch.dtype:
    normalized = name.strip().lower()
    if normalized == "bf16":
        return torch.bfloat16
    if normalized == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _device_from_name(name: str) -> torch.device:
    normalized = name.strip().lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda" or normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
        device = torch.device(normalized)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"--device {normalized} requested but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible"
            )
        return device
    raise ValueError(f"unsupported device: {name}")


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


def _tensor_rel_l2_and_cosine(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
) -> tuple[float, float]:
    diff = candidate - baseline
    output_norm = float(baseline.norm().item())
    rel_l2 = float(diff.norm().item()) / output_norm if output_norm > 0 else 0.0
    cosine = (
        float(
            F.cosine_similarity(
                baseline.reshape(1, -1),
                candidate.reshape(1, -1),
                dim=1,
            ).item()
        )
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


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * percentile) - 1))
    return float(ordered[idx])


def _profile_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    config = next((event for event in events if event.get("ev") == "block_swap_config"), None)
    waits = [event for event in events if event.get("ev") == "block_swap"]
    h2d_values = [float(event.get("h2d_ms", 0.0)) for event in waits]
    wait_values = [float(event.get("wait_ms", 0.0)) for event in waits]
    return {
        "path": str(path),
        "event_count": len(events),
        "wait_event_count": len(waits),
        "config": config,
        "h2d_ms_mean": sum(h2d_values) / len(h2d_values) if h2d_values else 0.0,
        "h2d_ms_p95": _percentile(h2d_values, 0.95),
        "h2d_ms_max": max(h2d_values, default=0.0),
        "wait_ms_mean": sum(wait_values) / len(wait_values) if wait_values else 0.0,
        "wait_ms_p95": _percentile(wait_values, 0.95),
        "wait_ms_max": max(wait_values, default=0.0),
    }


def _profile_path(profile_dir: Path | None, transfer_dtype: str) -> Path | None:
    if profile_dir is None:
        return None
    profile_dir.mkdir(parents=True, exist_ok=True)
    path = profile_dir / f"{transfer_dtype}_block_swap_profile.jsonl"
    if path.exists():
        raise FileExistsError(f"profile output already exists: {path}")
    return path


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-12:
        return 0.0 if abs(numerator) <= 1e-12 else math.inf
    return float(numerator) / float(denominator)


def _profile_ratios(
    baseline_profile: dict[str, Any] | None,
    int8_profile: dict[str, Any] | None,
) -> dict[str, float] | None:
    if baseline_profile is None or int8_profile is None:
        return None
    keys = (
        "h2d_ms_mean",
        "h2d_ms_p95",
        "h2d_ms_max",
        "wait_ms_mean",
        "wait_ms_p95",
        "wait_ms_max",
    )
    return {
        key: _safe_ratio(float(int8_profile.get(key, 0.0)), float(baseline_profile.get(key, 0.0)))
        for key in keys
    }


def _reset_peak_memory(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)


def _memory_summary(device: torch.device) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    return {
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _memory_ratios(
    baseline_memory: dict[str, int] | None,
    int8_memory: dict[str, int] | None,
) -> dict[str, float] | None:
    if baseline_memory is None or int8_memory is None:
        return None
    return {
        key: _safe_ratio(float(int8_memory.get(key, 0)), float(baseline_memory.get(key, 0)))
        for key in ("max_allocated_bytes", "max_reserved_bytes")
    }


def _run_offloaded_step(
    model: BlockSwapProbeModel,
    *,
    transfer_dtype: str,
    blocks_to_swap: int,
    repeat_steps: int,
    int8_restore_mode: str,
    int8_restore_chunk_rows: int,
    int8_scope: str,
    device: torch.device,
    profile_jsonl: Path | None,
    x: torch.Tensor,
    context: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, Any]:
    offloader = ModelOffloader(
        model.blocks,
        blocks_to_swap=blocks_to_swap,
        device=device,
        supports_backward=True,
        profile_jsonl=str(profile_jsonl) if profile_jsonl is not None else None,
        transfer_dtype=transfer_dtype,
        int8_restore_mode=int8_restore_mode,
        int8_restore_chunk_rows=int8_restore_chunk_rows,
        int8_scope=int8_scope,
    )
    captured: dict[str, torch.Tensor] = {}
    hooks = [
        block.register_forward_hook(
            lambda _module, _inputs, output, idx=idx: captured.__setitem__(
                f"blocks.{idx}",
                output.detach().float(),
            )
        )
        for idx, block in enumerate(model.blocks)
    ]
    if repeat_steps <= 0:
        raise ValueError("repeat_steps must be positive")
    model.zero_grad(set_to_none=True)
    try:
        _reset_peak_memory(device)
        output = None
        loss = None
        for _ in range(repeat_steps):
            model.zero_grad(set_to_none=True)
            offloader.prepare_block_devices_before_forward(model.blocks, free_cache=False)
            output = model(x, context, offloader=offloader)
            loss = F.mse_loss(output.float(), target.float())
            loss.backward()
        offloader.set_forward_only(True)
        offloader.restore_blocks_to_device(model.blocks, device)
        offloader.flush_profile_events(blocking=True)
        assert output is not None
        assert loss is not None
        return {
            "output": output.detach().float(),
            "loss": float(loss.detach().item()),
            "grad_norm": _grad_norm(model),
            "block_outputs": captured,
            "profile": _profile_summary(profile_jsonl),
            "memory": _memory_summary(device),
            "frozen_weight_master_bytes": offloader._frozen_weight_master_bytes,
            "bf16_master_bytes": offloader._bf16_master_bytes,
            "fp8_master_bytes": offloader._fp8_master_bytes,
            "int8_master_bytes": offloader._int8_master_bytes,
            "int8_quantized_tensors": offloader._int8_quantized_tensors,
            "int8_weight_bytes_by_block": list(offloader._int8_weight_bytes_by_block),
            "int8_relative_l2_by_block": list(offloader._int8_relative_l2_by_block),
        }
    finally:
        for hook in hooks:
            hook.remove()
        offloader.thread_pool.shutdown(wait=False)


def run_probe(
    *,
    seed: int = 0,
    dtype: torch.dtype = torch.bfloat16,
    batch_size: int = 4,
    dim: int = 16,
    hidden_dim: int = 64,
    num_blocks: int = 4,
    blocks_to_swap: int = 2,
    repeat_steps: int = 1,
    int8_restore_mode: str = "copy",
    int8_restore_chunk_rows: int = 0,
    int8_scope: str = "all",
    device: torch.device | None = None,
    profile_dir: Path | None = None,
    max_output_rel_l2: float = 0.03,
    max_loss_rel_delta: float = 0.05,
    max_grad_norm_rel_delta: float = 0.05,
) -> dict[str, Any]:
    if not 0 < blocks_to_swap < num_blocks:
        raise ValueError("blocks_to_swap must be between 1 and num_blocks - 1")
    if repeat_steps <= 0:
        raise ValueError("repeat_steps must be positive")

    device = device or torch.device("cpu")
    torch.manual_seed(seed)
    baseline = BlockSwapProbeModel(
        dim=dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
    ).to(device=device, dtype=dtype)
    int8_model = copy.deepcopy(baseline)

    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    x = torch.randn(batch_size, dim, generator=generator, dtype=dtype).to(device)
    context = torch.randn(batch_size, dim, generator=generator, dtype=dtype).to(device)
    target = torch.randn(batch_size, dim, generator=generator, dtype=dtype).to(device)
    bf16_profile_jsonl = _profile_path(profile_dir, "bf16")
    int8_profile_jsonl = _profile_path(profile_dir, "int8")

    baseline_step = _run_offloaded_step(
        baseline,
        transfer_dtype="bf16",
        blocks_to_swap=blocks_to_swap,
        repeat_steps=repeat_steps,
        int8_restore_mode="copy",
        int8_restore_chunk_rows=0,
        int8_scope="all",
        device=device,
        profile_jsonl=bf16_profile_jsonl,
        x=x,
        context=context,
        target=target,
    )
    int8_step = _run_offloaded_step(
        int8_model,
        transfer_dtype="int8",
        blocks_to_swap=blocks_to_swap,
        repeat_steps=repeat_steps,
        int8_restore_mode=int8_restore_mode,
        int8_restore_chunk_rows=int8_restore_chunk_rows,
        int8_scope=int8_scope,
        device=device,
        profile_jsonl=int8_profile_jsonl,
        x=x,
        context=context,
        target=target,
    )

    output_rel_l2, output_cosine = _tensor_rel_l2_and_cosine(
        baseline_step["output"],
        int8_step["output"],
    )
    block_output_deltas = _block_output_deltas(
        baseline_step["block_outputs"],
        int8_step["block_outputs"],
    )
    block_output_rel_l2_max = max(
        (item["relative_l2"] for item in block_output_deltas.values()),
        default=0.0,
    )
    loss_rel_delta = _relative_delta(int8_step["loss"], baseline_step["loss"])
    grad_norm_rel_delta = _relative_delta(
        int8_step["grad_norm"],
        baseline_step["grad_norm"],
    )
    gate_pass = (
        output_rel_l2 <= max_output_rel_l2
        and loss_rel_delta <= max_loss_rel_delta
        and grad_norm_rel_delta <= max_grad_norm_rel_delta
    )
    baseline_profile = baseline_step["profile"]
    int8_profile = int8_step["profile"]
    baseline_memory = baseline_step["memory"]
    int8_memory = int8_step["memory"]
    return {
        "model_kind": "blockswap_toy",
        "seed": seed,
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "batch_size": batch_size,
        "dim": dim,
        "hidden_dim": hidden_dim,
        "num_blocks": num_blocks,
        "blocks_to_swap": blocks_to_swap,
        "repeat_steps": repeat_steps,
        "baseline_transfer_dtype": "bf16",
        "candidate_transfer_dtype": "int8",
        "candidate_int8_restore_mode": int8_restore_mode,
        "candidate_int8_restore_chunk_rows": int8_restore_chunk_rows,
        "candidate_int8_scope": int8_scope,
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
        "bf16_master_bytes": int8_step["bf16_master_bytes"],
        "int8_master_bytes": int8_step["int8_master_bytes"],
        "int8_master_ratio_vs_bf16": (
            int8_step["int8_master_bytes"] / int8_step["bf16_master_bytes"]
            if int8_step["bf16_master_bytes"]
            else 0.0
        ),
        "int8_quantized_tensors": int8_step["int8_quantized_tensors"],
        "int8_weight_bytes_by_block": int8_step["int8_weight_bytes_by_block"],
        "int8_relative_l2_by_block": int8_step["int8_relative_l2_by_block"],
        "baseline_profile": baseline_profile,
        "int8_profile": int8_profile,
        "profile_ratios": _profile_ratios(baseline_profile, int8_profile),
        "baseline_memory": baseline_memory,
        "int8_memory": int8_memory,
        "memory_ratios": _memory_ratios(baseline_memory, int8_memory),
        "thresholds": {
            "max_output_rel_l2": max_output_rel_l2,
            "max_loss_rel_delta": max_loss_rel_delta,
            "max_grad_norm_rel_delta": max_grad_norm_rel_delta,
        },
        "gate_pass": gate_pass,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:<index>")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--blocks-to-swap", type=int, default=2)
    parser.add_argument("--repeat-steps", type=int, default=1)
    parser.add_argument(
        "--int8-restore-mode",
        choices=["copy", "direct_bind", "reuse_storage"],
        default="copy",
    )
    parser.add_argument("--int8-restore-chunk-rows", type=int, default=0)
    parser.add_argument(
        "--int8-scope",
        default="all",
        help=(
            "Comma-separated int8 CPU master scope for the candidate path, "
            "for example all, mlp, or mlp,cross_attn_q."
        ),
    )
    parser.add_argument("--max-output-rel-l2", type=float, default=0.03)
    parser.add_argument("--max-loss-rel-delta", type=float, default=0.05)
    parser.add_argument("--max-grad-norm-rel-delta", type=float, default=0.05)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=None,
        help="Optional directory for bf16/int8 block-swap JSONL profile outputs.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    result = run_probe(
        seed=args.seed,
        dtype=_dtype_from_name(args.dtype),
        device=_device_from_name(args.device),
        batch_size=args.batch_size,
        dim=args.dim,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        blocks_to_swap=args.blocks_to_swap,
        repeat_steps=args.repeat_steps,
        int8_restore_mode=args.int8_restore_mode,
        int8_restore_chunk_rows=args.int8_restore_chunk_rows,
        int8_scope=args.int8_scope,
        profile_dir=args.profile_dir,
        max_output_rel_l2=args.max_output_rel_l2,
        max_loss_rel_delta=args.max_loss_rel_delta,
        max_grad_norm_rel_delta=args.max_grad_norm_rel_delta,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
