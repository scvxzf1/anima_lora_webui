#!/usr/bin/env python
"""Full-checkpoint ConvRot equivalence probe (bf16 LoRA vs W8A* ConvRot base).

Loads ``anima-preview3-base`` + real LoRA on mlp scope, compares one training
step (output / loss / adapter grad) with and without ConvRot org_forward patch.

Examples::

    .venv/bin/python scripts/experiments/convrot_checkpoint_probe.py \\
        --seeds 0,1,2 --mode w8a16 --scope mlp

    .venv/bin/python scripts/experiments/convrot_checkpoint_probe.py \\
        --seeds 0 --mode w8a8 --json-out /tmp/convrot_ckpt.json
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn import functional as F

from library.runtime.convrot.apply import apply_convrot_to_lora_network
from library.runtime.int8_linear import selected_int8_linear_modules

# Reuse cache discovery helpers from the int8 probe.
from scripts.experiments.int8_linear_equivalence_probe import (
    DEFAULT_DATA_DIR,
    DEFAULT_DIT_PATH,
    _load_checkpoint_batch,
    _relative_delta,
    _tensor_rel_l2_and_cosine,
    select_cached_batch_pair,
)


def _dtype_from_name(name: str) -> torch.dtype:
    normalized = name.strip().lower()
    if normalized == "bf16":
        return torch.bfloat16
    if normalized == "fp16":
        return torch.float16
    if normalized == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _resolve_device(name: str | None) -> torch.device:
    if name is None:
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {name}")
    return device


def _cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def _grad_norm(params) -> float:
    total = 0.0
    for param in params:
        if not param.requires_grad or param.grad is None:
            continue
        grad = param.grad.detach().float()
        total += float(torch.sum(grad * grad).item())
    return math.sqrt(total)


def _create_lora_network(
    anima: nn.Module,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    network_dim: int,
    network_alpha: float,
    target_scope: str,
) -> nn.Module:
    from networks.lora_anima.factory import create_network

    target_modules = selected_int8_linear_modules(target_scope)
    target_pattern = "|".join(re.escape(name) for name in sorted(target_modules))
    torch.manual_seed(seed)
    network = create_network(
        1.0,
        network_dim,
        network_alpha,
        None,
        [],
        anima,
        exclude_patterns=[
            rf"^(?!blocks\.\d+\.({target_pattern})$).*",
        ],
        train_llm_adapter="false",
        lora_fp32_compute="false",
    )
    network.apply_to([], anima, apply_text_encoder=False, apply_unet=True)
    network.to(device=device, dtype=dtype)
    network.train()
    return network


def _load_model_and_network(
    *,
    dit_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    attn_mode: str,
    adapter_seed: int,
    lora_rank: int,
    lora_alpha: float,
    lora_target_scope: str,
    gradient_checkpointing: bool,
) -> tuple[nn.Module, nn.Module]:
    from library.anima.weights import load_anima_model

    anima = load_anima_model(
        device=device,
        dit_path=str(dit_path),
        attn_mode=attn_mode,
        loading_device=device,
        dit_weight_dtype=dtype,
    )
    anima.to(device=device, dtype=dtype).requires_grad_(False)
    anima.reset_mod_guidance()
    if gradient_checkpointing:
        anima.enable_gradient_checkpointing()
    network = _create_lora_network(
        anima,
        seed=adapter_seed,
        device=device,
        dtype=dtype,
        network_dim=lora_rank,
        network_alpha=lora_alpha,
        target_scope=lora_target_scope,
    )
    return anima, network


def _run_one(
    *,
    dit_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    attn_mode: str,
    adapter_seed: int,
    lora_rank: int,
    lora_alpha: float,
    lora_target_scope: str,
    gradient_checkpointing: bool,
    model_inputs: tuple[torch.Tensor, ...],
    target: torch.Tensor,
    mode: str | None,
    scope: str,
    group_size: int,
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    anima, network = _load_model_and_network(
        dit_path=dit_path,
        device=device,
        dtype=dtype,
        attn_mode=attn_mode,
        adapter_seed=adapter_seed,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_target_scope=lora_target_scope,
        gradient_checkpointing=gradient_checkpointing,
    )
    patch_info: dict[str, Any] = {"patched": 0, "skipped": 0}
    try:
        if mode is not None:
            result = apply_convrot_to_lora_network(
                network,
                mode=mode,  # type: ignore[arg-type]
                scope=scope,
                group_size=group_size,
                unet=anima,
            )
            patch_info = {
                "patched": result.patched_count,
                "skipped": result.skipped_count,
                "mode": result.mode,
                "group_size": result.group_size,
            }
        network.zero_grad(set_to_none=True)
        x, timesteps, context, padding_mask = model_inputs
        output = anima(
            x,
            timesteps,
            context,
            padding_mask=padding_mask,
        )
        loss = F.mse_loss(output.float(), target.float())
        loss.backward()
        grad_norm = _grad_norm(network.parameters())
        peak = 0
        if device.type == "cuda":
            peak = int(torch.cuda.max_memory_allocated(device))
        return {
            "output": output.detach().float().cpu(),
            "loss": float(loss.detach().item()),
            "grad_norm": grad_norm,
            "patch": patch_info,
            "peak_bytes": peak,
        }
    finally:
        del anima, network
        _cleanup(device)


def run_checkpoint_probe(
    *,
    mode: str = "w8a16",
    scope: str = "mlp",
    group_size: int = 256,
    seed: int = 0,
    dtype: torch.dtype = torch.bfloat16,
    batch_size: int = 1,
    dit_path: Path = DEFAULT_DIT_PATH,
    data_dir: Path = DEFAULT_DATA_DIR,
    cache_index: int = 0,
    text_variant: int | str = 0,
    device: str | None = None,
    attn_mode: str = "torch",
    lora_rank: int = 4,
    lora_alpha: float = 4.0,
    gradient_checkpointing: bool = True,
    max_output_rel_l2: float = 0.03,
    max_loss_rel_delta: float = 0.05,
    max_grad_norm_rel_delta: float = 0.05,
) -> dict[str, Any]:
    device_obj = _resolve_device(device)
    dit_path = Path(dit_path)
    data_dir = Path(data_dir)
    if not dit_path.exists():
        raise FileNotFoundError(f"DiT checkpoint not found: {dit_path}")

    pair = select_cached_batch_pair(data_dir, cache_index)
    model_inputs, target, batch_metadata = _load_checkpoint_batch(
        pair,
        batch_size=batch_size,
        seed=seed,
        device=device_obj,
        dtype=dtype,
        text_variant=text_variant,
    )
    # Keep CPU copies so each _run_one can re-place tensors after cleanup.
    model_inputs_cpu = tuple(t.detach().cpu() for t in model_inputs)
    target_cpu = target.detach().cpu()
    del model_inputs, target
    _cleanup(device_obj)

    adapter_seed = seed + 101
    baseline = _run_one(
        dit_path=dit_path,
        device=device_obj,
        dtype=dtype,
        attn_mode=attn_mode,
        adapter_seed=adapter_seed,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_target_scope=scope,
        gradient_checkpointing=gradient_checkpointing,
        model_inputs=tuple(t.to(device_obj) for t in model_inputs_cpu),
        target=target_cpu.to(device_obj),
        mode=None,
        scope=scope,
        group_size=group_size,
    )
    convrot = _run_one(
        dit_path=dit_path,
        device=device_obj,
        dtype=dtype,
        attn_mode=attn_mode,
        adapter_seed=adapter_seed,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_target_scope=scope,
        gradient_checkpointing=gradient_checkpointing,
        model_inputs=tuple(t.to(device_obj) for t in model_inputs_cpu),
        target=target_cpu.to(device_obj),
        mode=mode,
        scope=scope,
        group_size=group_size,
    )

    if baseline["output"].shape != convrot["output"].shape:
        raise RuntimeError(
            f"shape mismatch baseline={tuple(baseline['output'].shape)} "
            f"convrot={tuple(convrot['output'].shape)}"
        )

    out_rel, out_cos = _tensor_rel_l2_and_cosine(baseline["output"], convrot["output"])
    loss_rel = _relative_delta(convrot["loss"], baseline["loss"])
    grad_rel = _relative_delta(convrot["grad_norm"], baseline["grad_norm"])
    gate_pass = (
        out_rel <= max_output_rel_l2
        and loss_rel <= max_loss_rel_delta
        and grad_rel <= max_grad_norm_rel_delta
    )
    return {
        "model_kind": "checkpoint",
        "mode": mode,
        "scope": scope,
        "group_size": group_size,
        "seed": seed,
        "dtype": str(dtype).replace("torch.", ""),
        "batch_size": batch_size,
        "dit_path": str(dit_path),
        "data_dir": str(data_dir),
        "cache_index": cache_index,
        "device": str(device_obj),
        "attn_mode": attn_mode,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "gradient_checkpointing": gradient_checkpointing,
        "batch": batch_metadata,
        "patch": convrot["patch"],
        "baseline_loss": baseline["loss"],
        "convrot_loss": convrot["loss"],
        "loss_rel_delta": loss_rel,
        "baseline_grad_norm": baseline["grad_norm"],
        "convrot_grad_norm": convrot["grad_norm"],
        "grad_norm_rel_delta": grad_rel,
        "output_rel_l2": out_rel,
        "output_cosine": out_cos,
        "cuda_peak_bytes": {
            "baseline": baseline["peak_bytes"],
            "convrot": convrot["peak_bytes"],
        },
        "thresholds": {
            "max_output_rel_l2": max_output_rel_l2,
            "max_loss_rel_delta": max_loss_rel_delta,
            "max_grad_norm_rel_delta": max_grad_norm_rel_delta,
        },
        "gate_pass": gate_pass,
    }


def _metric_summary(values: list[float]) -> dict[str, float]:
    clean = sorted(values)
    return {
        "min": clean[0],
        "p50": clean[len(clean) // 2],
        "max": clean[-1],
        "mean": sum(clean) / len(clean),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["w8a16", "w8a8"], default="w8a16")
    parser.add_argument("--scope", default="mlp")
    parser.add_argument("--group-size", type=int, default=256)
    parser.add_argument(
        "--hadamard",
        choices=["sylvester", "regular"],
        default=None,
        help=(
            "Hadamard construction (sets ANIMA_CONVROT_HADAMARD). "
            "Default: current env or sylvester."
        ),
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dit-path", type=Path, default=DEFAULT_DIT_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-index", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--attn-mode", default="torch")
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=4.0)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--max-output-rel-l2", type=float, default=0.03)
    parser.add_argument("--max-loss-rel-delta", type=float, default=0.05)
    parser.add_argument("--max-grad-norm-rel-delta", type=float, default=0.05)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if args.hadamard is not None:
        os.environ["ANIMA_CONVROT_HADAMARD"] = args.hadamard
        # Force dense backend for regular (fwht is sylvester-only).
        if args.hadamard == "regular":
            os.environ["ANIMA_CONVROT_RHT"] = "dense"

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    dtype = _dtype_from_name(args.dtype)
    results = []
    for seed in seeds:
        print(
            f"[convrot-ckpt] running seed={seed} mode={args.mode} "
            f"group={args.group_size} hadamard={args.hadamard or 'env'} ...",
            flush=True,
        )
        result = run_checkpoint_probe(
            mode=args.mode,
            scope=args.scope,
            group_size=args.group_size,
            seed=seed,
            dtype=dtype,
            batch_size=args.batch_size,
            dit_path=args.dit_path,
            data_dir=args.data_dir,
            cache_index=args.cache_index,
            device=args.device,
            attn_mode=args.attn_mode,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            gradient_checkpointing=not args.no_gradient_checkpointing,
            max_output_rel_l2=args.max_output_rel_l2,
            max_loss_rel_delta=args.max_loss_rel_delta,
            max_grad_norm_rel_delta=args.max_grad_norm_rel_delta,
        )
        result["hadamard"] = args.hadamard or os.environ.get(
            "ANIMA_CONVROT_HADAMARD", "sylvester"
        )
        results.append(result)
        print(
            f"  seed={seed} gate={'PASS' if result['gate_pass'] else 'FAIL'} "
            f"out_rel={result['output_rel_l2']:.4f} "
            f"loss_rel={result['loss_rel_delta']:.4f} "
            f"grad_rel={result['grad_norm_rel_delta']:.4f} "
            f"patched={result['patch'].get('patched')} "
            f"peak_gb={result['cuda_peak_bytes']['convrot'] / (1024**3):.2f}",
            flush=True,
        )

    summary = {
        "mode": args.mode,
        "scope": args.scope,
        "group_size": args.group_size,
        "hadamard": args.hadamard
        or os.environ.get("ANIMA_CONVROT_HADAMARD", "sylvester"),
        "seeds": seeds,
        "gate_pass_all": all(r["gate_pass"] for r in results),
        "gate_pass_count": sum(1 for r in results if r["gate_pass"]),
        "run_count": len(results),
        "output_rel_l2": _metric_summary([r["output_rel_l2"] for r in results]),
        "loss_rel_delta": _metric_summary([r["loss_rel_delta"] for r in results]),
        "grad_norm_rel_delta": _metric_summary(
            [r["grad_norm_rel_delta"] for r in results]
        ),
        "patched": results[0]["patch"].get("patched") if results else 0,
        "results": results,
    }
    print(
        f"[convrot-ckpt] summary gate_pass_all={summary['gate_pass_all']} "
        f"out_max={summary['output_rel_l2']['max']:.4f} "
        f"grad_max={summary['grad_norm_rel_delta']['max']:.4f} "
        f"patched={summary['patched']}",
        flush=True,
    )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        # Drop bulky tensors already converted; results are JSON-safe.
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[convrot-ckpt] wrote {args.json_out}", flush=True)

    return 0 if summary["gate_pass_all"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
