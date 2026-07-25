#!/usr/bin/env python
"""Memory / speed microbench for ConvRot free-base + int8 GEMM.

Compares:
  * bf16 LoRA baseline
  * W8A16 with base weight freed (default)
  * W8A8 with ANIMA_CONVROT_INT8_GEMM=auto|float

Uses full anima-preview3 checkpoint + one cached batch (same as checkpoint probe).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from library.runtime.convrot.apply import apply_convrot_to_lora_network
from library.runtime.int8_linear import selected_int8_linear_modules
from scripts.experiments.int8_linear_equivalence_probe import (
    DEFAULT_DATA_DIR,
    DEFAULT_DIT_PATH,
    _load_checkpoint_batch,
    select_cached_batch_pair,
)
import re


def _create_lora(anima, *, seed, device, dtype, rank, alpha, scope):
    from networks.lora_anima.factory import create_network

    target_modules = selected_int8_linear_modules(scope)
    target_pattern = "|".join(re.escape(name) for name in sorted(target_modules))
    torch.manual_seed(seed)
    network = create_network(
        1.0,
        rank,
        alpha,
        None,
        [],
        anima,
        exclude_patterns=[rf"^(?!blocks\.\d+\.({target_pattern})$).*"],
        train_llm_adapter="false",
        lora_fp32_compute="false",
    )
    network.apply_to([], anima, apply_text_encoder=False, apply_unet=True)
    network.to(device=device, dtype=dtype)
    network.train()
    return network


def _bytes_of_module_params_buffers(module) -> int:
    total = 0
    for p in module.parameters(recurse=True):
        if p.device.type == "meta":
            continue
        total += p.numel() * p.element_size()
    for b in module.buffers(recurse=True):
        if b is None or b.device.type == "meta":
            continue
        total += b.numel() * b.element_size()
    return total


def run_case(
    *,
    label: str,
    mode: str | None,
    free_base: bool,
    gemm_env: str | None,
    dit_path: Path,
    data_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    steps: int,
    scope: str,
    group_size: int,
    seed: int,
    weight_source: str = "online_from_bf16",
    prequant_path: str | None = None,
    min_in_features: int = 0,
    largest_in_features_only: bool = False,
    large_layer_mode: str | None = None,
    large_min_in_features: int | None = None,
    torch_compile: bool = False,
    lora_rank: int = 4,
    lora_alpha: float | None = None,
    batch_size: int = 1,
) -> dict:
    if gemm_env is not None:
        os.environ["ANIMA_CONVROT_INT8_GEMM"] = gemm_env
    elif "ANIMA_CONVROT_INT8_GEMM" in os.environ:
        del os.environ["ANIMA_CONVROT_INT8_GEMM"]

    from library.anima.weights import load_anima_model

    rank = int(lora_rank)
    alpha = float(lora_alpha) if lora_alpha is not None else float(rank)

    torch.cuda.empty_cache()
    # Ensure context exists before peak-stat APIs (some drivers reject bare device objects).
    torch.zeros(1, device=device)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    bs = max(1, int(batch_size))
    pair = select_cached_batch_pair(data_dir, 0)
    model_inputs, target, _meta = _load_checkpoint_batch(
        pair,
        batch_size=bs,
        seed=seed,
        device=device,
        dtype=dtype,
        text_variant=0,
    )
    x, timesteps, context, padding_mask = model_inputs

    anima = load_anima_model(
        device=device,
        dit_path=str(dit_path),
        attn_mode="torch",
        loading_device=device,
        dit_weight_dtype=dtype,
    )
    anima.to(device=device, dtype=dtype).requires_grad_(False)
    anima.reset_mod_guidance()
    anima.enable_gradient_checkpointing()
    network = _create_lora(
        anima,
        seed=seed + 101,
        device=device,
        dtype=dtype,
        rank=rank,
        alpha=alpha,
        scope=scope,
    )
    free_stats = {"freed_modules": 0, "freed_bytes": 0}
    apply_wall_sec = None
    if mode is not None:
        t_apply0 = time.perf_counter()
        result = apply_convrot_to_lora_network(
            network,
            mode=mode,  # type: ignore[arg-type]
            scope=scope,
            group_size=group_size,
            free_base_weights=free_base,
            weight_source=weight_source,
            prequant_path=prequant_path,
            unet=anima,
            min_in_features=min_in_features,
            largest_in_features_only=largest_in_features_only,
            large_layer_mode=large_layer_mode,
            large_min_in_features=large_min_in_features,
        )
        torch.cuda.synchronize()
        apply_wall_sec = time.perf_counter() - t_apply0
        free_stats = {
            "freed_modules": result.freed_modules,
            "freed_bytes": result.freed_bytes,
            "patched": result.patched_count,
            "weight_source": result.weight_source,
            "min_in_features": result.min_in_features,
            "largest_in_features_only": result.largest_in_features_only,
            "large_layer_mode": result.large_layer_mode,
            "large_min_in_features": result.large_min_in_features,
            "patch_modes": sorted({p.mode for p in result.patches}),
            "patch_names_sample": [p.name for p in result.patches[:8]],
        }

    # Match training order: compile AFTER ConvRot apply so dynamo traces
    # the patched org_forward (AGENTS.md Compile After Apply).
    if torch_compile:
        from library.runtime.harness import compile_blocks_for_training

        # Minimal training-like compile: inductor, grad-ckpt on (probe enables it).
        compile_blocks_for_training(
            anima,
            network,
            backend="inductor",
            grad_ckpt=True,
        )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    storage_after_apply = torch.cuda.memory_allocated()

    params = [p for p in network.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=5e-5)

    # warmup (extra steps when compile is on so inductor settles)
    warm = 4 if torch_compile else 2
    for _ in range(warm):
        optim.zero_grad(set_to_none=True)
        out = anima(x, timesteps, context, padding_mask=padding_mask)
        loss = F.mse_loss(out.float(), target.float())
        loss.backward()
        optim.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    last_loss = None
    for _ in range(steps):
        optim.zero_grad(set_to_none=True)
        out = anima(x, timesteps, context, padding_mask=padding_mask)
        loss = F.mse_loss(out.float(), target.float())
        loss.backward()
        optim.step()
        last_loss = float(loss.detach().item())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak = int(torch.cuda.max_memory_allocated())
    allocated = int(torch.cuda.memory_allocated())

    # count meta base weights
    meta_bases = 0
    for lora in network.unet_loras:
        refs = getattr(lora, "org_module_ref", None)
        if refs and getattr(refs[0], "weight", None) is not None:
            if refs[0].weight.device.type == "meta":
                meta_bases += 1

    payload = {
        "label": label,
        "mode": mode or "bf16",
        "free_base": free_base,
        "gemm_env": gemm_env,
        "weight_source": weight_source if mode is not None else None,
        "prequant_path": prequant_path,
        "min_in_features": min_in_features,
        "largest_in_features_only": largest_in_features_only,
        "large_layer_mode": large_layer_mode,
        "large_min_in_features": large_min_in_features,
        "torch_compile": bool(torch_compile),
        "lora_rank": rank,
        "lora_alpha": alpha,
        "batch_size": bs,
        "steps": steps,
        "elapsed_sec": elapsed,
        "sec_per_step": elapsed / steps,
        "apply_wall_sec": apply_wall_sec,
        "peak_bytes": peak,
        "peak_gb": peak / (1024**3),
        "allocated_after_steps_gb": allocated / (1024**3),
        "allocated_after_apply_gb": storage_after_apply / (1024**3),
        "last_loss": last_loss,
        "free_stats": free_stats,
        "meta_base_linears": meta_bases,
        "network_param_buffer_bytes": _bytes_of_module_params_buffers(network),
    }
    print(
        f"[{label}] peak={payload['peak_gb']:.2f}GB "
        f"alloc_after_apply={payload['allocated_after_apply_gb']:.2f}GB "
        f"sec/step={payload['sec_per_step']:.3f} "
        f"apply={0.0 if apply_wall_sec is None else apply_wall_sec:.3f}s "
        f"patched={free_stats.get('patched', 0)} "
        f"meta_bases={meta_bases} freed_mb={free_stats.get('freed_bytes',0)/1024**2:.1f} "
        f"loss={last_loss:.4f}",
        flush=True,
    )

    del anima, network, optim, params, model_inputs, target
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group-size", type=int, default=256)
    parser.add_argument("--scope", default="mlp")
    parser.add_argument("--dit-path", type=Path, default=DEFAULT_DIT_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("output/tests/convrot_mem_speed.json"),
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="all",
        help=(
            "Comma list: bf16,w8a16_free,w8a16_nofree,w8a8_auto,w8a8_float,"
            "w8a16_prequant,w8a8_prequant,"
            "w8a16_largest,w8a16_min4096,w8a16_mixed_large_w8a8 or all / p1"
        ),
    )
    parser.add_argument(
        "--prequant-path",
        type=Path,
        default=None,
        help="Required for *prequant cases (native anima_lora_convrot_prequant_v1).",
    )
    parser.add_argument(
        "--min-in-features",
        type=int,
        default=0,
        help="Global default for cases that do not set their own min_in_features.",
    )
    parser.add_argument(
        "--largest-in-features-only",
        action="store_true",
        help="Global default largest-only flag for generic w8a* cases.",
    )
    parser.add_argument(
        "--large-layer-mode",
        type=str,
        default=None,
        help="Global default large-layer mode override.",
    )
    parser.add_argument(
        "--large-min-in-features",
        type=int,
        default=None,
        help="Global default large-layer in_features threshold.",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Compile DiT blocks after ConvRot apply (mirrors training order).",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=4,
        help="LoRA rank (network_dim). Default 4 matches historical probes.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=None,
        help="LoRA alpha; default equals --lora-rank.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Microbatch size (repeats one cached sample). Default 1.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    prequant_path = str(args.prequant_path) if args.prequant_path is not None else None

    # Shared P1 defaults (overridden per named case below).
    p1_defaults = dict(
        min_in_features=int(args.min_in_features or 0),
        largest_in_features_only=bool(args.largest_in_features_only),
        large_layer_mode=args.large_layer_mode,
        large_min_in_features=args.large_min_in_features,
    )

    all_cases = {
        "bf16": dict(label="bf16", mode=None, free_base=False, gemm_env=None),
        "w8a16_free": dict(
            label="w8a16_free",
            mode="w8a16",
            free_base=True,
            gemm_env=None,
            **p1_defaults,
        ),
        "w8a16_nofree": dict(
            label="w8a16_nofree",
            mode="w8a16",
            free_base=False,
            gemm_env=None,
            **p1_defaults,
        ),
        "w8a8_auto": dict(
            label="w8a8_auto",
            mode="w8a8",
            free_base=True,
            gemm_env="auto",
            **p1_defaults,
        ),
        "w8a8_float": dict(
            label="w8a8_float",
            mode="w8a8",
            free_base=True,
            gemm_env="float",
            **p1_defaults,
        ),
        "w8a16_prequant": dict(
            label="w8a16_prequant",
            mode="w8a16",
            free_base=True,
            gemm_env=None,
            weight_source="prequant_checkpoint",
            prequant_path=prequant_path,
            **p1_defaults,
        ),
        "w8a8_prequant": dict(
            label="w8a8_prequant",
            mode="w8a8",
            free_base=True,
            gemm_env="auto",
            weight_source="prequant_checkpoint",
            prequant_path=prequant_path,
            **p1_defaults,
        ),
        # P1 named presets (Anima mlp: layer1 in=2048, layer2 in=8192).
        "w8a16_largest": dict(
            label="w8a16_largest",
            mode="w8a16",
            free_base=True,
            gemm_env=None,
            min_in_features=0,
            largest_in_features_only=True,
            large_layer_mode=None,
            large_min_in_features=None,
        ),
        "w8a16_min4096": dict(
            label="w8a16_min4096",
            mode="w8a16",
            free_base=True,
            gemm_env=None,
            min_in_features=4096,
            largest_in_features_only=False,
            large_layer_mode=None,
            large_min_in_features=None,
        ),
        "w8a16_mixed_large_w8a8": dict(
            label="w8a16_mixed_large_w8a8",
            mode="w8a16",
            free_base=True,
            gemm_env="auto",
            min_in_features=0,
            largest_in_features_only=False,
            large_layer_mode="w8a8",
            large_min_in_features=4096,
        ),
    }
    if args.cases.strip().lower() == "all":
        # Keep historical default set without requiring a prequant file.
        case_keys = [
            "bf16",
            "w8a16_free",
            "w8a16_nofree",
            "w8a8_auto",
            "w8a8_float",
        ]
    elif args.cases.strip().lower() == "p1":
        case_keys = [
            "bf16",
            "w8a16_free",
            "w8a16_largest",
            "w8a16_min4096",
            "w8a16_mixed_large_w8a8",
            "w8a8_auto",
        ]
    else:
        case_keys = [c.strip() for c in args.cases.split(",") if c.strip()]
    if any(k.endswith("_prequant") for k in case_keys) and not prequant_path:
        raise SystemExit("--prequant-path is required for *prequant cases")
    unknown = [k for k in case_keys if k not in all_cases]
    if unknown:
        raise SystemExit(f"unknown cases: {unknown}; known={sorted(all_cases)}")
    results = []
    import gc

    for key in case_keys:
        case = all_cases[key]
        print(f"=== {case['label']} ===", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        results.append(
            run_case(
                **case,
                dit_path=args.dit_path,
                data_dir=args.data_dir,
                device=device,
                dtype=dtype,
                steps=args.steps,
                scope=args.scope,
                group_size=args.group_size,
                seed=args.seed,
                torch_compile=bool(args.torch_compile),
                lora_rank=int(args.lora_rank),
                lora_alpha=args.lora_alpha,
                batch_size=int(args.batch_size),
            )
        )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    # merge with existing if partial
    payload = {"results": results}
    if args.json_out.exists() and args.cases.strip().lower() != "all":
        try:
            prev = json.loads(args.json_out.read_text())
            by_label = {r["label"]: r for r in prev.get("results", [])}
            for r in results:
                by_label[r["label"]] = r
            payload = {"results": list(by_label.values())}
        except Exception:
            pass
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
