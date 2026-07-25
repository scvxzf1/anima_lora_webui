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
    discover_cached_batch_pairs,
    select_cached_batch_pair,
)
import re


def _pick_cuda_device() -> torch.device:
    """Prefer a PyTorch-supported GPU (skip sm_52 GTX 960 display adapters).

    Dual-GPU boxes often expose the display card as ``cuda:0`` under
    ``CUDA_DEVICE_ORDER=PCI_BUS_ID``. Probes should still find the 3080-class
    card without requiring the caller to set ``CUDA_VISIBLE_DEVICES``.
    """
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    n = torch.cuda.device_count()
    best = None
    best_score = -1
    for i in range(n):
        try:
            major, minor = torch.cuda.get_device_capability(i)
        except Exception:
            continue
        # Current torch wheels need >= sm_75; 3080 is sm_86.
        if major < 7 or (major == 7 and minor < 5):
            continue
        props = torch.cuda.get_device_properties(i)
        # Score: prefer higher CC, then more memory.
        score = major * 100 + minor * 10 + int(props.total_memory / (1024**3))
        if score > best_score:
            best_score = score
            best = i
    if best is None:
        raise SystemExit(
            "no CUDA device with compute capability >= 7.5 "
            f"(saw {n} device(s); set CUDA_VISIBLE_DEVICES to a 3080-class GPU)"
        )
    name = torch.cuda.get_device_name(best)
    print(f"using cuda:{best} ({name})", flush=True)
    return torch.device(f"cuda:{best}")


def _select_cached_batch_pair(
    data_dir: Path,
    cache_index: int,
    *,
    min_latent_tokens: int = 0,
) -> object:
    """Pick a cache pair; optionally require latent spatial tokens ≥ N.

    ``min_latent_tokens`` filters by ``H*W`` of the cached latent (VAE latent
    grid, not pixel). Used for same-VRAM higher-resolution KPI sweeps without
    re-preprocessing.
    """
    if int(min_latent_tokens or 0) <= 0:
        return select_cached_batch_pair(data_dir, cache_index)
    import numpy as np

    pairs = discover_cached_batch_pairs(data_dir)
    if not pairs:
        raise FileNotFoundError(
            f"no matched *_anima.npz / *_anima_te.safetensors pairs under {data_dir}"
        )
    matched = []
    for pair in pairs:
        try:
            with np.load(pair.latent_path) as z:
                # Anima caches use keys like ``latents_144x112``, not bare ``latents``.
                arr = None
                for key in z.files:
                    if "latent" in key.lower() and getattr(z[key], "ndim", 0) >= 3:
                        arr = z[key]
                        break
                if arr is None:
                    for key in z.files:
                        if getattr(z[key], "ndim", 0) >= 3:
                            arr = z[key]
                            break
                if arr is None:
                    continue
                tokens = int(arr.shape[-2] * arr.shape[-1])
        except Exception:
            continue
        if tokens >= int(min_latent_tokens):
            matched.append((tokens, pair))
    if not matched:
        raise FileNotFoundError(
            f"no cache pair with latent H*W >= {min_latent_tokens} under {data_dir}"
        )
    matched.sort(key=lambda t: t[0])  # smallest eligible first (stable)
    idx = int(cache_index)
    if idx < 0 or idx >= len(matched):
        raise IndexError(
            f"cache index {idx} outside 0..{len(matched) - 1} "
            f"(min_latent_tokens={min_latent_tokens})"
        )
    return matched[idx][1]


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
    compile_mode: str | None = None,
    lora_rank: int = 4,
    lora_alpha: float | None = None,
    batch_size: int = 1,
    cache_index: int = 0,
    min_latent_tokens: int = 0,
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
    pair = _select_cached_batch_pair(
        data_dir,
        int(cache_index),
        min_latent_tokens=int(min_latent_tokens or 0),
    )
    model_inputs, target, batch_meta = _load_checkpoint_batch(
        pair,
        batch_size=bs,
        seed=seed,
        device=device,
        dtype=dtype,
        text_variant=0,
    )
    x, timesteps, context, padding_mask = model_inputs
    # latent tokens = H*W of DiT spatial grid (x is 5D B,C,T,H,W)
    latent_hw = int(x.shape[-2] * x.shape[-1])
    latent_shape = list(x.shape)

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
        mode = (compile_mode or "").strip() or None
        compile_blocks_for_training(
            anima,
            network,
            backend="inductor",
            mode=mode,
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
        "compile_mode": (compile_mode or None),
        "lora_rank": rank,
        "lora_alpha": alpha,
        "batch_size": bs,
        "cache_index": int(cache_index),
        "min_latent_tokens": int(min_latent_tokens or 0),
        "latent_hw": latent_hw,
        "latent_shape": latent_shape,
        "latent_path": str(getattr(pair, "latent_path", "")),
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
        "--compile-mode",
        type=str,
        default=None,
        help=(
            "Optional torch.compile mode passed to compile_blocks_for_training "
            "(e.g. default, reduce-overhead, max-autotune). Default None."
        ),
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
    parser.add_argument(
        "--cache-index",
        type=int,
        default=0,
        help="Index into discovered (or min-token-filtered) cache pairs.",
    )
    parser.add_argument(
        "--min-latent-tokens",
        type=int,
        default=0,
        help=(
            "If >0, only use cache pairs whose latent H*W >= this. "
            "Example: 896x1200 latent is often 56x75=4200; 1792x2400 ~112x150=16800."
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = _pick_cuda_device()
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
                compile_mode=args.compile_mode,
                lora_rank=int(args.lora_rank),
                lora_alpha=args.lora_alpha,
                batch_size=int(args.batch_size),
                cache_index=int(args.cache_index),
                min_latent_tokens=int(args.min_latent_tokens or 0),
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
