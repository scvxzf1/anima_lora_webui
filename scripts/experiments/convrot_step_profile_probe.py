#!/usr/bin/env python
"""P0-A: single-step CUDA profiler for ConvRot vs bf16 LoRA.

Uses the same full DiT + LoRA + cached batch setup as ``convrot_mem_speed_probe``.
Warms up, then captures one train step under ``torch.profiler`` and aggregates
CUDA self-time into coarse buckets for a bottleneck pie table.

Examples::

    .venv/bin/python scripts/experiments/convrot_step_profile_probe.py \\
        --cases bf16,w8a16_free,w8a8_auto \\
        --json-out output/tests/convrot_step_profile.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F
from torch.profiler import ProfilerActivity, profile, record_function

from library.runtime.convrot.apply import apply_convrot_to_lora_network
from library.runtime.int8_linear import selected_int8_linear_modules
from scripts.experiments.int8_linear_equivalence_probe import (
    DEFAULT_DATA_DIR,
    DEFAULT_DIT_PATH,
    _load_checkpoint_batch,
    select_cached_batch_pair,
)


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


def _bucket_for_event(name: str) -> str:
    n = name.lower()
    # Explicit markers first (injected via record_function).
    if "convrot::rht" in n or "convrot::fwht" in n or "group_fwht" in n:
        return "convrot_rht"
    if "convrot::act_quant" in n or "quantize_activation" in n:
        return "convrot_act_quant"
    if "convrot::dequant" in n or "dequantize_weight" in n:
        return "convrot_dequant"
    if "convrot::gemm" in n or "convrot::w8a" in n or "int8_mm" in n or "_int_mm" in n:
        return "convrot_gemm"
    if "convrot::" in n:
        return "convrot_other"

    # Generic ATen / cuBLAS / attention.
    if any(
        k in n
        for k in (
            "aten::mm",
            "aten::addmm",
            "aten::bmm",
            "aten::linear",
            "cublas",
            "gemm",
            "cutlass",
            "nvjet",
        )
    ):
        return "gemm_generic"
    if any(
        k in n
        for k in (
            "scaled_dot_product",
            "sdpa",
            "attention",
            "flash",
            "xformers",
            "sageattn",
        )
    ):
        return "attention"
    if any(k in n for k in ("layer_norm", "rmsnorm", "group_norm", "softmax")):
        return "norm_act"
    if any(k in n for k in ("adam", "optimizer", "foreach", "fused_adam")):
        return "optimizer"
    if any(k in n for k in ("copy", "to(", "cast", "convert", "clone", "contiguous")):
        return "memcpy_cast"
    if "backward" in n or "autograd" in n or "accumulate_grad" in n:
        return "autograd_misc"
    return "other"


def _aggregate_profiler(prof: profile) -> dict:
    """Aggregate CUDA self-time (us) by coarse bucket."""
    bucket_us: dict[str, float] = defaultdict(float)
    top_events: list[dict] = []
    try:
        events = prof.key_averages()
    except Exception:
        return {"buckets_us": {}, "buckets_pct": {}, "top_events": [], "total_cuda_us": 0.0}

    for evt in events:
        # Prefer CUDA self time; fall back to CPU self for host-side markers.
        cuda_us = float(getattr(evt, "self_device_time_total", 0) or 0)
        cpu_us = float(getattr(evt, "self_cpu_time_total", 0) or 0)
        # PyTorch version naming: self_cuda_time_total vs self_device_time_total
        if cuda_us <= 0:
            cuda_us = float(getattr(evt, "self_cuda_time_total", 0) or 0)
        use_us = cuda_us if cuda_us > 0 else 0.0
        name = str(evt.key)
        bucket = _bucket_for_event(name)
        if use_us > 0:
            bucket_us[bucket] += use_us
        top_events.append(
            {
                "name": name,
                "bucket": bucket,
                "cuda_self_us": cuda_us,
                "cpu_self_us": cpu_us,
                "count": int(getattr(evt, "count", 0) or 0),
            }
        )

    top_events.sort(key=lambda e: e["cuda_self_us"], reverse=True)
    total = sum(bucket_us.values())
    pct = {
        k: (100.0 * v / total if total > 0 else 0.0)
        for k, v in sorted(bucket_us.items(), key=lambda kv: -kv[1])
    }
    return {
        "buckets_us": dict(sorted(bucket_us.items(), key=lambda kv: -kv[1])),
        "buckets_pct": pct,
        "top_events": top_events[:40],
        "total_cuda_us": total,
    }


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
    scope: str,
    group_size: int,
    seed: int,
    warmup: int,
    export_chrome: Path | None,
) -> dict:
    if gemm_env is not None:
        os.environ["ANIMA_CONVROT_INT8_GEMM"] = gemm_env
    elif "ANIMA_CONVROT_INT8_GEMM" in os.environ:
        del os.environ["ANIMA_CONVROT_INT8_GEMM"]

    from library.anima.weights import load_anima_model

    torch.cuda.empty_cache()
    torch.zeros(1, device=device)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    pair = select_cached_batch_pair(data_dir, 0)
    model_inputs, target, _meta = _load_checkpoint_batch(
        pair,
        batch_size=1,
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
        rank=4,
        alpha=4.0,
        scope=scope,
    )
    free_stats = {"freed_modules": 0, "freed_bytes": 0}
    if mode is not None:
        result = apply_convrot_to_lora_network(
            network,
            mode=mode,  # type: ignore[arg-type]
            scope=scope,
            group_size=group_size,
            free_base_weights=free_base,
            unet=anima,
        )
        free_stats = {
            "freed_modules": result.freed_modules,
            "freed_bytes": result.freed_bytes,
            "patched": result.patched_count,
        }

    params = [p for p in network.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=5e-5)

    def _one_step() -> float:
        optim.zero_grad(set_to_none=True)
        with record_function("step::forward"):
            out = anima(x, timesteps, context, padding_mask=padding_mask)
            loss = F.mse_loss(out.float(), target.float())
        with record_function("step::backward"):
            loss.backward()
        with record_function("step::optimizer"):
            optim.step()
        return float(loss.detach().item())

    for _ in range(max(1, warmup)):
        _one_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        with record_function("step::train"):
            t0 = time.perf_counter()
            last_loss = _one_step()
            torch.cuda.synchronize()
            wall_sec = time.perf_counter() - t0
        prof.step()

    agg = _aggregate_profiler(prof)
    peak = int(torch.cuda.max_memory_allocated())

    if export_chrome is not None:
        export_chrome.parent.mkdir(parents=True, exist_ok=True)
        try:
            prof.export_chrome_trace(str(export_chrome))
        except Exception as exc:  # pragma: no cover
            print(f"[{label}] chrome export failed: {exc}", flush=True)

    # Wall-clock multi-step microbench (profiler-off) for comparable sec/step.
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(3):
        _one_step()
    torch.cuda.synchronize()
    sec_per_step = (time.perf_counter() - t1) / 3.0

    payload = {
        "label": label,
        "mode": mode or "bf16",
        "free_base": free_base,
        "gemm_env": gemm_env,
        "rht_env": os.environ.get("ANIMA_CONVROT_RHT", "dense"),
        "w8a16_kernel_env": os.environ.get("ANIMA_CONVROT_W8A16_KERNEL", "auto"),
        "fused_env": os.environ.get("ANIMA_CONVROT_FUSED", "1"),
        "profiled_step_wall_sec": wall_sec,
        "sec_per_step_no_profiler": sec_per_step,
        "last_loss": last_loss,
        "peak_gb": peak / (1024**3),
        "free_stats": free_stats,
        "profile": agg,
    }
    print(
        f"[{label}] wall_step={wall_sec:.3f}s sec/step={sec_per_step:.3f} "
        f"peak={payload['peak_gb']:.2f}GB cuda_us={agg['total_cuda_us']:.0f} "
        f"top_buckets={list(agg['buckets_pct'].items())[:5]}",
        flush=True,
    )

    del anima, network, optim, params, model_inputs, target, prof
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return payload


def _decision_from_results(results: list[dict]) -> dict:
    """Heuristic branch decision for the optimization roadmap."""
    by = {r["label"]: r for r in results}
    w8 = by.get("w8a16_free") or by.get("w8a8_auto")
    if not w8:
        return {"branch": "insufficient_data", "reason": "need a convrot case"}

    pct = w8.get("profile", {}).get("buckets_pct", {})
    convrot_tax = (
        float(pct.get("convrot_rht", 0))
        + float(pct.get("convrot_act_quant", 0))
        + float(pct.get("convrot_dequant", 0))
        + float(pct.get("convrot_gemm", 0))
        + float(pct.get("convrot_other", 0))
    )
    gemm = float(pct.get("gemm_generic", 0)) + float(pct.get("convrot_gemm", 0))
    other_model = (
        float(pct.get("attention", 0))
        + float(pct.get("norm_act", 0))
        + float(pct.get("other", 0))
    )
    rht_pct = float(pct.get("convrot_rht", 0))
    dequant_linear_pct = float(pct.get("convrot_gemm", 0))

    # Prefer wall-clock ratio when bf16 present.
    bf = by.get("bf16")
    wall_ratio = None
    if bf is not None:
        bf_s = float(bf.get("sec_per_step_no_profiler") or 0)
        w_s = float(w8.get("sec_per_step_no_profiler") or 0)
        if bf_s > 0:
            wall_ratio = w_s / bf_s

    # W8A16 path often dequants to fp32 and loses bf16 Tensor Cores — top finding.
    if (
        w8.get("mode") == "w8a16"
        and dequant_linear_pct >= 8.0
        and rht_pct < 10.0
        and (wall_ratio is None or wall_ratio >= 1.3)
    ):
        branch = "fix_w8a16_keep_bf16_compute"
        reason = (
            f"W8A16 convrot_gemm (dequant linear) ~{dequant_linear_pct:.1f}% while "
            f"RHT only ~{rht_pct:.1f}%: likely fp32 F.linear tax vs bf16 TC"
        )
    elif convrot_tax >= 50.0:
        branch = "P2-K_triton_candidate"
        reason = f"convrot chain ~{convrot_tax:.1f}% of CUDA self-time (>=50%)"
    elif other_model >= 50.0 and convrot_tax < 25.0:
        branch = "stop_kernel_chase_do_prequant_or_product"
        reason = (
            f"non-convrot model ops dominate (~{other_model:.1f}%); "
            f"convrot tax only ~{convrot_tax:.1f}%"
        )
    elif gemm >= 40.0 and convrot_tax < 30.0:
        branch = "P1-I_shape_or_int_mm_tuning"
        reason = f"GEMM-heavy (~{gemm:.1f}%) with modest convrot tax (~{convrot_tax:.1f}%)"
    else:
        branch = "P0-C_prequant_or_P1-G_scope"
        reason = (
            f"mixed: convrot_tax={convrot_tax:.1f}% gemm={gemm:.1f}% "
            f"other_model={other_model:.1f}%"
        )
    return {
        "branch": branch,
        "reason": reason,
        "convrot_tax_pct": convrot_tax,
        "gemm_pct": gemm,
        "other_model_pct": other_model,
        "rht_pct": rht_pct,
        "convrot_gemm_pct": dequant_linear_pct,
        "wall_ratio_vs_bf16": wall_ratio,
        "based_on_label": w8["label"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group-size", type=int, default=256)
    parser.add_argument("--scope", default="mlp")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--dit-path", type=Path, default=DEFAULT_DIT_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("output/tests/convrot_step_profile.json"),
    )
    parser.add_argument(
        "--chrome-dir",
        type=Path,
        default=None,
        help="Optional dir for chrome traces (one file per case)",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="bf16,w8a16_free,w8a8_auto",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    all_cases = {
        "bf16": dict(label="bf16", mode=None, free_base=False, gemm_env=None),
        "w8a16_free": dict(
            label="w8a16_free", mode="w8a16", free_base=True, gemm_env=None
        ),
        "w8a8_auto": dict(
            label="w8a8_auto", mode="w8a8", free_base=True, gemm_env="auto"
        ),
        "w8a8_float": dict(
            label="w8a8_float", mode="w8a8", free_base=True, gemm_env="float"
        ),
    }
    case_keys = [c.strip() for c in args.cases.split(",") if c.strip()]
    results: list[dict] = []
    for key in case_keys:
        case = all_cases[key]
        print(f"=== profile {case['label']} ===", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        chrome = None
        if args.chrome_dir is not None:
            chrome = args.chrome_dir / f"{case['label']}.json"
        results.append(
            run_case(
                **case,
                dit_path=args.dit_path,
                data_dir=args.data_dir,
                device=device,
                dtype=dtype,
                scope=args.scope,
                group_size=args.group_size,
                seed=args.seed,
                warmup=args.warmup,
                export_chrome=chrome,
            )
        )

    decision = _decision_from_results(results)
    payload = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "results": results,
        "decision": decision,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"decision: {decision}", flush=True)
    print(f"wrote {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
