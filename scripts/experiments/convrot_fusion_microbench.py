#!/usr/bin/env python
"""ConvRot single-op microbench + fusion upper-bound estimate (P0 step 1+2).

Freezes kernel-level baselines and estimates how much *true* fusion could
recover **without** writing Triton yet:

1. **Baselines** (forward):
   - bf16 ``F.linear``
   - W8A16: dequant every call + ``F.linear`` (current default)
   - W8A16: pre-dequantized weight + ``F.linear`` (isolates dequant tax)
   - W8A16: ``_weight_int8pack_mm`` (opt-in; often slower on 3080)
   - W8A8: ``int8_mm_scaled`` int_mm + post-scale (current)
   - W8A8: ``_int_mm`` only (no scale; lower bound for GEMM body)
   - W8A8: float fallback path

2. **Fusion upper bounds** (not full kernels — bound the tax):
   - W8A8 epilogue: ``t(scaled) - t(int_mm_only)`` ≈ max save if scale
     folds into GEMM epilogue with zero extra cost.
   - W8A16: ``t(dequant+linear) - t(predequant linear)`` ≈ max save if
     dequant is free / fused into load.
   - Backward: full ``gy @ dequant(W)`` vs out-dim chunked dequant.

Shapes default to Anima DiT MLP (layer1 8192×2048, layer2 2048×8192) and
token counts near constant-token buckets.

Example:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \\
    .venv/bin/python scripts/experiments/convrot_fusion_microbench.py \\
      --json-out output/tests/convrot_fusion_microbench.json
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

from library.runtime.convrot.gemm import (
    can_use_torch_int_mm,
    int8_mm_scaled,
    quantize_activation_absmax_int8,
)
from library.runtime.convrot.quant import (
    dequantize_weight,
    quantize_weight_per_output_channel,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _bench(
    fn,
    *,
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict:
    for _ in range(max(0, warmup)):
        fn()
    _sync(device)
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        ms = start.elapsed_time(end) / max(1, iters)
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        ms = (time.perf_counter() - t0) * 1000.0 / max(1, iters)
    return {"ms": float(ms), "iters": int(iters), "warmup": int(warmup)}


def _peak_bytes(device: torch.device, fn) -> int:
    if device.type != "cuda":
        fn()
        return 0
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    # touch context
    torch.zeros(1, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    _sync(device)
    return int(torch.cuda.max_memory_allocated(device))


def _make_weights(n: int, k: int, device: torch.device, dtype: torch.dtype):
    w = torch.randn(n, k, device=device, dtype=dtype)
    w_q, w_s = quantize_weight_per_output_channel(w.float())
    w_q = w_q.to(device)
    w_s = w_s.to(device=device, dtype=torch.float32)
    return w, w_q, w_s


def bench_shape(
    *,
    m: int,
    k: int,
    n: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    bwd_chunk: int,
) -> dict:
    torch.manual_seed(0)
    x = torch.randn(m, k, device=device, dtype=dtype)
    w, w_q, w_s = _make_weights(n, k, device, dtype)
    w_hat = dequantize_weight(w_q, w_s, dtype=dtype).to(device=device, dtype=dtype)
    x_q, x_s = quantize_activation_absmax_int8(x)
    x_q = x_q.contiguous()
    gy = torch.randn(m, n, device=device, dtype=dtype)

    rows: dict[str, dict] = {}

    # --- Forward baselines ---
    rows["fwd_bf16_linear"] = _bench(
        lambda: F.linear(x, w.to(dtype), None),
        device=device,
        warmup=warmup,
        iters=iters,
    )

    def w8a16_dequant_linear():
        weight = dequantize_weight(w_q, w_s, dtype=dtype).to(device=device, dtype=dtype)
        return F.linear(x, weight, None)

    rows["fwd_w8a16_dequant_linear"] = _bench(
        w8a16_dequant_linear, device=device, warmup=warmup, iters=iters
    )

    # Isolates dequant tax: weight already materialised (best-case "free dequant").
    rows["fwd_w8a16_predequant_linear"] = _bench(
        lambda: F.linear(x, w_hat, None),
        device=device,
        warmup=warmup,
        iters=iters,
    )

    pack_ok = (
        device.type == "cuda"
        and hasattr(torch, "_weight_int8pack_mm")
        and dtype in (torch.float16, torch.bfloat16, torch.float32)
    )
    if pack_ok:
        try:

            def pack_mm():
                return torch._weight_int8pack_mm(
                    x.contiguous(),
                    w_q.contiguous(),
                    w_s.contiguous(),
                )

            # one correctness probe
            _ = pack_mm()
            rows["fwd_w8a16_int8pack"] = _bench(
                pack_mm, device=device, warmup=warmup, iters=max(1, iters // 2)
            )
        except RuntimeError as exc:
            rows["fwd_w8a16_int8pack"] = {"ms": None, "error": str(exc)}

    use_int = can_use_torch_int_mm(m, k, n, device=device)
    rows["meta"] = {
        "m": m,
        "k": k,
        "n": n,
        "dtype": str(dtype).replace("torch.", ""),
        "can_int_mm": bool(use_int),
        "device": str(device),
    }

    if use_int:

        def int_mm_only():
            b = w_q.t().contiguous()
            return torch._int_mm(x_q, b)

        rows["fwd_w8a8_int_mm_only"] = _bench(
            int_mm_only, device=device, warmup=warmup, iters=iters
        )

        def int_mm_scaled_current():
            return int8_mm_scaled(x_q, x_s, w_q, w_s, prefer="int_mm")

        rows["fwd_w8a8_int_mm_post_scale"] = _bench(
            int_mm_scaled_current, device=device, warmup=warmup, iters=iters
        )

        # Elementwise scale tax alone (on an int32→fp32 converted buffer).
        acc = torch._int_mm(x_q, w_q.t().contiguous()).to(torch.float32)

        def scale_only():
            y = acc * x_s.reshape(-1, 1).to(torch.float32)
            return y * w_s.reshape(1, -1).to(torch.float32)

        rows["fwd_w8a8_scale_only_on_acc"] = _bench(
            scale_only, device=device, warmup=warmup, iters=iters
        )
    else:
        rows["fwd_w8a8_int_mm_only"] = {"ms": None, "skipped": "shape/device"}
        rows["fwd_w8a8_int_mm_post_scale"] = {"ms": None, "skipped": "shape/device"}

    def float_scaled():
        return int8_mm_scaled(x_q, x_s, w_q, w_s, prefer="float")

    rows["fwd_w8a8_float_fallback"] = _bench(
        float_scaled, device=device, warmup=warmup, iters=iters
    )

    # --- Backward baselines ---
    def bwd_full():
        w_full = dequantize_weight(w_q, w_s, dtype=dtype).to(device=device, dtype=dtype)
        return gy @ w_full

    rows["bwd_full_dequant_mm"] = _bench(
        bwd_full, device=device, warmup=warmup, iters=iters
    )
    rows["bwd_full_peak_bytes"] = _peak_bytes(device, bwd_full)

    chunk = max(16, int(bwd_chunk))

    def bwd_chunked():
        out = torch.zeros(m, k, device=device, dtype=dtype)
        for n0 in range(0, n, chunk):
            n1 = min(n, n0 + chunk)
            tile = dequantize_weight(w_q[n0:n1], w_s[n0:n1], dtype=dtype).to(
                device=device, dtype=dtype
            )
            out = out + gy[:, n0:n1] @ tile
        return out

    rows["bwd_chunked_dequant_mm"] = _bench(
        bwd_chunked, device=device, warmup=warmup, iters=max(1, iters // 2)
    )
    rows["bwd_chunked_peak_bytes"] = _peak_bytes(device, bwd_chunked)
    rows["bwd_chunk"] = chunk

    # Numerical sanity: chunked ≈ full
    with torch.no_grad():
        y_full = bwd_full()
        y_chunk = bwd_chunked()
        rel = (y_full - y_chunk).float().norm() / y_full.float().norm().clamp_min(1e-8)
        rows["bwd_chunked_rel_err"] = float(rel.item())

    # --- Derived upper bounds ---
    def ms(key: str) -> float | None:
        v = rows.get(key) or {}
        return v.get("ms")

    t_bf16 = ms("fwd_bf16_linear")
    t_dq = ms("fwd_w8a16_dequant_linear")
    t_pre = ms("fwd_w8a16_predequant_linear")
    t_int_only = ms("fwd_w8a8_int_mm_only")
    t_int_scaled = ms("fwd_w8a8_int_mm_post_scale")
    t_scale_only = ms("fwd_w8a8_scale_only_on_acc")
    t_bwd_full = ms("bwd_full_dequant_mm")
    t_bwd_chunk = ms("bwd_chunked_dequant_mm")

    bounds: dict[str, float | None | str] = {}
    if t_dq is not None and t_pre is not None and t_dq > 0:
        bounds["w8a16_dequant_tax_ms"] = t_dq - t_pre
        bounds["w8a16_dequant_tax_pct_of_dequant_path"] = 100.0 * (t_dq - t_pre) / t_dq
        bounds["w8a16_vs_bf16_ratio"] = t_dq / t_bf16 if t_bf16 else None
        bounds["w8a16_predequant_vs_bf16_ratio"] = t_pre / t_bf16 if t_bf16 else None
        # Max relative step-op improvement if dequant becomes free.
        bounds["w8a16_fusion_upper_speedup_vs_current"] = t_dq / t_pre if t_pre else None
    if t_int_scaled is not None and t_int_only is not None and t_int_scaled > 0:
        bounds["w8a8_post_scale_tax_ms"] = t_int_scaled - t_int_only
        bounds["w8a8_post_scale_tax_pct"] = 100.0 * (t_int_scaled - t_int_only) / t_int_scaled
        bounds["w8a8_fusion_upper_speedup_vs_current"] = (
            t_int_scaled / t_int_only if t_int_only else None
        )
        bounds["w8a8_vs_bf16_ratio"] = t_int_scaled / t_bf16 if t_bf16 else None
        bounds["w8a8_int_mm_only_vs_bf16_ratio"] = (
            t_int_only / t_bf16 if t_bf16 and t_int_only else None
        )
    if t_scale_only is not None:
        bounds["w8a8_scale_only_ms"] = t_scale_only
    if t_bwd_full is not None and t_bwd_chunk is not None:
        bounds["bwd_chunked_vs_full_ratio"] = t_bwd_chunk / t_bwd_full if t_bwd_full else None
        peak_full = rows.get("bwd_full_peak_bytes") or 0
        peak_chunk = rows.get("bwd_chunked_peak_bytes") or 0
        bounds["bwd_peak_full_mb"] = peak_full / (1024 * 1024)
        bounds["bwd_peak_chunk_mb"] = peak_chunk / (1024 * 1024)
        if peak_full:
            bounds["bwd_peak_chunk_vs_full"] = peak_chunk / peak_full

    rows["upper_bounds"] = bounds
    return rows


def _shape_m(row: dict) -> int | None:
    meta = row.get("meta") or {}
    m = meta.get("m")
    return int(m) if m is not None else None


def _avg(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _collect_bounds(rows: list[dict]) -> dict[str, float | None]:
    w8a16_taxes: list[float] = []
    w8a8_taxes: list[float] = []
    w8a8_vs_bf16: list[float] = []
    w8a16_vs_bf16: list[float] = []
    w8a8_int_only_vs_bf16: list[float] = []
    w8a16_pre_vs_bf16: list[float] = []
    bwd_speed: list[float] = []
    bwd_peak: list[float] = []
    for s in rows:
        b = s.get("upper_bounds") or {}
        if b.get("w8a16_dequant_tax_pct_of_dequant_path") is not None:
            w8a16_taxes.append(float(b["w8a16_dequant_tax_pct_of_dequant_path"]))
        if b.get("w8a8_post_scale_tax_pct") is not None:
            w8a8_taxes.append(float(b["w8a8_post_scale_tax_pct"]))
        if b.get("w8a8_vs_bf16_ratio") is not None:
            w8a8_vs_bf16.append(float(b["w8a8_vs_bf16_ratio"]))
        if b.get("w8a16_vs_bf16_ratio") is not None:
            w8a16_vs_bf16.append(float(b["w8a16_vs_bf16_ratio"]))
        if b.get("w8a8_int_mm_only_vs_bf16_ratio") is not None:
            w8a8_int_only_vs_bf16.append(float(b["w8a8_int_mm_only_vs_bf16_ratio"]))
        if b.get("w8a16_predequant_vs_bf16_ratio") is not None:
            w8a16_pre_vs_bf16.append(float(b["w8a16_predequant_vs_bf16_ratio"]))
        if b.get("bwd_chunked_vs_full_ratio") is not None:
            bwd_speed.append(float(b["bwd_chunked_vs_full_ratio"]))
        if b.get("bwd_peak_chunk_vs_full") is not None:
            bwd_peak.append(float(b["bwd_peak_chunk_vs_full"]))
    return {
        "avg_w8a16_dequant_tax_pct": _avg(w8a16_taxes),
        "avg_w8a8_post_scale_tax_pct": _avg(w8a8_taxes),
        "avg_w8a16_vs_bf16_ratio": _avg(w8a16_vs_bf16),
        "avg_w8a8_vs_bf16_ratio": _avg(w8a8_vs_bf16),
        "avg_w8a8_int_mm_only_vs_bf16_ratio": _avg(w8a8_int_only_vs_bf16),
        "avg_w8a16_predequant_vs_bf16_ratio": _avg(w8a16_pre_vs_bf16),
        "avg_bwd_chunked_vs_full_speed": _avg(bwd_speed),
        "avg_bwd_chunked_vs_full_peak": _avg(bwd_peak),
    }


def _recommend(all_shapes: list[dict]) -> dict:
    """Aggregate decision text from measured upper bounds.

    Training-relevant decisions weight **large-M** rows (M>=512, ideally
    bucket-scale M≈4032). Tiny-M rows inflate dequant % because fixed overhead
    dominates; they must not open P2 Triton by themselves.
    """
    overall = _collect_bounds(all_shapes)
    large = [s for s in all_shapes if (_shape_m(s) or 0) >= 512]
    large_m = _collect_bounds(large) if large else overall
    bucket = [s for s in all_shapes if (_shape_m(s) or 0) >= 2000]
    bucket_m = _collect_bounds(bucket) if bucket else large_m

    a16 = bucket_m.get("avg_w8a16_dequant_tax_pct")
    a8 = bucket_m.get("avg_w8a8_post_scale_tax_pct")
    r8 = bucket_m.get("avg_w8a8_vs_bf16_ratio")
    r8_only = bucket_m.get("avg_w8a8_int_mm_only_vs_bf16_ratio")
    r16 = bucket_m.get("avg_w8a16_vs_bf16_ratio")
    r16_pre = bucket_m.get("avg_w8a16_predequant_vs_bf16_ratio")
    bspd = overall.get("avg_bwd_chunked_vs_full_speed")
    bpk = overall.get("avg_bwd_chunked_vs_full_peak")

    # Epilogue/K-fuse only if *training-M* tax is material AND free-fusion path
    # can beat or match bf16. Free dequant → predequant ≈ bf16 is a tax kill,
    # not a path to step < bf16 once RHT remains.
    open_w8a8_epilogue = (
        a8 is not None
        and a8 >= 15.0
        and r8_only is not None
        and r8_only <= 1.15  # free epilogue must land near bf16
    )
    # W8A16 "K-loop dequant fusion" upper bound is predequant linear ≈ bf16.
    # Worth a *cheap* dequant elimination only if tax is large at bucket M;
    # never open full Triton solely for this — ceiling is bf16 linear, not faster.
    open_w8a16_cheap_dequant_elim = a16 is not None and a16 >= 25.0
    open_w8a16_kfuse_triton = False  # explicitly not: ceiling == bf16 linear
    open_bwd_chunk_for_speed = bspd is not None and bspd < 0.95
    open_bwd_chunk_for_peak = bpk is not None and bpk < 0.85
    open_p2_triton = False  # need chain>50% e2e + op beat bf16; neither holds

    decision = {
        **{f"all_{k}": v for k, v in overall.items()},
        **{f"large_m_{k}": v for k, v in large_m.items()},
        **{f"bucket_m_{k}": v for k, v in bucket_m.items()},
        # Back-compat keys used by tests / docs (bucket-M primary).
        "avg_w8a16_dequant_tax_pct": a16,
        "avg_w8a8_post_scale_tax_pct": a8,
        "avg_w8a16_vs_bf16_ratio": r16,
        "avg_w8a8_vs_bf16_ratio": r8,
        "avg_bwd_chunked_vs_full_speed": bspd,
        "avg_bwd_chunked_vs_full_peak": bpk,
        "recommend_w8a8_epilogue_fusion_impl": open_w8a8_epilogue,
        "recommend_w8a16_cheap_dequant_elim": open_w8a16_cheap_dequant_elim,
        "recommend_w8a16_kloop_fusion_impl": open_w8a16_kfuse_triton,
        "recommend_bwd_chunk_for_speed": open_bwd_chunk_for_speed,
        "recommend_bwd_chunk_for_peak": open_bwd_chunk_for_peak,
        "recommend_p2_triton_now": open_p2_triton,
        "notes": [],
    }
    if a8 is not None:
        decision["notes"].append(
            f"[bucket-M] W8A8 post-scale tax ≈ {a8:.1f}% of int8 path; "
            f"int_mm-only vs bf16 ≈ {r8_only:.2f}×; "
            f"epilogue impl "
            f"{'YES' if open_w8a8_epilogue else 'NO'} "
            f"(need tax≥15% AND int_mm-only ≤1.15× bf16)."
        )
    if a16 is not None:
        decision["notes"].append(
            f"[bucket-M] W8A16 dequant tax ≈ {a16:.1f}% of dequant+linear; "
            f"predequant/bf16 ≈ {r16_pre:.3f}× (free dequant ceiling = bf16 linear). "
            f"Cheap elim {'interesting' if open_w8a16_cheap_dequant_elim else 'low ROI at bucket M'}; "
            f"Triton K-fuse NOT recommended."
        )
    if r8 is not None and (r8_only or 0) > 1.15:
        decision["notes"].append(
            f"W8A8 cannot win bf16 on 3080 via epilogue alone: int_mm body is already "
            f"{r8_only:.2f}× bf16 at bucket M."
        )
    if bspd is not None:
        decision["notes"].append(
            f"Bwd chunked speed {bspd:.2f}× full; peak {bpk:.2f}× full — "
            f"{'speed win' if open_bwd_chunk_for_speed else 'no speed win'}, "
            f"{'peak win' if open_bwd_chunk_for_peak else 'weak peak win'}."
        )
    decision["notes"].append(
        "P2 Triton K-loop/epilogue fusion: NOT next default. Keep P1 "
        "(compile / scope / mixed precision). Optional: peak-only bwd chunk; "
        "never default int8pack (microbench 9–100× slower than dequant)."
    )
    return decision


def parse_shapes(raw: str) -> list[tuple[int, int, int, str]]:
    """Parse ``name:M,K,N;...`` or default Anima MLP set."""
    if not raw or raw.strip().lower() in {"default", "auto"}:
        # M≈token count; layer1/2 from anima-preview3 mlp weights.
        return [
            ("mlp_l1_tokens64", 64, 2048, 8192),
            ("mlp_l1_tokens512", 512, 2048, 8192),
            ("mlp_l1_tokens4032", 4032, 2048, 8192),
            ("mlp_l2_tokens64", 64, 8192, 2048),
            ("mlp_l2_tokens512", 512, 8192, 2048),
            ("mlp_l2_tokens4032", 4032, 8192, 2048),
            ("square_mid", 512, 2048, 2048),
        ]
    out: list[tuple[int, int, int, str]] = []
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            name, dims = part.split(":", 1)
        else:
            name, dims = part, part
        m, k, n = (int(x) for x in dims.split(","))
        out.append((name.strip(), m, k, n))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("output/tests/convrot_fusion_microbench.json"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--bwd-chunk", type=int, default=256)
    parser.add_argument(
        "--shapes",
        default="default",
        help="name:M,K,N;... or 'default' for Anima MLP token sweeps",
    )
    args = parser.parse_args(argv)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 2
    device = torch.device(args.device if args.device != "cuda" else "cuda:0")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    # Stable env for fair compare (match production defaults).
    os.environ.setdefault("ANIMA_CONVROT_INT8_GEMM", "auto")
    os.environ.setdefault("ANIMA_CONVROT_W8A16_KERNEL", "dequant")

    if device.type == "cuda":
        torch.zeros(1, device=device)
        gpu_name = torch.cuda.get_device_name(device)
    else:
        gpu_name = "cpu"

    shapes = parse_shapes(args.shapes)
    results = []
    for name, m, k, n in shapes:
        print(f"[bench] {name} M={m} K={k} N={n} ...", flush=True)
        row = bench_shape(
            m=m,
            k=k,
            n=n,
            device=device,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
            bwd_chunk=args.bwd_chunk,
        )
        row["name"] = name
        results.append(row)
        ub = row.get("upper_bounds") or {}
        print(
            f"  bf16={row.get('fwd_bf16_linear', {}).get('ms')} ms  "
            f"w8a16_dq={row.get('fwd_w8a16_dequant_linear', {}).get('ms')} ms  "
            f"w8a8_scaled={row.get('fwd_w8a8_int_mm_post_scale', {}).get('ms')} ms  "
            f"tax16={ub.get('w8a16_dequant_tax_pct_of_dequant_path')}%  "
            f"tax8={ub.get('w8a8_post_scale_tax_pct')}%",
            flush=True,
        )

    decision = _recommend(results)
    # Anchor known end-to-end baselines (frozen from prior mem_speed A2 run).
    e2e_anchor = {
        "source": "output/tests/convrot_mem_speed_bf16_compute.json",
        "bf16_sec_per_step": 1.620758218166884,
        "w8a16_free_sec_per_step": 1.9881715060000715,
        "w8a8_auto_sec_per_step": 2.5470875309996095,
        "bf16_peak_gb": 4.995718479156494,
        "w8a16_free_peak_gb": 4.337944984436035,
        "w8a8_auto_peak_gb": 4.489767551422119,
        "note": "End-to-end short train microbench (6 steps), not re-run in this script.",
    }
    payload = {
        "device": str(device),
        "gpu_name": gpu_name,
        "torch": torch.__version__,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "iters": args.iters,
        "bwd_chunk": args.bwd_chunk,
        "e2e_anchor": e2e_anchor,
        "results": results,
        "decision": decision,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
