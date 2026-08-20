#!/usr/bin/env python3
"""Profile Krea-2 GEMM and attention shapes on the local CMP 90HX.

The probe uses CUDA events for low-overhead timing and wraps every case in an
NVTX range so Nsight Systems/Compute can select the same workload precisely.
It does not load the full DiT or modify GPU clocks/power settings.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("K2_NSIGHT_GPU", "1"))

import torch
import torch.nn.functional as F
from bitsandbytes.nn import Linear4bit, Params4bit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from library.models.krea2_raw.attention_backend import (  # noqa: E402
    run_krea2_attention,
    validate_krea2_attention_mode,
)

# #region agent log
_DEBUG_LOG_PATH = Path("/home/scv/nvme0n1p1/训练器相关/krea2-webui/.cursor/debug-c024b8.log")
_DEBUG_SESSION_ID = "c024b8"
_DEBUG_RUN_ID = os.environ.get("K2_DEBUG_RUN_ID", "pre-nsight")


def _agent_log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, object],
) -> None:
    payload = {
        "sessionId": _DEBUG_SESSION_ID,
        "runId": _DEBUG_RUN_ID,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        _DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _m_alignment(m: int) -> dict[str, object]:
    return {
        "m": m,
        "mod64": m % 64,
        "mod128": m % 128,
        "mod256": m % 256,
        "aligned64": m % 64 == 0,
        "aligned128": m % 128 == 0,
        "aligned256": m % 256 == 0,
        "token_family_member": m in {4032, 4200, 4608, 4864},
    }


# #endregion

BF16_GEMM_SHAPES = {
    "proj": (4608, 6144, 6144),
    "mlp_up": (4608, 16384, 6144),
    "mlp_down": (4608, 6144, 16384),
}
M_SWEEP = (4096, 4107, 4480, 4544, 4607, 4608, 4672, 4736, 4864)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("all", "gemm-sweep", "gemm-train", "nf4-train", "attention"),
        default="all",
    )
    parser.add_argument("--case", default=None, help="Run only one exact case label")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--preheat-seconds", type=float, default=2.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/nsight/krea2_90hx_tiles.json"),
    )
    parser.add_argument("--list-cases", action="store_true")
    return parser.parse_args()


def _case_labels() -> tuple[str, ...]:
    sweep = tuple(f"bf16_fwd_proj_m{m}" for m in M_SWEEP)
    train = tuple(f"bf16_train_{name}" for name in BF16_GEMM_SHAPES)
    nf4 = ("nf4_train_proj", "nf4_train_mlp_up")
    attention = (
        "attention_torch_l4608_valid4107",
        "attention_flash_l4608_valid4107",
        "attention_flash_l4608_valid4608",
    )
    return sweep + train + nf4 + attention


def _selected(args: argparse.Namespace, label: str) -> bool:
    return args.case is None or args.case == label


def _event_times(
    label: str,
    operation: Callable[[], None],
    *,
    warmup: int,
    repeats: int,
) -> list[float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    torch.cuda.nvtx.range_push(label)
    try:
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            operation()
            end.record()
            events.append((start, end))
    finally:
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in events]


def _summarize(
    label: str,
    times_ms: list[float],
    *,
    kind: str,
    shape: tuple[int, ...],
    flop_count: int | None = None,
) -> dict[str, object]:
    median_ms = statistics.median(times_ms)
    record: dict[str, object] = {
        "label": label,
        "kind": kind,
        "shape": list(shape),
        "times_ms": times_ms,
        "median_ms": median_ms,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }
    if flop_count is not None:
        record["effective_tflops"] = flop_count / (median_ms * 1e9)
    # #region agent log
    shape_list = list(shape)
    m_meta = _m_alignment(int(shape_list[0])) if shape_list else {}
    _agent_log(
        "H1-H2" if kind.startswith("bf16_linear") else ("H4" if "nf4" in kind else "H3"),
        "probe_90hx_nsight_tiles.py:_summarize",
        "case_timing",
        {
            "label": label,
            "kind": kind,
            "shape": shape_list,
            "median_ms": median_ms,
            "min_ms": record["min_ms"],
            "max_ms": record["max_ms"],
            "effective_tflops": record.get("effective_tflops"),
            "alignment": m_meta,
        },
    )
    # #endregion
    print(
        f"{label}: median={median_ms:.3f}ms"
        + (
            f" effective={record['effective_tflops']:.2f}TFLOP/s"
            if "effective_tflops" in record
            else ""
        ),
        flush=True,
    )
    return record


def _cleanup(*objects: object) -> None:
    del objects
    gc.collect()
    torch.cuda.empty_cache()


def _preheat_gpu(seconds: float) -> None:
    """Raise an idle mining card from P8 before collecting short cases."""
    if seconds <= 0:
        return
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(90)
    x = torch.randn(1024, 6144, device=device, dtype=torch.bfloat16, generator=generator)
    weight = torch.randn(
        6144, 6144, device=device, dtype=torch.bfloat16, generator=generator
    )
    deadline = time.perf_counter() + seconds
    iterations = 0
    while time.perf_counter() < deadline:
        for _ in range(8):
            F.linear(x, weight)
            iterations += 1
        torch.cuda.synchronize()
    del x, weight
    gc.collect()
    torch.cuda.empty_cache()
    print(f"preheat={seconds:.1f}s iterations={iterations}", flush=True)


def _run_bf16_linear(
    label: str,
    shape: tuple[int, int, int],
    *,
    training: bool,
    args: argparse.Namespace,
) -> dict[str, object]:
    m, n, k = shape
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(m + n + k)
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16, generator=generator)
    weight = torch.randn(n, k, device=device, dtype=torch.bfloat16, generator=generator)

    if training:
        x.requires_grad_(True)
        grad_output = torch.randn(
            m, n, device=device, dtype=torch.bfloat16, generator=generator
        )

        def operation() -> None:
            x.grad = None
            F.linear(x, weight).backward(grad_output)

        flop_count = 4 * m * n * k
        kind = "bf16_linear_forward_grad_input"
    else:

        def operation() -> None:
            F.linear(x, weight)

        flop_count = 2 * m * n * k
        kind = "bf16_linear_forward"

    times = _event_times(
        label, operation, warmup=args.warmup, repeats=args.repeats
    )
    record = _summarize(
        label, times, kind=kind, shape=shape, flop_count=flop_count
    )
    _cleanup(x, weight)
    return record


def _make_nf4_linear(n: int, k: int, device: torch.device) -> Linear4bit:
    layer = Linear4bit(
        k,
        n,
        bias=False,
        compute_dtype=torch.bfloat16,
        compress_statistics=True,
        quant_type="nf4",
        device="cpu",
    )
    generator = torch.Generator(device="cpu").manual_seed(n + k)
    weight = torch.randn(n, k, dtype=torch.bfloat16, generator=generator)
    layer.weight = Params4bit(
        weight,
        requires_grad=False,
        compress_statistics=True,
        quant_type="nf4",
        module=layer,
    )
    return layer.to(device)


def _run_nf4_linear(
    label: str,
    shape: tuple[int, int, int],
    *,
    args: argparse.Namespace,
) -> dict[str, object]:
    m, n, k = shape
    device = torch.device("cuda")
    layer = _make_nf4_linear(n, k, device)
    generator = torch.Generator(device=device).manual_seed(m + n + k)
    x = torch.randn(
        m, k, device=device, dtype=torch.bfloat16, generator=generator,
        requires_grad=True,
    )
    grad_output = torch.randn(
        m, n, device=device, dtype=torch.bfloat16, generator=generator
    )

    def operation() -> None:
        x.grad = None
        layer(x).backward(grad_output)

    times = _event_times(
        label, operation, warmup=args.warmup, repeats=args.repeats
    )
    record = _summarize(
        label,
        times,
        kind="nf4_linear_forward_grad_input",
        shape=shape,
        flop_count=4 * m * n * k,
    )
    record["quant_blocksize"] = int(layer.weight.quant_state.blocksize)
    _cleanup(layer, x, grad_output)
    return record


def _run_attention(
    label: str,
    *,
    mode: str,
    sequence: int,
    valid_tokens: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    validate_krea2_attention_mode(mode, dtype=dtype)
    generator = torch.Generator(device=device).manual_seed(sequence + valid_tokens)
    q = torch.randn(
        1, 48, sequence, 128, device=device, dtype=dtype, generator=generator,
        requires_grad=True,
    )
    k = torch.randn(
        1, 12, sequence, 128, device=device, dtype=dtype, generator=generator,
        requires_grad=True,
    )
    v = torch.randn(
        1, 12, sequence, 128, device=device, dtype=dtype, generator=generator,
        requires_grad=True,
    )
    valid = torch.arange(sequence, device=device).unsqueeze(0) < valid_tokens
    mask = valid[:, None, :, None] & valid[:, None, None, :]
    grad_output = torch.randn(
        1, sequence, 48 * 128,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    def operation() -> None:
        q.grad = k.grad = v.grad = None
        output = run_krea2_attention(q, k, v, mask=mask, gqa=True, mode=mode)
        output.backward(grad_output)

    times = _event_times(
        label, operation, warmup=args.warmup, repeats=args.repeats
    )
    record = _summarize(
        label,
        times,
        kind=f"attention_{mode}_forward_backward",
        shape=(1, 48, 12, sequence, 128),
    )
    record["valid_tokens"] = valid_tokens
    # #region agent log
    _agent_log(
        "H3",
        "probe_90hx_nsight_tiles.py:_run_attention",
        "attention_case",
        {
            "label": label,
            "mode": mode,
            "sequence": sequence,
            "valid_tokens": valid_tokens,
            "packed_ratio": valid_tokens / sequence,
            "seq_mod64": sequence % 64,
            "seq_mod128": sequence % 128,
            "valid_mod64": valid_tokens % 64,
            "valid_mod128": valid_tokens % 128,
            "median_ms": record["median_ms"],
            "headdim": 128,
            "hq": 48,
            "hkv": 12,
        },
    )
    # #endregion
    _cleanup(q, k, v, mask, grad_output)
    return record


def _run_suite(args: argparse.Namespace) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if args.suite in {"all", "gemm-sweep"}:
        for m in M_SWEEP:
            label = f"bf16_fwd_proj_m{m}"
            if _selected(args, label):
                records.append(
                    _run_bf16_linear(
                        label, (m, 6144, 6144), training=False, args=args
                    )
                )
    if args.suite in {"all", "gemm-train"}:
        for name, shape in BF16_GEMM_SHAPES.items():
            label = f"bf16_train_{name}"
            if _selected(args, label):
                records.append(
                    _run_bf16_linear(label, shape, training=True, args=args)
                )
    if args.suite in {"all", "nf4-train"}:
        for name in ("proj", "mlp_up"):
            label = f"nf4_train_{name}"
            if _selected(args, label):
                records.append(_run_nf4_linear(label, BF16_GEMM_SHAPES[name], args=args))
    if args.suite in {"all", "attention"}:
        specs = (
            ("attention_torch_l4608_valid4107", "torch", 4608, 4107),
            ("attention_flash_l4608_valid4107", "flash", 4608, 4107),
            ("attention_flash_l4608_valid4608", "flash", 4608, 4608),
        )
        for label, mode, sequence, valid_tokens in specs:
            if _selected(args, label):
                records.append(
                    _run_attention(
                        label,
                        mode=mode,
                        sequence=sequence,
                        valid_tokens=valid_tokens,
                        args=args,
                    )
                )
    return records


def main() -> int:
    args = _parse_args()
    if args.list_cases:
        print("\n".join(_case_labels()))
        return 0
    if args.case is not None and args.case not in _case_labels():
        raise SystemExit(f"unknown --case {args.case!r}; use --list-cases")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    props = torch.cuda.get_device_properties(0)
    if "90HX" not in props.name:
        raise SystemExit(
            f"refusing to profile unexpected GPU {props.name!r}; "
            "set K2_NSIGHT_GPU/CUDA_VISIBLE_DEVICES to the CMP 90HX"
        )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(0)

    # #region agent log
    l2_cache_bytes = getattr(props, "L2_cache_size", None)
    shared_mem_per_block = getattr(props, "shared_memory_per_block", None)
    shared_mem_per_mp = getattr(props, "shared_memory_per_multiprocessor", None)
    _agent_log(
        "H5",
        "probe_90hx_nsight_tiles.py:main",
        "gpu_props",
        {
            "name": props.name,
            "cc": f"{props.major}.{props.minor}",
            "sms": props.multi_processor_count,
            "memory_gib": props.total_memory / 2**30,
            "l2_cache_bytes": l2_cache_bytes,
            "l2_cache_mib": (l2_cache_bytes / (1024 * 1024)) if l2_cache_bytes else None,
            "shared_memory_per_block": shared_mem_per_block,
            "shared_memory_per_multiprocessor": shared_mem_per_mp,
            "shared_memory_per_mp_kib": (
                shared_mem_per_mp / 1024 if shared_mem_per_mp else None
            ),
            "suite": args.suite,
            "case": args.case,
        },
    )
    # #endregion

    print(
        f"GPU={props.name} SM={props.major}.{props.minor} sms={props.multi_processor_count} "
        f"memory={props.total_memory / 2**30:.2f}GiB",
        flush=True,
    )
    _preheat_gpu(args.preheat_seconds)
    torch.cuda.profiler.start()
    try:
        records = _run_suite(args)
    finally:
        torch.cuda.profiler.stop()
    if not records:
        raise SystemExit(f"case {args.case!r} is not part of suite {args.suite!r}")

    result = {
        "gpu": {
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "sms": props.multi_processor_count,
            "memory_gib": props.total_memory / 2**30,
        },
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "config": {
            "suite": args.suite,
            "case": args.case,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "preheat_seconds": args.preheat_seconds,
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    # #region agent log
    sweep = [
        r
        for r in records
        if str(r.get("kind", "")).startswith("bf16_linear_forward")
        and isinstance(r.get("shape"), list)
        and len(r["shape"]) == 3
        and r["shape"][1:] == [6144, 6144]
    ]
    if sweep:
        best = max(sweep, key=lambda r: float(r.get("effective_tflops") or 0.0))
        worst = min(sweep, key=lambda r: float(r.get("effective_tflops") or 0.0))
        aligned = [
            r for r in sweep if _m_alignment(int(r["shape"][0]))["aligned128"]
        ]
        misaligned = [
            r for r in sweep if not _m_alignment(int(r["shape"][0]))["aligned128"]
        ]
        aligned_med = (
            statistics.median([float(r["effective_tflops"]) for r in aligned])
            if aligned
            else None
        )
        misaligned_med = (
            statistics.median([float(r["effective_tflops"]) for r in misaligned])
            if misaligned
            else None
        )
        _agent_log(
            "H1-H2",
            "probe_90hx_nsight_tiles.py:main",
            "gemm_sweep_summary",
            {
                "n_cases": len(sweep),
                "best_label": best.get("label"),
                "best_tflops": best.get("effective_tflops"),
                "worst_label": worst.get("label"),
                "worst_tflops": worst.get("effective_tflops"),
                "aligned128_median_tflops": aligned_med,
                "misaligned128_median_tflops": misaligned_med,
                "aligned_vs_misaligned_ratio": (
                    (aligned_med / misaligned_med)
                    if aligned_med and misaligned_med
                    else None
                ),
            },
        )
    _agent_log(
        "H3-H5",
        "probe_90hx_nsight_tiles.py:main",
        "suite_complete",
        {
            "output": str(args.output),
            "n_records": len(records),
            "labels": [r.get("label") for r in records],
        },
    )
    # #endregion
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
