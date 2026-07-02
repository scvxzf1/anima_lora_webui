#!/usr/bin/env python
"""Measure LoKr grouped-delta forward/backward latency and kernel/launch shape.

Defaults intentionally mirror the hottest observed training shape from
`blocks.27.mlp.layer1`:

- input:  `[2, 1, 72, 56, 2048]`
- output: `[2, 1, 72, 56, 8192]`
- factor: `8` -> `in_dim=256`, `out_dim=1024`
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench._common import make_run_dir, write_result

DEFAULT_GPU_INDEX = "1"
DEFAULT_ROOT = "output/bench/lokr_grouped_delta_microbench"
DEFAULT_BACKENDS = ("eager", "triton")
DEFAULT_WARMUP = 5
DEFAULT_ITERS = 20
DEFAULT_SEED = 1234
DEFAULT_DTYPE = "bfloat16"
DEFAULT_WEIGHT_DTYPE = "float32"
DEFAULT_BACKWARD_BACKEND = "eager"


@dataclass(frozen=True)
class BenchShape:
    outer_shape: tuple[int, ...]
    factor: int
    in_features: int
    out_features: int
    group_size: int

    @property
    def in_dim(self) -> int:
        return self.in_features // self.factor

    @property
    def out_dim(self) -> int:
        return self.out_features // self.factor

    @property
    def rows(self) -> int:
        rows = 1
        for dim in self.outer_shape:
            rows *= dim
        return rows


def _parse_shape(text: str) -> tuple[int, ...]:
    dims = []
    for raw in text.split(","):
        raw = raw.strip()
        if raw:
            dims.append(int(raw))
    if not dims:
        raise argparse.ArgumentTypeError("shape must contain at least one dimension")
    if any(d <= 0 for d in dims):
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return tuple(dims)


def _parse_backends(text: str) -> tuple[str, ...]:
    values = []
    for raw in text.split(","):
        raw = raw.strip().lower()
        if raw:
            values.append(raw)
    if not values:
        raise argparse.ArgumentTypeError("expected at least one backend")
    unknown = sorted(set(values) - {"eager", "triton"})
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unsupported backend(s): {', '.join(unknown)}; expected eager,triton"
        )
    return tuple(dict.fromkeys(values))


def _gpu_rows() -> list[dict[str, str]]:
    import subprocess

    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
    except Exception:
        return []
    rows: list[dict[str, str]] = []
    for line in cp.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            rows.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "memory_total_mb": parts[2],
                    "memory_used_mb": parts[3],
                    "utilization_gpu_pct": parts[4],
                }
            )
    return rows


def _check_gpu(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = _gpu_rows()
    selected = next((r for r in rows if r.get("index") == args.gpu_index), None)
    if selected is None and rows:
        raise SystemExit(f"GPU index {args.gpu_index} not found in nvidia-smi rows: {rows}")
    if selected is not None:
        total = int(float(selected.get("memory_total_mb", "0") or 0))
        if total < args.min_vram_mb and not args.allow_low_vram:
            raise SystemExit(
                f"refusing to bench on GPU {args.gpu_index} ({selected.get('name')}) "
                f"with only {total} MB VRAM; pass --allow-low-vram to override"
            )
    return rows


def _set_cuda_env(gpu_index: str) -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _torch_dtype(name: str):
    import torch

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:  # pragma: no cover - argparse guards values
        raise ValueError(name) from exc


def _make_inputs(
    shape: BenchShape,
    *,
    dtype: str,
    weight_dtype: str,
    seed: int,
    w1_scale: float,
    w2_scale: float,
    gate_value: float,
):
    import torch

    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    base = torch.randn(
        *shape.outer_shape,
        shape.out_features,
        device=device,
        dtype=_torch_dtype(dtype),
    )
    x = torch.randn(
        *shape.outer_shape,
        shape.in_features,
        device=device,
        dtype=_torch_dtype(dtype),
    )
    w1 = torch.randn(
        shape.factor,
        shape.factor,
        device=device,
        dtype=_torch_dtype(weight_dtype),
    ) * float(w1_scale)
    w2 = torch.randn(
        shape.out_dim,
        shape.in_dim,
        device=device,
        dtype=_torch_dtype(weight_dtype),
    ) * float(w2_scale)
    gate = torch.tensor([[float(gate_value)]], device=device, dtype=torch.float32)
    return base, x, w1, w2, gate


def _run_once(
    backend: str,
    inputs: tuple[Any, ...],
    shape: BenchShape,
    chunk_bytes: int,
    *,
    backward_backend: str,
):
    from networks.plugins.lokr.autograd import lokr_add_grouped_delta_

    base, x, w1, w2, gate = inputs
    return lokr_add_grouped_delta_(
        base.clone(),
        x,
        w1,
        w2,
        gate,
        shape.factor,
        shape.in_dim,
        shape.out_dim,
        shape.group_size,
        chunk_bytes,
        backend=backend,
        backward_backend=backward_backend,
    )


def _make_backward_state(inputs: tuple[Any, ...], *, seed: int) -> dict[str, Any]:
    import torch

    base, x, w1, w2, gate = inputs
    torch.manual_seed(int(seed) + 1)
    x_leaf = x.detach().clone().requires_grad_(True)
    w1_leaf = w1.detach().clone().requires_grad_(True)
    w2_leaf = w2.detach().clone().requires_grad_(True)
    grad_out = torch.randn_like(base)
    return {
        "base": base.detach(),
        "x": x_leaf,
        "w1": w1_leaf,
        "w2": w2_leaf,
        "gate": gate.detach(),
        "grad_out": grad_out,
    }


def _clear_backward_grads(state: dict[str, Any]) -> None:
    for key in ("x", "w1", "w2"):
        tensor = state[key]
        if tensor.grad is not None:
            tensor.grad = None


def _run_forward_backward_once(
    backend: str,
    state: dict[str, Any],
    shape: BenchShape,
    chunk_bytes: int,
    *,
    backward_backend: str,
    sync_ranges: bool = False,
):
    import torch
    from torch.autograd.profiler import record_function

    _clear_backward_grads(state)
    inputs = (
        state["base"],
        state["x"],
        state["w1"],
        state["w2"],
        state["gate"],
    )

    with record_function("lokr_forward"):
        y = _run_once(
            backend,
            inputs,
            shape,
            chunk_bytes,
            backward_backend=backward_backend,
        )
        if sync_ranges:
            torch.cuda.synchronize()

    with record_function("lokr_backward"):
        y.backward(state["grad_out"])
        if sync_ranges:
            torch.cuda.synchronize()

    return y


def _time_forward_ms(
    backend: str,
    inputs: tuple[Any, ...],
    shape: BenchShape,
    *,
    chunk_bytes: int,
    backward_backend: str,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    import torch

    for _ in range(warmup):
        _run_once(
            backend,
            inputs,
            shape,
            chunk_bytes,
            backward_backend=backward_backend,
        )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples_ms: list[float] = []
    for _ in range(iters):
        start.record()
        _run_once(
            backend,
            inputs,
            shape,
            chunk_bytes,
            backward_backend=backward_backend,
        )
        end.record()
        torch.cuda.synchronize()
        samples_ms.append(float(start.elapsed_time(end)))

    ordered = sorted(samples_ms)
    return {
        "avg_ms": round(sum(samples_ms) / len(samples_ms), 6),
        "median_ms": round(ordered[len(ordered) // 2], 6),
        "min_ms": round(min(samples_ms), 6),
        "max_ms": round(max(samples_ms), 6),
    }


def _time_forward_backward_ms(
    backend: str,
    inputs: tuple[Any, ...],
    shape: BenchShape,
    *,
    chunk_bytes: int,
    backward_backend: str,
    warmup: int,
    iters: int,
    seed: int,
) -> dict[str, Any]:
    import torch

    state = _make_backward_state(inputs, seed=seed)
    for _ in range(warmup):
        _run_forward_backward_once(
            backend,
            state,
            shape,
            chunk_bytes,
            backward_backend=backward_backend,
        )
    torch.cuda.synchronize()

    f_start = torch.cuda.Event(enable_timing=True)
    f_end = torch.cuda.Event(enable_timing=True)
    b_start = torch.cuda.Event(enable_timing=True)
    b_end = torch.cuda.Event(enable_timing=True)
    forward_ms: list[float] = []
    backward_ms: list[float] = []
    total_ms: list[float] = []
    peak_allocated_gb: list[float] = []
    peak_reserved_gb: list[float] = []
    peak_allocated_delta_gb: list[float] = []
    peak_reserved_delta_gb: list[float] = []

    for _ in range(iters):
        _clear_backward_grads(state)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_allocated = torch.cuda.memory_allocated()
        start_reserved = torch.cuda.memory_reserved()
        inputs_iter = (
            state["base"],
            state["x"],
            state["w1"],
            state["w2"],
            state["gate"],
        )
        f_start.record()
        y = _run_once(
            backend,
            inputs_iter,
            shape,
            chunk_bytes,
            backward_backend=backward_backend,
        )
        f_end.record()
        b_start.record()
        y.backward(state["grad_out"])
        b_end.record()
        torch.cuda.synchronize()

        forward_ms.append(float(f_start.elapsed_time(f_end)))
        backward_ms.append(float(b_start.elapsed_time(b_end)))
        total_ms.append(float(f_start.elapsed_time(b_end)))
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        peak_allocated_gb.append(float(peak_allocated / (1024**3)))
        peak_reserved_gb.append(float(peak_reserved / (1024**3)))
        peak_allocated_delta_gb.append(
            float(max(0, peak_allocated - start_allocated) / (1024**3))
        )
        peak_reserved_delta_gb.append(
            float(max(0, peak_reserved - start_reserved) / (1024**3))
        )

    def _summarize(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        return {
            "avg_ms": round(sum(values) / len(values), 6),
            "median_ms": round(ordered[len(ordered) // 2], 6),
            "min_ms": round(min(values), 6),
            "max_ms": round(max(values), 6),
        }

    def _summarize_gb(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        return {
            "avg_gb": round(sum(values) / len(values), 6),
            "median_gb": round(ordered[len(ordered) // 2], 6),
            "min_gb": round(min(values), 6),
            "max_gb": round(max(values), 6),
        }

    forward_summary = _summarize(forward_ms)
    backward_summary = _summarize(backward_ms)
    total_summary = _summarize(total_ms)
    return {
        "forward_ms": forward_summary,
        "backward_ms": backward_summary,
        "total_ms": total_summary,
        "backward_share_pct": round(
            backward_summary["avg_ms"] / max(total_summary["avg_ms"], 1e-9) * 100.0,
            3,
        ),
        "peak_allocated_gb": _summarize_gb(peak_allocated_gb),
        "peak_reserved_gb": _summarize_gb(peak_reserved_gb),
        "peak_allocated_delta_gb": _summarize_gb(peak_allocated_delta_gb),
        "peak_reserved_delta_gb": _summarize_gb(peak_reserved_delta_gb),
    }


def _load_trace_events(trace_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    return payload.get("traceEvents", [])


def _find_named_range(
    events: list[dict[str, Any]], name: str
) -> tuple[float, float] | None:
    candidates: list[tuple[float, float]] = []
    for ev in events:
        if ev.get("name") != name or ev.get("ph") != "X":
            continue
        try:
            start = float(ev["ts"])
            end = start + float(ev["dur"])
        except (KeyError, TypeError, ValueError):
            continue
        candidates.append((start, end))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1] - item[0])


def _find_named_ranges(events: list[dict[str, Any]], name: str) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    for ev in events:
        if ev.get("name") != name or ev.get("ph") != "X":
            continue
        try:
            start = float(ev["ts"])
            end = start + float(ev["dur"])
        except (KeyError, TypeError, ValueError):
            continue
        ranges.append((start, end))
    return ranges


def _event_overlaps(
    ev: dict[str, Any],
    interval: tuple[float, float] | list[tuple[float, float]] | None,
) -> bool:
    if interval is None:
        return True
    try:
        start = float(ev["ts"])
        end = start + float(ev.get("dur", 0.0) or 0.0)
    except (KeyError, TypeError, ValueError):
        return False
    intervals = interval if isinstance(interval, list) else [interval]
    for active in intervals:
        if start < active[1] and end > active[0]:
            return True
    return False


def _summarize_trace_events(
    events: list[dict[str, Any]],
    *,
    interval: tuple[float, float] | list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    kernels = [
        ev
        for ev in events
        if ev.get("cat") == "kernel" and _event_overlaps(ev, interval)
    ]
    launch_events = [
        ev
        for ev in events
        if ev.get("cat") in {"cuda_runtime", "cuda_driver"}
        and "LaunchKernel" in str(ev.get("name"))
        and _event_overlaps(ev, interval)
    ]

    thresholds_us = (10.0, 20.0, 50.0, 100.0)
    small_kernel_counts = {
        str(int(limit)): sum(
            1 for ev in kernels if float(ev.get("dur", 0.0) or 0.0) < limit
        )
        for limit in thresholds_us
    }

    kernel_name_counts: dict[str, int] = {}
    kernel_total_us = 0.0
    for ev in kernels:
        name = str(ev.get("name"))
        kernel_name_counts[name] = kernel_name_counts.get(name, 0) + 1
        kernel_total_us += float(ev.get("dur", 0.0) or 0.0)

    top_kernels = sorted(
        kernel_name_counts.items(), key=lambda item: (-item[1], item[0])
    )[:10]

    launch_name_counts: dict[str, int] = {}
    for ev in launch_events:
        name = str(ev.get("name"))
        launch_name_counts[name] = launch_name_counts.get(name, 0) + 1

    return {
        "kernel_count": len(kernels),
        "launch_count": len(launch_events),
        "kernel_total_us": round(kernel_total_us, 3),
        "small_kernel_counts": small_kernel_counts,
        "top_kernels": top_kernels,
        "launch_names": launch_name_counts,
    }


def _summarize_named_range_events(
    events: list[dict[str, Any]], name: str
) -> dict[str, Any]:
    ranges = _find_named_ranges(events, name)
    range_total_us = sum(end - start for start, end in ranges)
    range_max_us = max((end - start) for start, end in ranges) if ranges else 0.0
    return {
        "range_count": len(ranges),
        "range_total_us": round(range_total_us, 3),
        "range_avg_us": round(range_total_us / max(len(ranges), 1), 3),
        "range_max_us": round(range_max_us, 3),
        **_summarize_trace_events(events, interval=ranges),
    }


def _profile_forward_trace(
    backend: str,
    inputs: tuple[Any, ...],
    shape: BenchShape,
    *,
    chunk_bytes: int,
    backward_backend: str,
    trace_path: Path,
) -> dict[str, Any]:
    import torch
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        _run_once(
            backend,
            inputs,
            shape,
            chunk_bytes,
            backward_backend=backward_backend,
        )
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace_path))
    events = _load_trace_events(trace_path)
    return _summarize_trace_events(events)


def _profile_forward_backward_trace(
    backend: str,
    inputs: tuple[Any, ...],
    shape: BenchShape,
    *,
    chunk_bytes: int,
    backward_backend: str,
    trace_path: Path,
    seed: int,
) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile
    from networks.plugins.lokr import autograd as lokr_autograd

    state = _make_backward_state(inputs, seed=seed)
    previous = lokr_autograd.set_lokr_backward_phase_ranges(True)
    try:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            _run_forward_backward_once(
                backend,
                state,
                shape,
                chunk_bytes,
                backward_backend=backward_backend,
                sync_ranges=True,
            )
    finally:
        lokr_autograd.set_lokr_backward_phase_ranges(previous)
    prof.export_chrome_trace(str(trace_path))
    events = _load_trace_events(trace_path)
    forward_range = _find_named_range(events, "lokr_forward")
    backward_range = _find_named_range(events, "lokr_backward")
    phase_names = getattr(lokr_autograd, "_LOKR_BACKWARD_PHASE_NAMES", ())
    return {
        "all": _summarize_trace_events(events),
        "forward": _summarize_trace_events(events, interval=forward_range),
        "backward": _summarize_trace_events(events, interval=backward_range),
        "backward_phases": {
            name: _summarize_named_range_events(events, name) for name in phase_names
        },
    }


def _forward_diff(
    shape: BenchShape,
    inputs: tuple[Any, ...],
    *,
    chunk_bytes: int,
    backward_backend: str,
) -> dict[str, float]:
    import torch

    eager = _run_once(
        "eager",
        inputs,
        shape,
        chunk_bytes,
        backward_backend=backward_backend,
    )
    triton = _run_once(
        "triton",
        inputs,
        shape,
        chunk_bytes,
        backward_backend=backward_backend,
    )
    eager_f = eager.float()
    triton_f = triton.float()
    diff = (triton_f - eager_f).abs()
    ref = eager_f.abs().clamp_min(1e-8)
    rel = diff / ref
    torch.cuda.synchronize()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rms_abs": float(diff.square().mean().sqrt().item()),
        "max_rel": float(rel.max().item()),
        "mean_rel": float(rel.mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-index", default=DEFAULT_GPU_INDEX)
    parser.add_argument(
        "--allow-low-vram", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--min-vram-mb", type=int, default=8000)
    parser.add_argument("--outer-shape", type=_parse_shape, default=(2, 1, 72, 56))
    parser.add_argument("--in-features", type=int, default=2048)
    parser.add_argument("--out-features", type=int, default=8192)
    parser.add_argument("--factor", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default=DEFAULT_DTYPE,
    )
    parser.add_argument(
        "--weight-dtype",
        choices=("float16", "bfloat16", "float32"),
        default=DEFAULT_WEIGHT_DTYPE,
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--w1-scale", type=float, default=1.0)
    parser.add_argument("--w2-scale", type=float, default=1e-3)
    parser.add_argument("--gate-value", type=float, default=0.75)
    parser.add_argument("--backends", type=_parse_backends, default=DEFAULT_BACKENDS)
    parser.add_argument(
        "--backward-backend",
        choices=(
            "eager",
            "triton_grad_x",
            "triton_grad_w2_partial",
            "triton_grad_w2_grad_x",
        ),
        default=DEFAULT_BACKWARD_BACKEND,
    )
    parser.add_argument("--label", default="hot_mlp1")
    parser.add_argument("--output-root", default=DEFAULT_ROOT)
    args = parser.parse_args()

    if args.in_features <= 0 or args.out_features <= 0 or args.factor <= 0:
        raise SystemExit("in/out features and factor must be positive")
    if args.in_features % args.factor or args.out_features % args.factor:
        raise SystemExit("in/out features must be divisible by factor")
    if args.group_size <= 0:
        raise SystemExit("group-size must be positive")

    _check_gpu(args)
    _set_cuda_env(args.gpu_index)

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    shape = BenchShape(
        outer_shape=tuple(args.outer_shape),
        factor=int(args.factor),
        in_features=int(args.in_features),
        out_features=int(args.out_features),
        group_size=min(int(args.group_size), int(args.factor)),
    )
    inputs = _make_inputs(
        shape,
        dtype=args.dtype,
        weight_dtype=args.weight_dtype,
        seed=args.seed,
        w1_scale=args.w1_scale,
        w2_scale=args.w2_scale,
        gate_value=args.gate_value,
    )

    run_dir = make_run_dir(
        "lokr_grouped_delta",
        label=args.label,
        root=args.output_root,
    )

    diff_metrics = _forward_diff(
        shape,
        inputs,
        chunk_bytes=args.chunk_bytes,
        backward_backend=args.backward_backend,
    )
    forward_backend_metrics: dict[str, Any] = {}
    backward_backend_metrics: dict[str, Any] = {}
    artifacts: list[str] = []

    for backend in args.backends:
        forward_timing = _time_forward_ms(
            backend,
            inputs,
            shape,
            chunk_bytes=args.chunk_bytes,
            backward_backend=args.backward_backend,
            warmup=args.warmup,
            iters=args.iters,
        )
        forward_trace_name = f"{backend}_forward_trace.json"
        forward_trace_path = run_dir / forward_trace_name
        forward_trace_metrics = _profile_forward_trace(
            backend,
            inputs,
            shape,
            chunk_bytes=args.chunk_bytes,
            backward_backend=args.backward_backend,
            trace_path=forward_trace_path,
        )
        forward_backend_metrics[backend] = {
            **forward_timing,
            **forward_trace_metrics,
            "launch_density_per_ms": round(
                forward_trace_metrics["launch_count"]
                / max(forward_timing["avg_ms"], 1e-9),
                6,
            ),
            "kernel_density_per_ms": round(
                forward_trace_metrics["kernel_count"]
                / max(forward_timing["avg_ms"], 1e-9),
                6,
            ),
        }
        artifacts.append(forward_trace_name)

        backward_timing = _time_forward_backward_ms(
            backend,
            inputs,
            shape,
            chunk_bytes=args.chunk_bytes,
            backward_backend=args.backward_backend,
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed,
        )
        backward_trace_name = f"{backend}_forward_backward_trace.json"
        backward_trace_path = run_dir / backward_trace_name
        backward_trace_metrics = _profile_forward_backward_trace(
            backend,
            inputs,
            shape,
            chunk_bytes=args.chunk_bytes,
            backward_backend=args.backward_backend,
            trace_path=backward_trace_path,
            seed=args.seed,
        )

        forward_avg_ms = backward_timing["forward_ms"]["avg_ms"]
        backward_avg_ms = backward_timing["backward_ms"]["avg_ms"]
        total_avg_ms = backward_timing["total_ms"]["avg_ms"]
        backward_backend_metrics[backend] = {
            **backward_timing,
            "trace": backward_trace_metrics,
            "forward_launch_density_per_ms": round(
                backward_trace_metrics["forward"]["launch_count"]
                / max(forward_avg_ms, 1e-9),
                6,
            ),
            "forward_kernel_density_per_ms": round(
                backward_trace_metrics["forward"]["kernel_count"]
                / max(forward_avg_ms, 1e-9),
                6,
            ),
            "backward_launch_density_per_ms": round(
                backward_trace_metrics["backward"]["launch_count"]
                / max(backward_avg_ms, 1e-9),
                6,
            ),
            "backward_kernel_density_per_ms": round(
                backward_trace_metrics["backward"]["kernel_count"]
                / max(backward_avg_ms, 1e-9),
                6,
            ),
            "total_launch_density_per_ms": round(
                backward_trace_metrics["all"]["launch_count"]
                / max(total_avg_ms, 1e-9),
                6,
            ),
            "total_kernel_density_per_ms": round(
                backward_trace_metrics["all"]["kernel_count"]
                / max(total_avg_ms, 1e-9),
                6,
            ),
        }
        artifacts.append(backward_trace_name)

    forward_comparison = None
    if "eager" in forward_backend_metrics and "triton" in forward_backend_metrics:
        eager = forward_backend_metrics["eager"]
        triton = forward_backend_metrics["triton"]
        forward_comparison = {
            "forward_latency_drop_pct": round(
                (eager["avg_ms"] - triton["avg_ms"]) / eager["avg_ms"] * 100.0,
                3,
            ),
            "launch_drop_pct": round(
                (eager["launch_count"] - triton["launch_count"])
                / max(eager["launch_count"], 1)
                * 100.0,
                3,
            ),
            "kernel_drop_pct": round(
                (eager["kernel_count"] - triton["kernel_count"])
                / max(eager["kernel_count"], 1)
                * 100.0,
                3,
            ),
            "small_kernel_lt50us_drop_pct": round(
                (
                    eager["small_kernel_counts"]["50"]
                    - triton["small_kernel_counts"]["50"]
                )
                / max(eager["small_kernel_counts"]["50"], 1)
                * 100.0,
                3,
            ),
        }

    backward_comparison = None
    if "eager" in backward_backend_metrics and "triton" in backward_backend_metrics:
        eager = backward_backend_metrics["eager"]
        triton = backward_backend_metrics["triton"]
        backward_comparison = {
            "forward_ms_drop_pct": round(
                (
                    eager["forward_ms"]["avg_ms"] - triton["forward_ms"]["avg_ms"]
                )
                / eager["forward_ms"]["avg_ms"]
                * 100.0,
                3,
            ),
            "backward_ms_drop_pct": round(
                (
                    eager["backward_ms"]["avg_ms"] - triton["backward_ms"]["avg_ms"]
                )
                / eager["backward_ms"]["avg_ms"]
                * 100.0,
                3,
            ),
            "total_ms_drop_pct": round(
                (eager["total_ms"]["avg_ms"] - triton["total_ms"]["avg_ms"])
                / eager["total_ms"]["avg_ms"]
                * 100.0,
                3,
            ),
            "eager_backward_share_pct": eager["backward_share_pct"],
            "triton_backward_share_pct": triton["backward_share_pct"],
            "eager_backward_launch_share_pct": round(
                eager["trace"]["backward"]["launch_count"]
                / max(eager["trace"]["all"]["launch_count"], 1)
                * 100.0,
                3,
            ),
            "triton_backward_launch_share_pct": round(
                triton["trace"]["backward"]["launch_count"]
                / max(triton["trace"]["all"]["launch_count"], 1)
                * 100.0,
                3,
            ),
            "peak_allocated_delta_gb_diff": round(
                triton["peak_allocated_delta_gb"]["avg_gb"]
                - eager["peak_allocated_delta_gb"]["avg_gb"],
                6,
            ),
            "peak_reserved_delta_gb_diff": round(
                triton["peak_reserved_delta_gb"]["avg_gb"]
                - eager["peak_reserved_delta_gb"]["avg_gb"],
                6,
            ),
        }

    metrics = {
        "shape": {
            "outer_shape": shape.outer_shape,
            "rows": shape.rows,
            "factor": shape.factor,
            "group_size": shape.group_size,
            "in_features": shape.in_features,
            "out_features": shape.out_features,
            "in_dim": shape.in_dim,
            "out_dim": shape.out_dim,
        },
        "dtypes": {"activation": args.dtype, "weight": args.weight_dtype},
        "backend_config": {
            "forward_backends": list(args.backends),
            "backward_backend": args.backward_backend,
        },
        "input_scale": {
            "w1_scale": args.w1_scale,
            "w2_scale": args.w2_scale,
            "gate_value": args.gate_value,
        },
        "forward_diff_vs_eager": diff_metrics,
        "forward_only": {
            "backends": forward_backend_metrics,
            "comparison": forward_comparison,
        },
        "forward_backward": {
            "backends": backward_backend_metrics,
            "comparison": backward_comparison,
        },
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    artifacts.append(summary_path.name)

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=artifacts,
        device="cuda:0",
        extra={"gpu_rows": _gpu_rows()},
    )

    print(json.dumps(metrics, indent=2))
    print(f"run_dir: {run_dir}")


if __name__ == "__main__":
    main()
