#!/usr/bin/env python
"""Run a four-cell LoKr grouped-delta ablation matrix.

The matrix is intentionally small and direct:

- small/before: rows below the suspected fusion threshold, eager path
- small/after:  rows below the suspected fusion threshold, fused Triton path
- big/before:   rows at/above the suspected fusion threshold, eager path
- big/after:    rows at/above the suspected fusion threshold, fused Triton path

It reuses ``run_microbench`` timing helpers but skips profiler traces by
default, so this can be used as a quick hot-test before running full training.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench._common import make_run_dir, write_result
from bench.lokr_grouped_delta.run_microbench import (
    DEFAULT_BACKWARD_BACKEND,
    DEFAULT_DTYPE,
    DEFAULT_GPU_INDEX,
    DEFAULT_ITERS,
    DEFAULT_ROOT,
    DEFAULT_SEED,
    DEFAULT_WARMUP,
    DEFAULT_WEIGHT_DTYPE,
    BenchShape,
    _check_gpu,
    _forward_diff,
    _gpu_rows,
    _make_inputs,
    _parse_backends,
    _set_cuda_env,
    _time_forward_backward_ms,
    _time_forward_ms,
)

DEFAULT_SMALL_ROWS = 1024
DEFAULT_BIG_ROWS = 2048
DEFAULT_IN_FEATURES = 2048
DEFAULT_OUT_FEATURES = 8192
DEFAULT_FACTOR = 8
DEFAULT_GROUP_SIZE = 8
DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024
DEFAULT_BEFORE_BACKEND = "eager"
DEFAULT_AFTER_BACKEND = "triton"


@dataclass(frozen=True)
class MatrixCell:
    size: str
    variant: str
    backend: str
    rows: int

    @property
    def key(self) -> str:
        return f"{self.size}_{self.variant}"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _drop_pct(before: float, after: float) -> float:
    if before <= 0:
        return 0.0
    return round((before - after) / before * 100.0, 3)


def _diff_value(after: float, before: float) -> float:
    return round(after - before, 6)


def _shape_for_rows(args: argparse.Namespace, rows: int) -> BenchShape:
    return BenchShape(
        outer_shape=(int(rows),),
        factor=int(args.factor),
        in_features=int(args.in_features),
        out_features=int(args.out_features),
        group_size=min(int(args.group_size), int(args.factor)),
    )


def _run_cell(args: argparse.Namespace, cell: MatrixCell) -> dict[str, Any]:
    shape = _shape_for_rows(args, cell.rows)
    inputs = _make_inputs(
        shape,
        dtype=args.dtype,
        weight_dtype=args.weight_dtype,
        seed=args.seed,
        w1_scale=args.w1_scale,
        w2_scale=args.w2_scale,
        gate_value=args.gate_value,
    )
    forward = _time_forward_ms(
        cell.backend,
        inputs,
        shape,
        chunk_bytes=args.chunk_bytes,
        backward_backend=args.backward_backend,
        warmup=args.warmup,
        iters=args.iters,
    )
    forward_backward = None
    if not args.forward_only:
        forward_backward = _time_forward_backward_ms(
            cell.backend,
            inputs,
            shape,
            chunk_bytes=args.chunk_bytes,
            backward_backend=args.backward_backend,
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed,
        )
    return {
        "cell": {
            "key": cell.key,
            "size": cell.size,
            "variant": cell.variant,
            "backend": cell.backend,
        },
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
        "forward_only": forward,
        "forward_backward": forward_backward,
    }


def _run_diffs(args: argparse.Namespace, rows_by_size: dict[str, int]) -> dict[str, Any]:
    diffs: dict[str, Any] = {}
    if not ({args.before_backend, args.after_backend} <= {"eager", "triton"}):
        return diffs
    for size, rows in rows_by_size.items():
        shape = _shape_for_rows(args, rows)
        inputs = _make_inputs(
            shape,
            dtype=args.dtype,
            weight_dtype=args.weight_dtype,
            seed=args.seed,
            w1_scale=args.w1_scale,
            w2_scale=args.w2_scale,
            gate_value=args.gate_value,
        )
        diffs[size] = _forward_diff(
            shape,
            inputs,
            chunk_bytes=args.chunk_bytes,
            backward_backend=args.backward_backend,
        )
    return diffs


def _build_comparisons(
    cells: dict[str, dict[str, Any]],
    *,
    small_key: str = "small",
    big_key: str = "big",
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for size in (small_key, big_key):
        before = cells[f"{size}_before"]
        after = cells[f"{size}_after"]
        item: dict[str, Any] = {
            "forward_latency_drop_pct": _drop_pct(
                before["forward_only"]["avg_ms"],
                after["forward_only"]["avg_ms"],
            ),
            "forward_avg_ms_before": before["forward_only"]["avg_ms"],
            "forward_avg_ms_after": after["forward_only"]["avg_ms"],
        }
        before_fb = before.get("forward_backward")
        after_fb = after.get("forward_backward")
        if before_fb is not None and after_fb is not None:
            item.update(
                {
                    "forward_backward_total_drop_pct": _drop_pct(
                        before_fb["total_ms"]["avg_ms"],
                        after_fb["total_ms"]["avg_ms"],
                    ),
                    "forward_backward_forward_drop_pct": _drop_pct(
                        before_fb["forward_ms"]["avg_ms"],
                        after_fb["forward_ms"]["avg_ms"],
                    ),
                    "forward_backward_backward_drop_pct": _drop_pct(
                        before_fb["backward_ms"]["avg_ms"],
                        after_fb["backward_ms"]["avg_ms"],
                    ),
                    "total_avg_ms_before": before_fb["total_ms"]["avg_ms"],
                    "total_avg_ms_after": after_fb["total_ms"]["avg_ms"],
                    "peak_allocated_delta_gb_diff": _diff_value(
                        after_fb["peak_allocated_delta_gb"]["avg_gb"],
                        before_fb["peak_allocated_delta_gb"]["avg_gb"],
                    ),
                    "peak_reserved_delta_gb_diff": _diff_value(
                        after_fb["peak_reserved_delta_gb"]["avg_gb"],
                        before_fb["peak_reserved_delta_gb"]["avg_gb"],
                    ),
                }
            )
        comparisons[size] = item

    comparisons["big_minus_small"] = {
        "forward_latency_drop_pct_delta": round(
            comparisons[big_key]["forward_latency_drop_pct"]
            - comparisons[small_key]["forward_latency_drop_pct"],
            3,
        )
    }
    if "forward_backward_total_drop_pct" in comparisons[big_key]:
        comparisons["big_minus_small"]["forward_backward_total_drop_pct_delta"] = round(
            comparisons[big_key]["forward_backward_total_drop_pct"]
            - comparisons[small_key]["forward_backward_total_drop_pct"],
            3,
        )
    return comparisons


def _write_matrix_csv(
    path: Path,
    cells: dict[str, dict[str, Any]],
    comparisons: dict[str, Any],
) -> None:
    fieldnames = [
        "cell",
        "size",
        "variant",
        "backend",
        "rows",
        "forward_avg_ms",
        "total_avg_ms",
        "forward_drop_pct_for_size",
        "total_drop_pct_for_size",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in ("small_before", "small_after", "big_before", "big_after"):
            item = cells[key]
            size = item["cell"]["size"]
            fb = item.get("forward_backward")
            writer.writerow(
                {
                    "cell": key,
                    "size": size,
                    "variant": item["cell"]["variant"],
                    "backend": item["cell"]["backend"],
                    "rows": item["shape"]["rows"],
                    "forward_avg_ms": item["forward_only"]["avg_ms"],
                    "total_avg_ms": "" if fb is None else fb["total_ms"]["avg_ms"],
                    "forward_drop_pct_for_size": (
                        comparisons[size]["forward_latency_drop_pct"]
                        if item["cell"]["variant"] == "after"
                        else ""
                    ),
                    "total_drop_pct_for_size": (
                        comparisons[size].get("forward_backward_total_drop_pct", "")
                        if item["cell"]["variant"] == "after"
                        else ""
                    ),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-index", default=DEFAULT_GPU_INDEX)
    parser.add_argument(
        "--allow-low-vram", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--min-vram-mb", type=int, default=8000)
    parser.add_argument("--small-rows", type=_positive_int, default=DEFAULT_SMALL_ROWS)
    parser.add_argument("--big-rows", type=_positive_int, default=DEFAULT_BIG_ROWS)
    parser.add_argument("--in-features", type=_positive_int, default=DEFAULT_IN_FEATURES)
    parser.add_argument(
        "--out-features", type=_positive_int, default=DEFAULT_OUT_FEATURES
    )
    parser.add_argument("--factor", type=_positive_int, default=DEFAULT_FACTOR)
    parser.add_argument("--group-size", type=_positive_int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--chunk-bytes", type=_positive_int, default=DEFAULT_CHUNK_BYTES)
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
    parser.add_argument("--before-backend", default=DEFAULT_BEFORE_BACKEND)
    parser.add_argument("--after-backend", default=DEFAULT_AFTER_BACKEND)
    parser.add_argument(
        "--extra-backends",
        type=_parse_backends,
        default=(),
        help="Optional extra backends to time outside the four required cells.",
    )
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
    parser.add_argument(
        "--forward-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip backward timing for a faster four-cell smoke.",
    )
    parser.add_argument("--label", default="xw_w1_ablation")
    parser.add_argument(
        "--output-root",
        default=os.path.join(DEFAULT_ROOT, "ablation_matrix"),
    )
    args = parser.parse_args()

    if args.small_rows >= args.big_rows:
        raise SystemExit("small-rows must be smaller than big-rows")
    if args.in_features % args.factor or args.out_features % args.factor:
        raise SystemExit("in/out features must be divisible by factor")

    _check_gpu(args)
    _set_cuda_env(args.gpu_index)

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    rows_by_size = {"small": int(args.small_rows), "big": int(args.big_rows)}
    cells_to_run = [
        MatrixCell("small", "before", args.before_backend, rows_by_size["small"]),
        MatrixCell("small", "after", args.after_backend, rows_by_size["small"]),
        MatrixCell("big", "before", args.before_backend, rows_by_size["big"]),
        MatrixCell("big", "after", args.after_backend, rows_by_size["big"]),
    ]

    run_dir = make_run_dir(
        "lokr_grouped_delta",
        label=args.label,
        root=args.output_root,
    )

    cells: dict[str, dict[str, Any]] = {}
    for cell in cells_to_run:
        cells[cell.key] = _run_cell(args, cell)

    extra: dict[str, Any] = {}
    for backend in tuple(args.extra_backends):
        for size, rows in rows_by_size.items():
            cell = MatrixCell(size, f"extra_{backend}", backend, rows)
            extra[cell.key] = _run_cell(args, cell)

    comparisons = _build_comparisons(cells)
    metrics = {
        "matrix": {
            "small_rows": args.small_rows,
            "big_rows": args.big_rows,
            "before_backend": args.before_backend,
            "after_backend": args.after_backend,
            "forward_only": args.forward_only,
        },
        "shape_base": {
            "factor": args.factor,
            "group_size": min(args.group_size, args.factor),
            "in_features": args.in_features,
            "out_features": args.out_features,
            "in_dim": args.in_features // args.factor,
            "out_dim": args.out_features // args.factor,
        },
        "dtypes": {"activation": args.dtype, "weight": args.weight_dtype},
        "backend_config": {
            "backward_backend": args.backward_backend,
            "extra_backends": list(args.extra_backends),
        },
        "input_scale": {
            "w1_scale": args.w1_scale,
            "w2_scale": args.w2_scale,
            "gate_value": args.gate_value,
        },
        "cells": cells,
        "extra_cells": extra,
        "comparison": comparisons,
        "forward_diff_vs_eager": _run_diffs(args, rows_by_size),
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    csv_path = run_dir / "matrix.csv"
    _write_matrix_csv(csv_path, cells, comparisons)

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=[summary_path.name, csv_path.name],
        device="cuda:0",
        extra={"gpu_rows": _gpu_rows()},
    )

    print(json.dumps(metrics["comparison"], indent=2))
    print(f"summary: {summary_path}")
    print(f"matrix: {csv_path}")


if __name__ == "__main__":
    main()
