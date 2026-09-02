#!/usr/bin/env python
"""Compare LoKr grouped-delta backward gradients against the eager backend."""

from __future__ import annotations

import argparse
import json

from bench._common import make_run_dir, write_result
from bench.lokr_grouped_delta.run_microbench import (
    DEFAULT_DTYPE,
    DEFAULT_GPU_INDEX,
    DEFAULT_ROOT,
    DEFAULT_SEED,
    DEFAULT_WEIGHT_DTYPE,
    BenchShape,
    _check_gpu,
    _make_backward_state,
    _make_inputs,
    _parse_shape,
    _run_forward_backward_once,
    _set_cuda_env,
)


DEFAULT_CANDIDATE = "triton_grad_w1_w2_grad_x"


def _tensor_error_metrics(actual, reference) -> dict[str, float | bool]:
    import torch

    actual_float = actual.float()
    reference_float = reference.float()
    diff = actual_float - reference_float
    reference_norm = torch.linalg.vector_norm(reference_float)
    actual_norm = torch.linalg.vector_norm(actual_float)
    denominator = reference_float.abs().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        actual_float.reshape(1, -1),
        reference_float.reshape(1, -1),
    )[0]
    return {
        "finite": bool(torch.isfinite(actual_float).all().item()),
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "rms_abs": float(diff.square().mean().sqrt().item()),
        "max_rel": float((diff.abs() / denominator).max().item()),
        "relative_l2": float(
            (torch.linalg.vector_norm(diff) / reference_norm.clamp_min(1e-12)).item()
        ),
        "cosine": float(cosine.item()),
        "reference_l2": float(reference_norm.item()),
        "actual_l2": float(actual_norm.item()),
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
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--w1-scale", type=float, default=1.0)
    parser.add_argument("--w2-scale", type=float, default=1e-3)
    parser.add_argument("--gate-value", type=float, default=0.75)
    parser.add_argument("--candidate-backend", default=DEFAULT_CANDIDATE)
    parser.add_argument("--reference-backend", default="eager")
    parser.add_argument("--label", default="backward-parity")
    parser.add_argument("--output-root", default=DEFAULT_ROOT)
    args = parser.parse_args()

    if args.in_features % args.factor or args.out_features % args.factor:
        raise SystemExit("in/out features must be divisible by factor")
    _check_gpu(args)
    _set_cuda_env(args.gpu_index)

    import torch
    from networks.plugins.lokr.autograd import (
        normalize_lokr_grouped_delta_backward_backend,
    )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    candidate_backend = normalize_lokr_grouped_delta_backward_backend(
        args.candidate_backend
    )
    reference_backend = normalize_lokr_grouped_delta_backward_backend(
        args.reference_backend
    )
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
    candidate_state = _make_backward_state(inputs, seed=args.seed)
    reference_state = _make_backward_state(inputs, seed=args.seed)

    candidate_output = _run_forward_backward_once(
        "triton",
        candidate_state,
        shape,
        args.chunk_bytes,
        backward_backend=candidate_backend,
    )
    reference_output = _run_forward_backward_once(
        "triton",
        reference_state,
        shape,
        args.chunk_bytes,
        backward_backend=reference_backend,
    )
    torch.cuda.synchronize()

    metrics = {
        "output": _tensor_error_metrics(candidate_output, reference_output),
        "grad_x": _tensor_error_metrics(
            candidate_state["x"].grad, reference_state["x"].grad
        ),
        "grad_w1": _tensor_error_metrics(
            candidate_state["w1"].grad, reference_state["w1"].grad
        ),
        "grad_w2": _tensor_error_metrics(
            candidate_state["w2"].grad, reference_state["w2"].grad
        ),
    }
    run_dir = make_run_dir(
        "lokr_grouped_delta_backward_parity",
        label=args.label,
        root=args.output_root,
    )
    result_path = write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        device=torch.device("cuda:0"),
        extra={
            "physical_gpu_index": args.gpu_index,
            "shape": {
                "outer_shape": shape.outer_shape,
                "rows": shape.rows,
                "factor": shape.factor,
                "group_size": shape.group_size,
                "in_dim": shape.in_dim,
                "out_dim": shape.out_dim,
            },
            "candidate_backend": candidate_backend,
            "reference_backend": reference_backend,
        },
    )
    print(json.dumps(metrics, indent=2))
    print(f"result: {result_path}")


if __name__ == "__main__":
    main()
