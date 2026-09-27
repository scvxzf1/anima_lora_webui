#!/usr/bin/env python3
"""Measure Qwen 2.1 image attention backends with CUDA events.

This is intentionally a standalone eager attention probe.  It does not load
the DiT or alter the production backend; the full training windows are kept in
the neighbouring phase2b run directories.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Callable

import torch
from torch.backends.cuda import SDPAParams
from torch.nn.attention import SDPBackend, sdpa_kernel


def _stats(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "count": len(samples),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _measure(
    fn: Callable[[], torch.Tensor], *, warmup: int, iters: int, label: str
) -> dict[str, object]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    forward_samples: list[float] = []
    backward_samples: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        middle = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.autograd.profiler.record_function(f"phase2b::{label}::forward"):
            output = fn()
        middle.record()
        with torch.autograd.profiler.record_function(f"phase2b::{label}::backward"):
            output.float().sum().backward()
        end.record()
        torch.cuda.synchronize()
        forward_samples.append(float(start.elapsed_time(middle)))
        backward_samples.append(float(middle.elapsed_time(end)))

    return {
        "forward": _stats(forward_samples),
        "backward": _stats(backward_samples),
        "forward_backward": _stats(
            [forward + backward for forward, backward in zip(forward_samples, backward_samples)]
        ),
    }


def _make_inputs(
    *, batch: int, query_length: int, key_length: int, heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shape_q = (batch, query_length, heads, head_dim)
    shape_kv = (batch, key_length, heads, head_dim)
    query = torch.randn(shape_q, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(shape_kv, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    value = torch.randn(shape_kv, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    mask = torch.ones((batch, 1, 1, key_length), device="cuda", dtype=torch.bool)
    return query, key, value, mask


def _native_fn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    backend: SDPBackend | None,
) -> Callable[[], torch.Tensor]:
    def run() -> torch.Tensor:
        if backend is None:
            output = torch.nn.functional.scaled_dot_product_attention(
                query.permute(0, 2, 1, 3),
                key.permute(0, 2, 1, 3),
                value.permute(0, 2, 1, 3),
                attn_mask=mask,
                dropout_p=0.0,
            )
        else:
            with sdpa_kernel([backend]):
                output = torch.nn.functional.scaled_dot_product_attention(
                    query.permute(0, 2, 1, 3),
                    key.permute(0, 2, 1, 3),
                    value.permute(0, 2, 1, 3),
                    attn_mask=mask,
                    dropout_p=0.0,
                )
        return output.permute(0, 2, 1, 3)

    return run


def _flash_fn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> Callable[[], torch.Tensor]:
    from networks import attention_dispatch

    batch, query_length, heads, head_dim = query.shape
    key_length = key.shape[1]
    cu_q = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * query_length
    cu_k = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * key_length
    q_packed = query.reshape(batch * query_length, heads, head_dim)
    k_packed = key.reshape(batch * key_length, heads, head_dim)
    v_packed = value.reshape(batch * key_length, heads, head_dim)

    def run() -> torch.Tensor:
        output = attention_dispatch.flash_attn_varlen_func(
            q_packed,
            k_packed,
            v_packed,
            cu_q,
            cu_k,
            query_length,
            key_length,
            dropout_p=0.0,
        )
        return output.reshape(batch, query_length, heads, head_dim)

    return run


def _qwen_flash_wrapper_fn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
) -> Callable[[], torch.Tensor]:
    from library.models.qwen_image_2_1.attention_backend import _flash_varlen_attention

    def run() -> torch.Tensor:
        return _flash_varlen_attention(query, key, value, mask=mask)

    return run


def _qwen_cached_full_wrapper_fn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> Callable[[], torch.Tensor]:
    """Prototype the fixed all-valid metadata path used by the training bucket."""
    from networks import attention_dispatch

    batch, query_length, heads, head_dim = query.shape
    key_length = key.shape[1]
    cu_q = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * query_length
    cu_k = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * key_length
    max_key_length = key_length

    def run() -> torch.Tensor:
        output = attention_dispatch.flash_attn_varlen_func(
            query.reshape(batch * query_length, heads, head_dim),
            key.reshape(batch * key_length, heads, head_dim),
            value.reshape(batch * key_length, heads, head_dim),
            cu_q,
            cu_k,
            query_length,
            max_key_length,
            dropout_p=0.0,
        )
        return output.reshape(batch, query_length, heads, head_dim)

    return run


def _backend_capabilities(query, key, value, mask) -> dict[str, object]:
    params = SDPAParams(
        query.permute(0, 2, 1, 3),
        key.permute(0, 2, 1, 3),
        value.permute(0, 2, 1, 3),
        mask,
        0.0,
        False,
        False,
    )
    return {
        "cudnn": torch.backends.cuda.can_use_cudnn_attention(params, debug=False),
        "flash": torch.backends.cuda.can_use_flash_attention(params, debug=False),
        "efficient": torch.backends.cuda.can_use_efficient_attention(params, debug=False),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.cuda.set_device(0)
    query, key, value, mask = _make_inputs(
        batch=1, query_length=4032, key_length=4107, heads=24, head_dim=128
    )
    results: dict[str, object] = {}
    candidates: list[tuple[str, Callable[[], torch.Tensor]]] = [
        ("native_default", _native_fn(query, key, value, mask, None)),
        ("native_cudnn", _native_fn(query, key, value, mask, SDPBackend.CUDNN_ATTENTION)),
        ("native_efficient", _native_fn(query, key, value, mask, SDPBackend.EFFICIENT_ATTENTION)),
        ("flash_varlen", _flash_fn(query, key, value)),
        ("qwen_flash_wrapper", _qwen_flash_wrapper_fn(query, key, value, mask)),
        ("qwen_cached_full_prototype", _qwen_cached_full_wrapper_fn(query, key, value)),
    ]
    for name, fn in candidates:
        for tensor in (query, key, value):
            tensor.grad = None
        try:
            results[name] = _measure(fn, warmup=args.warmup, iters=args.iters, label=name)
        except Exception as exc:  # Keep unsupported forced backends in the artifact.
            torch.cuda.synchronize()
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}

    payload = {
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "flash_attn": os.environ.get("FLASH_ATTENTION_VERSION", "installed extension"),
        },
        "shape": {"batch": 1, "query_length": 4032, "key_length": 4107, "heads": 24, "head_dim": 128},
        "mask": {"dtype": str(mask.dtype), "valid_tokens": int(mask.sum().item())},
        "sdpa_capabilities": _backend_capabilities(query, key, value, mask),
        "warmup": args.warmup,
        "iterations": args.iters,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
