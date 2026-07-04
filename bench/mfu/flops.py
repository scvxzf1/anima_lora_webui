"""Anima DiT training FLOPs / MFU estimator.

This module intentionally models the dominant DiT path only:

- patch embedding
- 28x transformer blocks
- final projection

It uses tensor shapes recovered from the peak-probe JSONL stream together with
known Anima architecture constants. The result is an estimate suitable for
relative benchmarking across training variants, not a cycle-accurate profiler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_FORWARD_BACKWARD_MULTIPLIER = 3.0


@dataclass(frozen=True)
class AnimaModelSpec:
    """Fixed architectural constants for the shipped Anima DiT."""

    model_channels: int = 2048
    num_blocks: int = 28
    num_heads: int = 16
    cross_attn_dim: int = 1024
    cross_attn_tokens: int = 512
    mlp_ratio: float = 4.0
    in_channels: int = 16
    concat_padding_mask: bool = True
    out_channels: int = 16
    patch_spatial: int = 2
    patch_temporal: int = 1
    forward_backward_multiplier: float = DEFAULT_FORWARD_BACKWARD_MULTIPLIER

    @property
    def patch_in_channels(self) -> int:
        return self.in_channels + (1 if self.concat_padding_mask else 0)

    @property
    def patch_volume(self) -> int:
        return self.patch_spatial * self.patch_spatial * self.patch_temporal

    @property
    def mlp_hidden_dim(self) -> int:
        return int(self.model_channels * self.mlp_ratio)

    @property
    def head_dim(self) -> int:
        return self.model_channels // self.num_heads


@dataclass(frozen=True)
class StepShape:
    """Recovered token shape for one measured training step."""

    batch_size: int
    time_patches: int
    height_patches: int
    width_patches: int

    @property
    def token_count(self) -> int:
        return self.time_patches * self.height_patches * self.width_patches


def _require_positive_int(name: str, value: Any) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return ivalue


def parse_step_shape_from_peak_probe_event(event: dict[str, Any]) -> StepShape:
    """Parse a `block_before`/`block_after` probe event into a token shape."""

    shape = event.get("tensor_shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 5:
        raise ValueError(f"expected 5D tensor_shape, got {shape!r}")
    b, t, h, w, _d = shape
    return StepShape(
        batch_size=_require_positive_int("batch_size", b),
        time_patches=_require_positive_int("time_patches", t),
        height_patches=_require_positive_int("height_patches", h),
        width_patches=_require_positive_int("width_patches", w),
    )


def _matmul_flops(m: int, k: int, n: int) -> int:
    """FLOPs for a dense GEMM using the common 2*m*k*n convention."""

    return 2 * int(m) * int(k) * int(n)


def patch_embed_forward_flops(shape: StepShape, spec: AnimaModelSpec) -> int:
    return shape.batch_size * _matmul_flops(
        shape.token_count,
        spec.patch_in_channels * spec.patch_volume,
        spec.model_channels,
    )


def self_attn_forward_flops(shape: StepShape, spec: AnimaModelSpec) -> int:
    b = shape.batch_size
    s = shape.token_count
    d = spec.model_channels
    h = spec.num_heads
    qkv = _matmul_flops(b * s, d, 3 * d)
    scores = 2 * b * h * s * s * spec.head_dim
    weighted_sum = 2 * b * h * s * s * spec.head_dim
    out_proj = _matmul_flops(b * s, d, d)
    return qkv + scores + weighted_sum + out_proj


def cross_attn_forward_flops(shape: StepShape, spec: AnimaModelSpec) -> int:
    b = shape.batch_size
    s = shape.token_count
    c = spec.cross_attn_tokens
    d = spec.model_channels
    h = spec.num_heads
    ctx = spec.cross_attn_dim
    q_proj = _matmul_flops(b * s, d, d)
    kv_proj = _matmul_flops(b * c, ctx, 2 * d)
    scores = 2 * b * h * s * c * spec.head_dim
    weighted_sum = 2 * b * h * s * c * spec.head_dim
    out_proj = _matmul_flops(b * s, d, d)
    return q_proj + kv_proj + scores + weighted_sum + out_proj


def mlp_forward_flops(shape: StepShape, spec: AnimaModelSpec) -> int:
    b = shape.batch_size
    s = shape.token_count
    d = spec.model_channels
    hdim = spec.mlp_hidden_dim
    return _matmul_flops(b * s, d, hdim) + _matmul_flops(b * s, hdim, d)


def final_layer_forward_flops(shape: StepShape, spec: AnimaModelSpec) -> int:
    out_dim = spec.out_channels * spec.patch_volume
    return shape.batch_size * _matmul_flops(shape.token_count, spec.model_channels, out_dim)


def per_block_forward_flops(shape: StepShape, spec: AnimaModelSpec) -> int:
    return (
        self_attn_forward_flops(shape, spec)
        + cross_attn_forward_flops(shape, spec)
        + mlp_forward_flops(shape, spec)
    )


def total_forward_flops(shape: StepShape, spec: AnimaModelSpec) -> int:
    return (
        patch_embed_forward_flops(shape, spec)
        + spec.num_blocks * per_block_forward_flops(shape, spec)
        + final_layer_forward_flops(shape, spec)
    )


def total_train_step_flops(shape: StepShape, spec: AnimaModelSpec) -> float:
    """Estimated forward+backward FLOPs for one optimizer step."""

    return float(total_forward_flops(shape, spec)) * float(spec.forward_backward_multiplier)


def estimate_mfu(
    *,
    shape: StepShape,
    avg_step_sec: float,
    peak_flops: float,
    spec: AnimaModelSpec,
) -> float | None:
    if avg_step_sec <= 0 or peak_flops <= 0:
        return None
    achieved = total_train_step_flops(shape, spec) / float(avg_step_sec)
    return achieved / float(peak_flops)
