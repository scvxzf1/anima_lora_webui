"""Experimental opaque native LayerNorm kernels for first-order Qwen NF4 VJPs.

This diagnostic preserves native forward and backward calls inside fullgraph
compilation. It is not a production compile policy and does not support double
backward. Only non-affine img_norm1/img_norm2 LayerNorm forwards are patched.
"""
from __future__ import annotations

from types import MethodType

import torch


FORWARD_OP = "qwen_nf4_diagnostic::native_layer_norm_forward"
BACKWARD_OP = "qwen_nf4_diagnostic::native_layer_norm_backward"


@torch.library.custom_op(FORWARD_OP, mutates_args=())
def native_layer_norm_forward(
    input: torch.Tensor, normalized_shape: list[int], eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.native_layer_norm(input, normalized_shape, None, None, eps)


@native_layer_norm_forward.register_fake
def _forward_fake(input, normalized_shape, eps):
    stats_shape = (*input.shape[:-len(normalized_shape)], *([1] * len(normalized_shape)))
    # CUDA native LayerNorm accumulates low-precision statistics in FP32;
    # CPU statistics keep the input dtype when there is no affine weight.
    stats_dtype = (
        torch.float32
        if input.device.type == "cuda" and input.dtype in (torch.float16, torch.bfloat16)
        else input.dtype
    )
    return (
        input.new_empty(input.shape),
        input.new_empty(stats_shape, dtype=stats_dtype),
        input.new_empty(stats_shape, dtype=stats_dtype),
    )


@torch.library.custom_op(BACKWARD_OP, mutates_args=())
def native_layer_norm_backward(
    gradient: torch.Tensor, input: torch.Tensor, normalized_shape: list[int],
    mean: torch.Tensor, rstd: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.aten.native_layer_norm_backward(
        gradient, input, normalized_shape, mean, rstd, None, None, [True, False, False],
    )[0]


@native_layer_norm_backward.register_fake
def _backward_fake(gradient, input, normalized_shape, mean, rstd):
    return input.new_empty(input.shape)


def _setup_context(ctx, inputs, output):
    input, normalized_shape, _eps = inputs
    _normalized, mean, rstd = output
    ctx.save_for_backward(input, mean, rstd)
    ctx.normalized_shape = normalized_shape
    ctx.mark_non_differentiable(mean, rstd)


def _backward(ctx, gradient, _mean_gradient, _rstd_gradient):
    input, mean, rstd = ctx.saved_tensors
    return native_layer_norm_backward(gradient, input, ctx.normalized_shape, mean, rstd), None, None


native_layer_norm_forward.register_autograd(_backward, setup_context=_setup_context)


def _native_forward(module, input):
    # LayerNorm's CUDA autocast policy is FP32; CPU and non-autocast inputs keep
    # their existing dtype. The cast remains visible to AOTAutograd/Inductor.
    if input.is_cuda and torch.is_autocast_enabled("cuda"):
        input = input.float()
    return native_layer_norm_forward(input, list(module.normalized_shape), module.eps)[0]


def _validate_norm(module, path):
    if not isinstance(module, torch.nn.LayerNorm):
        raise TypeError(f"Native LayerNorm diagnostic requires nn.LayerNorm: {path}")
    if module.elementwise_affine or module.weight is not None or module.bias is not None:
        raise ValueError(f"Native LayerNorm diagnostic requires non-affine LayerNorm: {path}")
    shape = tuple(module.normalized_shape)
    if not shape or any(not isinstance(size, int) or size <= 0 for size in shape):
        raise ValueError(f"Native LayerNorm diagnostic requires a positive normalized_shape: {path}")


def validate_native_layer_norm(model, *, norm_names=("img_norm1", "img_norm2")):
    """Collect and validate targets without changing forwards."""
    return _norm_targets(model, norm_names)


def _norm_targets(model, norm_names):
    """Patch Qwen image norms only, leaving parameters and module structure intact.

    Validate every target before modifying any forward. The returned metadata
    describes the experiment; GPU numerical attribution still requires a probe.
    """
    if (
        not norm_names or len(set(norm_names)) != len(norm_names)
        or any(name not in ("img_norm1", "img_norm2") for name in norm_names)
    ):
        raise ValueError("Select unique image LayerNorm targets from img_norm1/img_norm2")
    targets = []
    for index, block in enumerate(model.transformer_blocks):
        for name in norm_names:
            path = f"transformer_blocks.{index}.{name}"
            module = getattr(block, name)
            _validate_norm(module, path)
            targets.append((path, module))
    return targets


def patch_native_layer_norm(model, *, norm_names=("img_norm1", "img_norm2")):
    """Install validated first-order native image LayerNorm forwards."""
    targets = validate_native_layer_norm(model, norm_names=norm_names)
    for _path, module in targets:
        module.forward = MethodType(_native_forward, module)
    return {
        "enabled": True,
        "diagnostic_only": True,
        "patch_count": len(targets),
        "norm_names": list(norm_names),
        "patched_modules": [path for path, _module in targets],
        "forward_op": FORWARD_OP,
        "backward_op": BACKWARD_OP,
        "cuda_autocast_policy": "FP32 input/output; preserve input dtype otherwise",
        "autograd_scope": "first-order VJP only; double backward unsupported",
        "attribution": "Native-kernel comparison; not proof of a unique error source",
    }
