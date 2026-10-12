"""Explicit W8A8 engineering/backward experiments; production defaults stay unchanged.

Call after ``apply_convrot_to_lora_network`` and before compile/offloader setup.
Only the frozen base's input gradient changes: LoRA parameters/optimizer remain
floating point. No activation or second persistent base payload is retained.
The returned JSON-compatible manifest is intended for effective config and
checkpoint metadata; resume callers must compare/restore the same policies.
CUDA engineering and INT8-backward paths never fall back to floating GEMM.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from library.runtime.convrot.apply import (
    _ATTR_GROUP,
    _ATTR_MODE,
    _ATTR_WEIGHT_LAYOUT,
    _BUFFER_HADAMARD,
    _BUFFER_Q,
    _BUFFER_SCALE,
    _resolve_base_linear,
    _swap_payload,
)
from library.runtime.convrot.checks import raise_if_compiled
from library.runtime.convrot.fused import _FusedW8A8Fn, _resolve_compute_dtype
from library.runtime.convrot.gemm import int8_mm_scaled, quantize_activation_absmax_int8
from library.runtime.convrot.rht import (
    assert_group_divides,
    group_rht,
    normalized_hadamard,
)

ForwardPolicy = Literal["torch", "triton_quant", "triton_gemm", "triton_both"]
BackwardPolicy = Literal["fp32", "bf16", "int8"]
_FORWARD_POLICIES = {"torch", "triton_quant", "triton_gemm", "triton_both"}
_BACKWARD_POLICIES = {"fp32", "bf16", "int8"}


def _validate_policies(forward_policy: str, backward_policy: str) -> None:
    if forward_policy not in _FORWARD_POLICIES:
        raise ValueError(f"unsupported forward_policy={forward_policy!r}")
    if backward_policy not in _BACKWARD_POLICIES:
        raise ValueError(f"unsupported backward_policy={backward_policy!r}")


def _dense_rotate(
    x: torch.Tensor, group_size: int, hadamard: torch.Tensor
) -> torch.Tensor:
    # An explicit matrix prevents group_rht from consulting the FWHT setting.
    return group_rht(x, group_size, hadamard=hadamard)


def _forward_gemm(x_rot, w_q, w_scale, layout, dtype, policy):
    if policy in {"triton_quant", "triton_both"}:
        from library.runtime.convrot.triton_kernels import quantize_rowwise

        x_q, x_scale = quantize_rowwise(x_rot)
    else:
        x_q, x_scale = quantize_activation_absmax_int8(x_rot)
    if policy in {"triton_gemm", "triton_both"}:
        from library.runtime.convrot.triton_kernels import int8_matmul_scaled

        return int8_matmul_scaled(
            x_q,
            x_scale,
            w_q,
            w_scale.reshape(-1),
            weight_layout=layout,
            out_dtype=dtype,
        )
    # CPU is an explicit numerical reference; CUDA must execute real INT8.
    return int8_mm_scaled(
        x_q,
        x_scale,
        w_q,
        w_scale,
        weight_layout=layout,
        out_dtype=dtype,
        prefer="int_mm" if x_rot.is_cuda else "float",
    )


def _low_precision_backward(ctx, grad_y):
    layout = ctx.weight_layout
    scale = ctx.w_scale.to(device=grad_y.device)
    w = ctx.w_q
    if w.device != grad_y.device:
        raise RuntimeError(
            "ConvRot payload moved before backward; offloader must restore its block"
        )
    if ctx.backward_policy == "bf16":
        # musubi ordering: form cast(Wq) * cast(scale) before the BF16 dot.
        w_bf16 = w.to(torch.bfloat16)
        scale_bf16 = scale.to(torch.bfloat16).reshape(-1)
        weight = w_bf16 * scale_bf16 if layout == "kn" else w_bf16 * scale_bf16[:, None]
        right = weight.transpose(0, 1) if layout == "kn" else weight
        grad_rot = grad_y.to(torch.bfloat16) @ right
    else:
        from library.runtime.convrot.triton_kernels import (
            int8_matmul_scaled,
            quantize_rowwise,
        )

        # Fold output-channel scales into dy, then quantize rows over N.
        folded = grad_y.to(torch.float32) * scale.to(torch.float32).reshape(-1)
        grad_q, grad_scale = quantize_rowwise(folded)
        # Reverse the interpretation of the same payload; transpose is a view,
        # with no cached/transposed copy or additional resident base storage.
        reverse_layout = "nk" if layout == "kn" else "kn"
        k = w.shape[0] if layout == "kn" else w.shape[1]
        unit_scale = torch.ones(k, device=grad_y.device, dtype=torch.float32)
        grad_rot = int8_matmul_scaled(
            grad_q,
            grad_scale,
            w,
            unit_scale,
            weight_layout=reverse_layout,
            out_dtype=ctx.compute_dtype,
        )
    grad_x = _dense_rotate(grad_rot, ctx.group_size, ctx.hadamard)
    return grad_x.to(dtype=ctx.x_dtype)


class _ExperimentalW8A8Fn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        w_q,
        w_scale,
        group_size,
        hadamard,
        layout,
        forward_policy,
        backward_policy,
    ):
        dtype = _resolve_compute_dtype(x)
        x_rot = _dense_rotate(x.to(dtype=dtype), group_size, hadamard)
        y = _forward_gemm(x_rot, w_q, w_scale, layout, dtype, forward_policy)
        ctx.group_size = group_size
        ctx.hadamard = hadamard
        ctx.weight_layout = layout
        ctx.compute_dtype = dtype
        ctx.x_dtype = x.dtype
        ctx.w_q = w_q
        ctx.w_scale = w_scale
        ctx.backward_policy = backward_policy
        # Constants are transient ctx attributes, matching the existing fused
        # path; no input activation and no differentiable base weight is saved.
        # Match production's Function output dtype. The public wrapper's cast
        # must stay outside autograd.Function, so grad_y is rounded back to
        # compute dtype before the custom backward for FP32 residual inputs.
        return y

    @staticmethod
    def backward(ctx, grad_y):
        if ctx.backward_policy == "fp32":
            # Reuse the exact production STE rule (including optional TF32 and
            # compute-dtype RHT), adapting only the Function argument count.
            grad_x = _FusedW8A8Fn.backward(ctx, grad_y)[0]
        else:
            grad_x = _low_precision_backward(ctx, grad_y)
        return grad_x, None, None, None, None, None, None, None


def experimental_w8a8_forward(
    x: torch.Tensor,
    quantized_weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    group_size: int,
    hadamard: torch.Tensor | None = None,
    weight_layout: str = "kn",
    forward_policy: ForwardPolicy = "torch",
    backward_policy: BackwardPolicy = "fp32",
) -> torch.Tensor:
    """Explicit frozen-base Function; CPU torch policy is reference-only."""
    _validate_policies(forward_policy, backward_policy)
    if weight_layout not in {"kn", "nk"}:
        raise ValueError("weight_layout must be 'kn' or 'nk'")
    if not x.is_floating_point():
        raise TypeError("x must be floating point")
    if quantized_weight.device != x.device or scale.device != x.device:
        raise RuntimeError(
            "ConvRot payload must be resident with x; restore the block before forward"
        )
    k = quantized_weight.shape[0 if weight_layout == "kn" else 1]
    assert_group_divides(int(k), int(group_size))
    if hadamard is None:
        hadamard = normalized_hadamard(
            int(group_size),
            device=x.device,
            dtype=_resolve_compute_dtype(x),
        )
    result = _ExperimentalW8A8Fn.apply(
        x,
        quantized_weight,
        scale,
        int(group_size),
        hadamard,
        weight_layout,
        forward_policy,
        backward_policy,
    )
    return result.to(dtype=x.dtype) if result.dtype != x.dtype else result


def _install_policy(lora, owner, forward_policy, backward_policy):
    def base_forward(x, *, _lora=lora, _owner=owner):
        # Read carrier tensors every call. The closure contains owner objects,
        # never the payloads which block swap may replace or reuse.
        return experimental_w8a8_forward(
            x,
            _swap_payload(_owner, _BUFFER_Q),
            _swap_payload(_owner, _BUFFER_SCALE),
            group_size=getattr(_lora, _ATTR_GROUP),
            hadamard=getattr(_lora, _BUFFER_HADAMARD, None),
            weight_layout=getattr(_lora, _ATTR_WEIGHT_LAYOUT),
            forward_policy=forward_policy,
            backward_policy=backward_policy,
        )

    lora.org_forward = base_forward


def apply_experimental_policy(
    network: nn.Module,
    *,
    forward_policy: ForwardPolicy = "torch",
    backward_policy: BackwardPolicy = "fp32",
) -> dict[str, object]:
    """Replace only already-patched W8A8 LoRA org_forward, before compilation.

    Validate every candidate before mutation. W8A16 and unpatched modules are
    untouched. A zero-candidate network is rejected, preventing a false trial.
    """
    _validate_policies(forward_policy, backward_policy)
    raise_if_compiled(network, context="apply_experimental_policy")
    candidates = []
    for lora in getattr(network, "unet_loras", []) or []:
        if getattr(lora, _ATTR_MODE, None) != "w8a8":
            continue
        owner = _resolve_base_linear(lora)
        if owner is None:
            raise RuntimeError("W8A8 module has no live base owner")
        _swap_payload(owner, _BUFFER_Q)
        _swap_payload(owner, _BUFFER_SCALE)
        if getattr(lora, _ATTR_WEIGHT_LAYOUT, None) not in {"kn", "nk"}:
            raise ValueError("W8A8 module is missing a valid weight layout")
        if not hasattr(lora, _ATTR_GROUP):
            raise ValueError("W8A8 module is missing its group size")
        candidates.append((lora, owner))
    if not candidates:
        raise ValueError(
            "apply_experimental_policy requires an already-patched W8A8 module"
        )
    for lora, owner in candidates:
        _install_policy(lora, owner, forward_policy, backward_policy)
    return {
        "implementation": "convrot_experimental_policy_v1",
        "forward_policy": forward_policy,
        "backward_policy": backward_policy,
        "strict_no_fallback": True,
        "cpu_torch_reference_only": True,
        "rht_backend": "dense",
        "lora_parameter_quantization": False,
        "extra_persistent_base_payload_bytes": 0,
        "patched_count": len(candidates),
        "patched_modules": [
            str(getattr(lora, "lora_name", "")) for lora, _ in candidates
        ],
    }
