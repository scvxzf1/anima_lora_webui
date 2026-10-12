"""Opaque native SiLU diagnostic for first-order Qwen NF4 VJPs only.

This is an explicit diagnostic, not a production compile policy. Double
backward is unsupported: the native backward op has no autograd registration.
"""
from __future__ import annotations

from types import MethodType

import torch


FORWARD_OP = "qwen_nf4_diagnostic::native_silu_forward"
BACKWARD_OP = "qwen_nf4_diagnostic::native_silu_backward"


@torch.library.custom_op(FORWARD_OP, mutates_args=())
def native_silu_forward(input: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.silu.default(input)


@native_silu_forward.register_fake
def _forward_fake(input):
    # Native metadata also handles transposes and non-dense views correctly.
    return torch.ops.aten.silu.default(input)


@torch.library.custom_op(BACKWARD_OP, mutates_args=())
def native_silu_backward(gradient: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.silu_backward.default(gradient, input)


@native_silu_backward.register_fake
def _backward_fake(gradient, input):
    return torch.ops.aten.silu_backward.default(gradient, input)


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(inputs[0])


def _backward(ctx, gradient):
    return native_silu_backward(gradient, ctx.saved_tensors[0])


native_silu_forward.register_autograd(_backward, setup_context=_setup_context)


def _native_forward(module, input):
    return native_silu_forward(input)


def validate_native_silu(model):
    """Validate all image-MLP activations, then patch their forward in place.

    Module objects, parameters and state_dict entries remain intact. Every
    target must be nn.SiLU(inplace=False); no partial patch occurs on failure.
    """
    targets = []
    for index, block in enumerate(model.transformer_blocks):
        path = f"transformer_blocks.{index}.img_mlp.activation_fn"
        module = block.img_mlp.activation_fn
        if not isinstance(module, torch.nn.SiLU):
            raise TypeError(f"Native SiLU diagnostic requires nn.SiLU: {path}")
        if module.inplace:
            raise ValueError(f"Native SiLU diagnostic requires inplace=False: {path}")
        targets.append((path, module))
    if not targets:
        raise ValueError("Native SiLU diagnostic requires image MLP activation targets")
    return targets


def patch_native_silu(model):
    """Install validated first-order native image SiLU forwards."""
    targets = validate_native_silu(model)
    for _path, module in targets:
        module.forward = MethodType(_native_forward, module)
    return {
        "enabled": True,
        "diagnostic_only": True,
        "first_order_only": True,
        "patch_count": len(targets),
        "patched_modules": [path for path, _module in targets],
        "forward_op": FORWARD_OP,
        "backward_op": BACKWARD_OP,
        "autograd_scope": "first-order VJP only; double backward unsupported",
        "attribution": "Native-kernel comparison; exact divergent instruction unproven",
    }
