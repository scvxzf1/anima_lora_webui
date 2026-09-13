"""Small helpers for reading quantized base weights without corrupting them.

NF4 weights are ``bitsandbytes.Params4bit`` containers.  Their ``data`` field is
packed storage and must never be treated as the logical ``(out, in)`` matrix.
This module deliberately exposes read-only materialization for initialization
and numerical checks; merge/fuse code must still refuse to write NF4 weights.
"""

from __future__ import annotations

from math import prod
from typing import Sequence

import torch


def _is_4bit_parameter(weight: object) -> bool:
    # Keep bitsandbytes optional for dense adapters, including Params4bit subclasses.
    return any(cls.__name__ == "Params4bit" for cls in type(weight).__mro__)


def is_nf4_weight(weight: object) -> bool:
    """Recognize packed bitsandbytes weights, including restored parameters."""

    return (
        _is_4bit_parameter(weight) and getattr(weight, "quant_state", None) is not None
    )


def materialize_weight(
    weight: torch.Tensor,
    *,
    expected_shape: Sequence[int] | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Read a logical dense view of a normal or NF4 weight.

    The returned tensor is detached and writable by the caller.  For NF4 the
    packed parameter and its quantization state remain untouched.  Shape is
    checked explicitly because some CPU bitsandbytes builds return a flattened
    view for one-dimensional inputs.
    """

    if is_nf4_weight(weight):
        import bitsandbytes.functional as bnb_functional

        dense = bnb_functional.dequantize_4bit(
            weight.data,
            quant_state=weight.quant_state,
        )
    else:
        if _is_4bit_parameter(weight) and (
            not weight.is_floating_point() or getattr(weight, "bnb_quantized", False)
        ):
            raise ValueError("Packed Params4bit weight is missing quant_state")
        dense = weight.detach().clone()

    if expected_shape is not None:
        shape = tuple(int(value) for value in expected_shape)
        if dense.numel() != prod(shape):
            raise ValueError(
                "quantized weight element count does not match module shape: "
                f"got {tuple(dense.shape)}, expected {shape}"
            )
        if tuple(dense.shape) != shape:
            dense = dense.reshape(shape)

    return dense.detach().to(device=device, dtype=dtype).contiguous()


def materialize_module_weight(
    module: torch.nn.Module,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Materialize a module weight using its declared logical dimensions."""

    weight = module.weight
    if hasattr(module, "in_features") and hasattr(module, "out_features"):
        expected_shape = (module.out_features, module.in_features)
    else:
        expected_shape = tuple(weight.shape)
    return materialize_weight(
        weight,
        expected_shape=expected_shape,
        device=device,
        dtype=dtype,
    )
