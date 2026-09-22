"""Explicit precision islands for frozen, nonquantized units before adapters.

The surrounding graph stays FP32. Low-precision units return FP32 tensors to
avoid silently downcasting a sensitive residual stream. State-dict keys stay
unchanged; integration must install before offloader masters/compile/optimizers.
"""

from __future__ import annotations

from types import MethodType

import torch
from torch.utils._pytree import tree_flatten

from .precision import floating_tree


DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def _island_forward(self, *args, **kwargs):
    tensors, _ = tree_flatten((args, kwargs))
    floating = [t for t in tensors if isinstance(t, torch.Tensor) and t.is_floating_point()]
    if not floating:
        raise ValueError("Precision islands require floating tensor inputs")
    device = floating[0].device
    if any(t.device != device for t in floating) or self.weight.device != device:
        raise ValueError("Precision islands require colocated inputs and weights")
    args, kwargs = floating_tree((args, kwargs), device=device, dtype=self._adaptive_dtype)
    with torch.autocast(device.type, enabled=False):
        result = self._adaptive_original_forward(*args, **kwargs)
    return floating_tree(result, device=device, dtype=torch.float32)


def install_precision_islands(model, assignments: dict[str, str]):
    """Apply validated assignments to disjoint frozen Linear modules only.

    Linear-only installation deliberately excludes RoPE, normalization, mixed
    argument contracts and quantized subclasses until family adapters validate
    them. This function does not certify the supplied assignments as safe.
    """
    prepared = []
    seen = set()
    for name, dtype_name in assignments.items():
        if not name or dtype_name not in DTYPES:
            raise ValueError("Expected a named unit and fp16/bf16/fp32 dtype")
        module = model.get_submodule(name)
        if id(module) in seen:
            raise ValueError("Aliased precision islands must be assigned once")
        seen.add(id(module))
        if type(module) is not torch.nn.Linear:
            raise ValueError(f"Precision island {name} must be a plain Linear")
        if hasattr(module, "_adaptive_original_forward"):
            raise ValueError(f"Precision island {name} is already installed")
        for parameter in module.parameters():
            if type(parameter) is not torch.nn.Parameter or not parameter.is_floating_point():
                raise ValueError("Quantized/custom parameters cannot be precision islands")
            if parameter.requires_grad:
                raise ValueError("Install precision islands on frozen base weights before adapters")
            if parameter.device.type == "meta":
                raise ValueError("Load actual weights before installing precision islands")
            if not torch.isfinite(parameter).all():
                raise ValueError(f"Nonfinite base weights in {name}")
            if parameter.abs().max() > torch.finfo(DTYPES[dtype_name]).max:
                raise ValueError(f"Base weights in {name} overflow {dtype_name}")
        prepared.append((module, DTYPES[dtype_name]))
    for module, dtype in prepared:
        module.to(dtype=dtype)
        module._adaptive_dtype = dtype
        module._adaptive_original_forward = module.forward
        module.forward = MethodType(_island_forward, module)
    return model
