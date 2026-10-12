"""Disposable Anima MLP base-dx counterfactual; not a training precision mode."""

from __future__ import annotations

from types import MethodType
import weakref

import torch
from torch.nn import functional as F


SCOPE = "same_fp16_forward_current_fp16_weight_fp32_base_dx_only"


class CurrentWeightBaseDx(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, module):
        if (type(module) is not torch.nn.Linear or module.bias is not None
                or x.dtype != torch.float32 or module.weight.dtype != torch.float16
                or x.device != module.weight.device or module.weight.requires_grad):
            raise ValueError("Counterfactual requires colocated FP32 input and frozen biasless FP16 Linear")
        weight = module.weight
        ctx.module_ref = weakref.ref(module)
        ctx.parameter_id = id(weight)
        ctx.shape = tuple(weight.shape)
        ctx.device = weight.device
        ctx.dtype = weight.dtype
        with torch.autocast(x.device.type, enabled=False):
            return F.linear(x.half(), weight, None).float()

    @staticmethod
    def backward(ctx, grad):
        module = ctx.module_ref()
        if module is None:
            raise RuntimeError("Counterfactual base module disappeared before backward")
        weight = module.weight
        if (id(weight) != ctx.parameter_id or tuple(weight.shape) != ctx.shape
                or weight.dtype != ctx.dtype or weight.device != ctx.device
                or grad.device != ctx.device or grad.dtype != torch.float32
                or weight.requires_grad):
            raise RuntimeError("Counterfactual current base weight identity/shape/dtype/device drift")
        with torch.autocast(grad.device.type, enabled=False):
            return grad.float() @ weight.float(), None


def _counterfactual_forward(self, x):
    return CurrentWeightBaseDx.apply(x, self)


def target_names(block_count=28):
    return {f"blocks.{index}.mlp.layer{layer}" for index in range(block_count)
            for layer in (1, 2)}


def install(model, manifest, *, block_count=28):
    assignments = manifest.get("assignments", {})
    targets = target_names(block_count)
    if (getattr(model, "num_blocks", block_count) != block_count
            or {name for name, dtype in assignments.items() if dtype == "fp16"} != targets
            or any(dtype not in {"fp16", "fp32"} for dtype in assignments.values())
            or (block_count == 28 and len(assignments) != 377)):
        raise ValueError("Counterfactual requires exact MLP-only FP16 precision assignments")
    modules = [model.get_submodule(name) for name in sorted(targets)]
    if any(type(module) is not torch.nn.Linear or module.bias is not None
           or module.weight.dtype != torch.float16 or module.weight.requires_grad
           or getattr(module, "_adaptive_dtype", None) != torch.float16
           or not hasattr(module, "_adaptive_original_forward") for module in modules):
        raise ValueError("Counterfactual requires frozen plain biasless FP16 precision islands")
    for module in modules:
        module.forward = MethodType(_counterfactual_forward, module)
    return sorted(targets)
