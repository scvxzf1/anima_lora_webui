"""Opt-in finite-value tracing for calibration, never default training overhead."""

from __future__ import annotations

import torch
from torch.utils._pytree import tree_flatten


def _tensor_summary(value):
    leaves, _ = tree_flatten(value)
    result = []
    for tensor in leaves:
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            continue
        detached = tensor.detach()
        finite = torch.isfinite(detached)
        valid = detached[finite]
        result.append({"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                       "finite": bool(finite.all()),
                       "max_abs_finite": float(valid.abs().max()) if valid.numel() else None})
    return result


class FiniteTrace:
    def __init__(self, model, report):
        self.handles = []
        self.report = report
        for name, module in model.named_modules():
            if not list(module.children()):
                self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def hook(module, inputs, outputs):
            if "first_nonfinite" in self.report:
                return
            leaves, _ = tree_flatten(outputs)
            if any(not torch.isfinite(t.detach()).all() for t in leaves
                   if isinstance(t, torch.Tensor) and t.is_floating_point()):
                self.report["first_nonfinite"] = {
                    "module": name, "type": type(module).__name__,
                    "inputs": _tensor_summary(inputs), "outputs": _tensor_summary(outputs),
                }
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
