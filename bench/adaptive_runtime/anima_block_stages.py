"""Bench-only checkpoint-aware tracing of Anima's three gated residual stages."""

from __future__ import annotations

from types import MethodType

import torch
from torch.utils.checkpoint import checkpoint


STAGES = ("self_attn", "cross_attn", "mlp")


class _RecordingContext:
    """Checkpoint may re-enter the same recompute context on repeated VJPs."""
    def __init__(self, trace, recording):
        self.trace, self.recording = trace, recording

    def __enter__(self):
        if self.trace.recording:
            raise ValueError("Nested stage recording")
        self.trace.recording = self.recording

    def __exit__(self, _type, _value, _traceback):
        self.trace.recording = False


class ResidualStageTrace:
    def __init__(self, block):
        if (not block.training or not block.gradient_checkpointing or block.fp32_residual
                or any(getattr(block, flag, False) for flag in (
                    "unsloth_offload_checkpointing", "cpu_offload_checkpointing",
                    "adapter_aware_checkpointing", "selective_checkpointing"))):
            raise ValueError("Stage trace requires reference FP32/full non-reentrant checkpoint")
        self.block = block
        self.original_forward = block.forward
        self.original_residual = block._gated_residual_add
        self.recording = False
        self.stages = []
        self.block.forward = MethodType(self._forward, block)
        self.block._gated_residual_add = MethodType(self._residual, block)

    def _contexts(self):
        return _RecordingContext(self, True), _RecordingContext(self, False)

    def _forward(self, block, x, **kwargs):
        if self.stages:
            raise ValueError("Repeated stage trace forward")
        return checkpoint(block._forward, x, **kwargs, use_reentrant=False,
                          context_fn=self._contexts)

    def _residual(self, _block, residual, gate, branch):
        output = self.original_residual(residual, gate, branch)
        if self.recording:
            if (len(self.stages) >= len(STAGES) or residual.shape != output.shape
                    or branch.shape != residual.shape
                    or any(t.dtype != torch.float32 for t in (residual, gate, branch, output))):
                raise ValueError("Unexpected gated residual stage contract")
            if self.stages and residual is not self.stages[-1]["output"]:
                raise ValueError("Nonadjacent internal residual stages")
            self.stages.append({"name": STAGES[len(self.stages)], "input": residual,
                                "output": output, "gate": gate, "branch": branch})
        return output

    def restore(self):
        self.block.forward = self.original_forward
        self.block._gated_residual_add = self.original_residual
        self.recording = False

    def boundaries(self):
        if len(self.stages) != len(STAGES):
            raise ValueError("Missing gated residual stage")
        return [self.stages[0]["input"], *(stage["output"] for stage in self.stages)]

    def cotangents(self, output, terminal_cotangent):
        boundaries = self.boundaries()
        if output is not boundaries[-1]:
            raise ValueError("Trace output identity mismatch")
        return torch.autograd.grad(output, boundaries, grad_outputs=terminal_cotangent,
                                   retain_graph=True)

    def local_vjp(self, index, cotangent):
        stage = self.stages[index]
        (complete,) = torch.autograd.grad(stage["output"], stage["input"],
                                          grad_outputs=cotangent, retain_graph=True)
        (branch,) = torch.autograd.grad(stage["branch"], stage["input"],
                                        grad_outputs=cotangent * stage["gate"], retain_graph=True)
        return complete, branch
