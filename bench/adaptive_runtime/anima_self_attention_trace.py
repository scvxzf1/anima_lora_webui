"""Bench-only differentiable boundary around Anima self-attention."""

import torch

from bench.adaptive_runtime.anima_block_stages import ResidualStageTrace


class SelfAttentionTrace(ResidualStageTrace):
    """Split A(N(x)) at actual module input, without rebuilding either path."""
    def __init__(self, block):
        super().__init__(block)
        self.attention_input = self.attention_output = None
        self.handles = []
        try:
            self.handles.append(block.self_attn.register_forward_pre_hook(self._before))
            self.handles.append(block.self_attn.register_forward_hook(self._after))
        except BaseException:
            self.restore()
            raise

    def _before(self, _module, args):
        if self.recording:
            if self.attention_input is not None:
                raise ValueError("Repeated self-attention input")
            self.attention_input = args[0]

    def _after(self, _module, _args, output):
        if self.recording:
            if self.attention_output is not None:
                raise ValueError("Repeated self-attention output")
            self.attention_output = output

    def validate(self):
        self.boundaries()
        z, b = self.attention_input, self.attention_output
        shape = self.stages[0]["input"].shape
        if (not isinstance(z, torch.Tensor) or not isinstance(b, torch.Tensor)
                or z.shape != b.shape or z.shape != (shape[0], shape[1] * shape[2] * shape[3], shape[4])
                or z.dtype != torch.float32 or b.dtype != torch.float32):
            raise ValueError("Unexpected self-attention boundary layout")
        if not torch.equal(b.detach().reshape(shape), self.stages[0]["branch"].detach()):
            raise ValueError("Self-attention output is not the first residual branch")

    def attention_vjp(self, terminal):
        self.validate()
        (h,) = torch.autograd.grad(self.attention_output, self.attention_input,
                                  grad_outputs=terminal.reshape_as(self.attention_output), retain_graph=True)
        return h

    def normalization_vjp(self, cotangent):
        self.validate()
        (dx,) = torch.autograd.grad(self.attention_input, self.stages[0]["input"],
                                   grad_outputs=cotangent.reshape_as(self.attention_input), retain_graph=True)
        return dx

    def branch_vjp(self, terminal):
        self.validate()
        (dx,) = torch.autograd.grad(self.attention_output, self.stages[0]["input"],
                                   grad_outputs=terminal.reshape_as(self.attention_output), retain_graph=True)
        return dx

    def restore(self):
        for handle in getattr(self, "handles", []):
            handle.remove()
        self.handles = []
        super().restore()
