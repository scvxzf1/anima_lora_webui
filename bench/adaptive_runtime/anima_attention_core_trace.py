"""Actual fused-QKV and output-projection boundaries, preserving applied LoRA."""

import torch

from bench.adaptive_runtime.anima_self_attention_trace import SelfAttentionTrace


class AttentionCoreTrace(SelfAttentionTrace):
    def __init__(self, block):
        super().__init__(block)
        self.projected = self.core_output = None
        try:
            self.handles.append(block.self_attn.qkv_proj.register_forward_hook(self._projected))
            self.handles.append(block.self_attn.output_proj.register_forward_pre_hook(self._core_output))
        except BaseException:
            self.restore()
            raise

    def _projected(self, _module, _args, value):
        if self.recording:
            if self.projected is not None:
                raise ValueError("Repeated fused QKV boundary")
            self.projected = value

    def _core_output(self, _module, args):
        if self.recording:
            if self.core_output is not None:
                raise ValueError("Repeated core output boundary")
            self.core_output = args[0]

    def validate(self):
        super().validate()
        z = self.attention_input
        if (not isinstance(self.projected, torch.Tensor) or not isinstance(self.core_output, torch.Tensor)
                or self.projected.shape != (*z.shape[:-1], 3 * z.shape[-1])
                or self.core_output.shape != z.shape
                or self.projected.dtype != torch.float32 or self.core_output.dtype != torch.float32):
            raise ValueError("Unexpected fused QKV/core boundary layout")

    def projection_vjp(self, terminal):
        self.validate()
        (du,) = torch.autograd.grad(self.attention_output, self.core_output,
                                   terminal.reshape_as(self.attention_output), retain_graph=True)
        return du

    def core_vjp(self, du):
        self.validate()
        (dp,) = torch.autograd.grad(self.core_output, self.projected,
                                   du.reshape_as(self.core_output), retain_graph=True)
        return dp

    def qkv_vjp(self, dp):
        self.validate()
        (dz,) = torch.autograd.grad(self.projected, self.attention_input,
                                   dp.reshape_as(self.projected), retain_graph=True)
        return dz
