"""Frozen INT4 payload with BF16 reference or staged sparse computation.

The cuSPARSELt backend is deliberately *not* called fused W4: it stages BF16
weights. Both forward and input gradient use exactly the same decoded matrix.
"""

import torch
from torch import nn
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

from .config import NMConfig
from .packing import PackedWeight, dequantize, pack_weight, validate_payload


def _compress_bf16(weight):
    if weight.device.type != "cuda":
        raise ValueError("cusparselt requires CUDA; no silent reference fallback")
    if torch.cuda.get_device_capability(weight.device) < (8, 0):
        raise ValueError("cusparselt experiment requires SM80 or newer")
    if not hasattr(torch, "_cslt_compress") or not hasattr(torch, "_cslt_sparse_mm"):
        raise RuntimeError("PyTorch was built without cuSPARSELt support")
    from torch.sparse import SparseSemiStructuredTensorCUSPARSELT

    return SparseSemiStructuredTensorCUSPARSELT.from_dense(weight.contiguous())


class _FrozenMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, runtime):
        payload = runtime.payload
        ctx.save_for_backward(payload.values, payload.indices, payload.scales)
        ctx.payload = payload
        ctx.runtime = runtime
        return runtime._matmul(x, payload, transpose=False)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        # Access saved tensors to enforce PyTorch's in-place version check.
        values, indices, scales = ctx.saved_tensors
        payload = ctx.payload
        if any(a is not b for a, b in zip((values, indices, scales), (payload.values, payload.indices, payload.scales))):
            payload = PackedWeight(values, indices, scales, payload.out_features,
                                   payload.in_features, payload.group_size, payload.mask_mode)
        grad_x = ctx.runtime._matmul(grad_output, payload, transpose=True)
        return grad_x, None


class FrozenNMLinear(nn.Module):
    """First-order frozen-base experiment. Activations must explicitly be BF16.

    ``cache_sparse=True`` stores two additional BF16 sparse orientations;
    ``False`` includes decode/compression in every forward/backward call.
    Neither mode promises the bandwidth of a fused sparse INT4 kernel.
    """

    def __init__(self, packed, *, config=NMConfig(), bias=None):
        super().__init__()
        validate_payload(packed)
        config.validate_shape(packed.out_features, packed.in_features)
        if (config.group_size, config.mask_mode) != (packed.group_size, packed.mask_mode):
            raise ValueError("config and payload do not agree")
        if bias is not None and (bias.shape != (packed.out_features,) or bias.requires_grad):
            raise ValueError("bias must be a frozen vector of out_features")
        self.config = config
        self.in_features = packed.in_features
        self.out_features = packed.out_features
        # Experimental buffers stay out of ordinary model/LoRA checkpoints.
        self.register_buffer("values", packed.values.detach().clone(), persistent=False)
        self.register_buffer("indices", packed.indices.detach().clone(), persistent=False)
        self.register_buffer("scales", packed.scales.detach().clone(), persistent=False)
        self.register_buffer("bias", None if bias is None else bias.detach().to(
            device=packed.values.device, dtype=torch.bfloat16).clone(), persistent=False)
        self.register_buffer("_sparse_forward", None, persistent=False)
        self.register_buffer("_sparse_backward", None, persistent=False)

    @classmethod
    def from_linear(cls, linear, *, config=NMConfig(), input_second_moment=None):
        if type(linear) is not nn.Linear:
            raise TypeError("only unquantized nn.Linear is supported")
        if linear.weight.requires_grad:
            raise ValueError("base weight must be frozen")
        packed = pack_weight(linear.weight, config, input_second_moment=input_second_moment)
        return cls(packed, config=config, bias=linear.bias)

    @property
    def payload(self):
        return PackedWeight(self.values, self.indices, self.scales, self.out_features,
                            self.in_features, self.config.group_size, self.config.mask_mode)

    def clear_cache(self):
        self._sparse_forward = None
        self._sparse_backward = None

    def _apply(self, fn, recurse=True):
        # Preserve FP32 scale precision even if a caller uses module.bfloat16().
        old_scales = self.scales
        self.clear_cache()
        result = super()._apply(fn, recurse=recurse)
        self.scales = old_scales.to(device=self.scales.device, dtype=torch.float32)
        if self.bias is not None:
            self.bias = self.bias.to(dtype=torch.bfloat16)
        return result

    @torch.no_grad()
    def prepare_sparse(self):
        """Explicit, potentially expensive CUDA preparation; never run on import."""
        if self.config.backend != "cusparselt" or not self.config.cache_sparse:
            raise ValueError("preparation requires cusparselt with cache_sparse=True")
        validate_payload(self.payload)
        weight = dequantize(self.payload)
        self._sparse_forward = _compress_bf16(weight)
        self._sparse_backward = _compress_bf16(weight.t().contiguous())

    def resident_bytes(self):
        """Payload and actual packed cache bytes, excluding temporary/workspace."""
        bias_bytes = 0 if self.bias is None else self.bias.numel() * self.bias.element_size()
        caches = (self._sparse_forward, self._sparse_backward)
        cache_bytes = sum(s.packed.numel() * s.packed.element_size() for s in caches if s is not None)
        return {"payload_bytes": self.payload.nbytes, "bias_bytes": bias_bytes,
                "sparse_cache_bytes": cache_bytes,
                "resident_bytes": self.payload.nbytes + bias_bytes + cache_bytes}

    def _matmul(self, x, payload, *, transpose):
        if x.dtype != torch.bfloat16:
            raise TypeError("NM W4A16 activation/input gradient must be BF16, not FP16")
        if x.device != payload.values.device:
            raise ValueError("move the runtime to the input device before execution")
        width = self.out_features if transpose else self.in_features
        output_width = self.in_features if transpose else self.out_features
        if x.ndim < 1 or x.shape[-1] != width:
            raise ValueError("input last dimension does not match the frozen weight")
        if x.numel() == 0:
            return x.new_empty((*x.shape[:-1], output_width))
        with torch.autocast(device_type=x.device.type, enabled=False):
            if self.config.backend == "reference":
                weight = dequantize(payload)
                return F.linear(x, weight.t() if transpose else weight)
            sparse = self._sparse_backward if transpose else self._sparse_forward
            current_payload = (
                payload.values.data_ptr() == self.values.data_ptr()
                and payload.scales.data_ptr() == self.scales.data_ptr()
            )
            if not self.config.cache_sparse or not current_payload:
                weight = dequantize(payload)
                sparse = _compress_bf16(weight.t() if transpose else weight)
            elif sparse is None:
                raise RuntimeError("call prepare_sparse() explicitly before using the cached backend")
            flat = x.reshape(-1, width)
            result = torch.mm(sparse, flat.t()).t()
            return result.reshape(*x.shape[:-1], output_width)

    def forward(self, x):
        result = _FrozenMatmul.apply(x, self)
        return result if self.bias is None else result + self.bias
