"""Shared packed Linear with the CUDA BF16 autocast output contract."""
from __future__ import annotations

import torch
from bitsandbytes.nn import Linear4bit
from .constants import STATE_KEYS


class BF16AutocastLinear4bit(Linear4bit):
    """Keep bnb autograd/storage, but avoid restoring FP32 autocast outputs.

    Linear4bit casts its result back to the input dtype. Casting the input
    before its forward preserves ordinary Linear's BF16 autocast contract
    without allocating a full FP32 result or changing saved packed aliases.
    Other devices, compute dtypes and disabled/nested autocast use stock bnb.
    """

    def forward(self, x: torch.Tensor):
        if (
            x.device.type == "cuda"
            and x.dtype in (torch.float32, torch.float16)
            and self.compute_dtype == torch.bfloat16
            and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.bfloat16
        ):
            x = x.to(torch.bfloat16)
        return super().forward(x)

def _valid_nested_code(code) -> bool:
    from bitsandbytes.functional import create_dynamic_map
    expected = create_dynamic_map()
    if torch.equal(code, expected):
        return True
    # bnb's native CPU blockwise kernel changes only the first endpoint to -1.
    expected[0] = -1.0
    return torch.equal(code, expected)


def quant_state(state: dict, weight_key: str, shape) -> tuple[dict, object]:
    from bitsandbytes.functional import QuantState, get_4bit_type
    qs = {k: state[weight_key + '.' + k] for k in STATE_KEYS if weight_key + '.' + k in state}
    if set(qs) != STATE_KEYS or qs['quant_state.bitsandbytes__nf4'].numel() > 65536:
        raise ValueError(f'Incomplete or oversized NF4 state: {weight_key}')
    try:
        q = QuantState.from_dict(dict(qs), device=torch.device('cpu'))
    except Exception as exc:
        raise ValueError(f'Invalid NF4 quant_state: {weight_key}') from exc
    n = shape[0] * shape[1]
    if (q.quant_type != 'nf4' or q.blocksize != 64 or q.dtype != torch.bfloat16
            or tuple(q.shape) != tuple(shape) or not q.nested or q.state2.blocksize != 256
            or q.state2.dtype != torch.float32
            or q.absmax.dtype != torch.uint8 or tuple(q.absmax.shape) != (n // 64,)
            or q.code.dtype != torch.float32 or tuple(q.code.shape) != (16,)
            or q.state2.code.dtype != torch.float32 or tuple(q.state2.code.shape) != (256,)
            or q.state2.absmax.dtype != torch.float32
            or tuple(q.state2.absmax.shape) != ((n // 64 + 255) // 256,)
            or not torch.isfinite(q.code).all() or not torch.isfinite(q.state2.code).all()
            or not torch.isfinite(q.state2.absmax).all() or (q.state2.absmax < 0).any()
            or q.offset.numel() != 1 or not torch.isfinite(q.offset).all() or (q.offset < 0).any()
            or not torch.equal(q.code, get_4bit_type('nf4', device='cpu'))
            or not _valid_nested_code(q.state2.code)):
        raise ValueError(f'NF4 quant_state violates fixed policy: {weight_key}')
    return qs, q


def restore_nf4_linear(data, qs, *, in_features, out_features, bias=None, device='cpu'):
    """Restore validated packed storage directly, without requantization."""
    from bitsandbytes.nn import Params4bit
    shape = (out_features, in_features)
    if (in_features <= 0 or out_features <= 0 or in_features % 64 or out_features % 64
            or data.dtype != torch.uint8 or tuple(data.shape) != (in_features * out_features // 2, 1)):
        raise ValueError('Invalid packed NF4 weight shape or dtype')
    quant_state({'weight.' + k: v for k, v in qs.items()}, 'weight', shape)
    if bias is not None and (tuple(bias.shape) != (out_features,) or bias.dtype != torch.bfloat16):
        raise ValueError('Invalid NF4 bias')
    new = BF16AutocastLinear4bit(in_features, out_features, bias=bias is not None,
        compute_dtype=torch.bfloat16, compress_statistics=True, quant_type='nf4', device='meta')
    new.weight = Params4bit.from_prequantized(data, dict(qs), requires_grad=False,
                                             device=device, module=new)
    if bias is not None:
        new.bias = torch.nn.Parameter(bias.to(device), requires_grad=False)
    return new
