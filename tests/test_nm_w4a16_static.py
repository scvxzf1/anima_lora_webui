from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from library.runtime.nm_w4a16.config import NMConfig
from library.runtime.nm_w4a16.cost import linear_cost
from library.runtime.nm_w4a16.linear import FrozenNMLinear
from library.runtime.nm_w4a16.mask import make_mask, validate_mask
from library.runtime.nm_w4a16.packing import dequantize, pack_weight, unpack_indices, validate_payload


def test_bidirectional_mask_and_roundtrip():
    torch.manual_seed(0)
    weight = torch.randn(16, 32, dtype=torch.bfloat16)
    mask = make_mask(weight, mode="bidirectional")
    validate_mask(mask, bidirectional=True)
    packed = pack_weight(weight, NMConfig(group_size=8))
    validate_payload(packed)
    restored = dequantize(packed)
    assert restored.shape == weight.shape
    assert restored.dtype == torch.bfloat16
    assert torch.isfinite(restored).all()
    assert packed.nbytes * 8 / weight.numel() == pytest.approx(3 + 32 / 8)
    # Adjacent output rows have different masks; row boundaries must not share
    # one metadata byte, otherwise roundtrip applies the wrong positions.
    expected_positions = mask.reshape(16, -1, 4).nonzero(as_tuple=False)[:, -1].reshape(16, 8, 2)
    assert torch.equal(unpack_indices(packed), expected_positions)


def test_reference_forward_and_input_gradient():
    torch.manual_seed(1)
    source = nn.Linear(32, 16, bias=False, dtype=torch.bfloat16)
    source.weight.requires_grad_(False)
    layer = FrozenNMLinear.from_linear(source, config=NMConfig(group_size=8))
    x = torch.randn(3, 32, dtype=torch.bfloat16, requires_grad=True)
    y = layer(x)
    y.sum().backward()
    assert y.shape == (3, 16)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert not any(parameter.requires_grad for parameter in layer.parameters())
    reference_weight = dequantize(layer.payload)
    expected_y = F.linear(x.detach(), reference_weight)
    expected_grad = F.linear(torch.ones_like(expected_y), reference_weight.t())
    assert torch.equal(y, expected_y)
    assert torch.equal(x.grad, expected_grad)
    assert all(not name.endswith(("values", "indices", "scales")) for name in layer.state_dict())
    assert "_nm_w4a16_experiment.bias" not in layer.state_dict()


def test_forward_only_mask_is_not_accepted_by_cached_transpose_backend():
    with pytest.raises(ValueError, match="transpose-safe"):
        NMConfig(group_size=8, mask_mode="forward", backend="cusparselt", cache_sparse=True).validate_shape(16, 32)


def test_cost_is_analytic_and_bf16_baselined():
    cost = linear_cost(4032, 6144, 16384, group_size=128)
    assert cost.payload_bytes == cost.int4_value_bytes + cost.index_bytes + cost.fp32_scale_bytes
    assert cost.sparse_forward_nonzero_flops == cost.bf16_forward_flops // 2
    assert cost.storage_ratio_vs_bf16 > 1
