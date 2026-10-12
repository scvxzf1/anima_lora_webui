"""Portable INT4 values + two 2-bit indices, independent of CUDA layouts."""

from dataclasses import dataclass

import torch

from .config import NMConfig
from .mask import make_mask, validate_mask


@dataclass(frozen=True)
class PackedWeight:
    values: torch.Tensor
    indices: torch.Tensor
    scales: torch.Tensor
    out_features: int
    in_features: int
    group_size: int
    mask_mode: str

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.values, self.indices, self.scales))


@torch.no_grad()
def pack_weight(weight, config=NMConfig(), *, input_second_moment=None):
    if weight.ndim != 2 or weight.dtype != torch.bfloat16:
        raise TypeError("source weight must be a 2D BF16 tensor (not NF4/FP16)")
    n, k = weight.shape
    config.validate_shape(n, k)
    mask = make_mask(weight, mode=config.mask_mode, input_second_moment=input_second_moment)
    validate_mask(mask, bidirectional=config.mask_mode == "bidirectional")
    sparse = weight.detach().float().masked_fill(~mask, 0)
    groups = sparse.reshape(n, k // config.group_size, config.group_size)
    scales = groups.abs().amax(-1) / 7.0
    # Zero groups have a harmless unit scale; no epsilon clips tiny BF16 weights.
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    quantized = (groups / scales[..., None]).round().clamp(-7, 7).to(torch.int8).reshape(n, k)
    positions = torch.arange(4, device=weight.device).expand(n, k // 4, 4)
    positions = positions.masked_fill(~mask.reshape(n, -1, 4), 4).sort(-1).values[..., :2]
    selected = quantized.reshape(n, -1, 4).gather(-1, positions).to(torch.int16) + 8
    values = (selected[..., 0] | (selected[..., 1] << 4)).to(torch.uint8).contiguous()
    meta = (positions[..., 0] | (positions[..., 1] << 2)).reshape(-1)
    # Pair adjacent K quartets within each output row (not across row edges).
    indices = (meta.reshape(n, -1, 2)[..., 0] | (meta.reshape(n, -1, 2)[..., 1] << 4))
    indices = indices.reshape(-1).to(torch.uint8).contiguous()
    return PackedWeight(values.reshape(-1).contiguous(), indices, scales.contiguous(), n, k, config.group_size, config.mask_mode)


def unpack_indices(packed):
    meta = torch.stack((packed.indices & 15, packed.indices >> 4), dim=-1).reshape(packed.out_features, -1)
    return torch.stack((meta & 3, meta >> 2), dim=-1).long()


def dequantize(packed, *, dtype=torch.bfloat16):
    positions = unpack_indices(packed)
    values = torch.stack((packed.values & 15, packed.values >> 4), dim=-1).float() - 8
    values = values.reshape(packed.out_features, packed.in_features // 4, 2)
    restored = torch.zeros(
        (packed.out_features, packed.in_features // 4, 4),
        device=packed.values.device, dtype=torch.float32,
    ).scatter_(-1, positions, values)
    groups = restored.reshape(packed.out_features, -1, packed.group_size)
    return (groups * packed.scales[..., None]).reshape(packed.out_features, packed.in_features).to(dtype)


def validate_payload(packed):
    config = NMConfig(group_size=packed.group_size, mask_mode=packed.mask_mode)
    config.validate_shape(packed.out_features, packed.in_features)
    n, k = packed.out_features, packed.in_features
    if packed.values.dtype != torch.uint8 or packed.values.shape != (n * k // 4,):
        raise ValueError("invalid packed INT4 values")
    if packed.indices.dtype != torch.uint8 or packed.indices.shape != (n * k // 8,):
        raise ValueError("invalid packed 2:4 indices")
    if packed.scales.dtype != torch.float32 or packed.scales.shape != (n, k // packed.group_size):
        raise ValueError("invalid FP32 scale shape/dtype")
    if len({t.device for t in (packed.values, packed.indices, packed.scales)}) != 1:
        raise ValueError("all payload tensors must share one device")
    if not bool(torch.isfinite(packed.scales).all()) or bool((packed.scales <= 0).any()):
        raise ValueError("scales must be finite and positive")
    if bool(((packed.values & 15) == 0).any()) or bool(((packed.values >> 4) == 0).any()):
        raise ValueError("reserved INT4 code -8 is not valid in the symmetric [-7,7] format")
    positions = unpack_indices(packed)
    if bool((positions[..., 0] >= positions[..., 1]).any()):
        raise ValueError("2:4 positions must be distinct and sorted")
    mask = torch.zeros((n, k // 4, 4), device=packed.values.device, dtype=torch.bool)
    mask.scatter_(-1, positions, True)
    validate_mask(mask.reshape(n, k), bidirectional=packed.mask_mode == "bidirectional")
