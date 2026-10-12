"""Deterministic 2:4 masks, optionally valid in both matrix orientations."""

from functools import lru_cache
from itertools import combinations, product

import torch


@lru_cache(maxsize=1)
def balanced_patterns():
    """All 90 binary 4x4 masks with two retained entries per row and column."""
    patterns = []
    for rows in product(tuple(combinations(range(4), 2)), repeat=4):
        if all(sum(column in row for row in rows) == 2 for column in range(4)):
            patterns.append(tuple(int(column in row) for row in rows for column in range(4)))
    return tuple(patterns)


@torch.no_grad()
def make_mask(weight, *, mode="bidirectional", input_second_moment=None, chunk_tiles=4096):
    if weight.ndim != 2 or any(size <= 0 or size % 4 for size in weight.shape):
        raise ValueError("weight must be a nonempty 2D matrix with dimensions divisible by 4")
    if chunk_tiles <= 0:
        raise ValueError("chunk_tiles must be positive")
    scores = weight.detach().float().square()
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("weight must be finite with finite FP32 squared magnitudes")
    if input_second_moment is not None:
        moment = input_second_moment.detach().to(device=weight.device, dtype=torch.float32)
        if moment.shape != (weight.shape[1],):
            raise ValueError("input_second_moment must contain one value per input channel")
        if not bool(torch.isfinite(moment).all()) or bool((moment < 0).any()):
            raise ValueError("input_second_moment must be finite and nonnegative")
        scores = scores * moment[None, :]
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("weighted pruning scores overflowed FP32")
    if mode == "forward":
        grouped = scores.reshape(weight.shape[0], -1, 4)
        indices = grouped.argsort(dim=-1, descending=True, stable=True)[..., :2]
        return torch.zeros_like(grouped, dtype=torch.bool).scatter_(-1, indices, True).reshape_as(weight)
    if mode != "bidirectional":
        raise ValueError("unknown mask mode")
    n, k = weight.shape
    tiles = scores.reshape(n // 4, 4, k // 4, 4).permute(0, 2, 1, 3).reshape(-1, 16)
    patterns = torch.tensor(balanced_patterns(), device=weight.device, dtype=torch.float32)
    chosen = torch.empty_like(tiles, dtype=torch.bool)
    for start in range(0, tiles.shape[0], chunk_tiles):
        tile_scores = tiles[start : start + chunk_tiles]
        # Explicit products avoid TF32 matmul changing the offline pattern choice.
        totals = (tile_scores[:, None, :] * patterns[None, :, :]).sum(-1)
        chosen[start : start + chunk_tiles] = patterns[totals.argmax(-1)].bool()
    return chosen.reshape(n // 4, k // 4, 4, 4).permute(0, 2, 1, 3).reshape(n, k)


def validate_mask(mask, *, bidirectional=False):
    if mask.ndim != 2 or mask.dtype != torch.bool or any(size <= 0 or size % 4 for size in mask.shape):
        raise ValueError("mask must be a nonempty boolean matrix with dimensions divisible by 4")
    if not bool((mask.reshape(mask.shape[0], -1, 4).sum(-1) == 2).all()):
        raise ValueError("mask must retain two positions per row quartet")
    if bidirectional:
        # W[out, in]: dX uses the transpose's reduction axis, so every
        # aligned quartet of output rows must also keep two entries per input.
        tiles = mask.reshape(mask.shape[0] // 4, 4, mask.shape[1] // 4, 4)
        if not bool((tiles.sum(dim=1) == 2).all()):
            raise ValueError("mask is not 2:4 along the transpose reduction dimension")
