"""How much can per-row INT8 scale optimisation actually buy?

The int8 block-swap CPU master stores scale = amax / 127 per output channel. The
proposal for this round assumed a single 3.5-5 sigma outlier inflates that scale
enough that a per-row clip search would cut the quantisation error by 60-80%.

This probe measures the truth on both synthetic heavy-tailed rows and real Anima
DiT weights:

  * baseline   : scale = amax / 127 (shipped encoder)
  * clip grid  : the opt-in L2 search over 0.85..1.00 amax + the 99.95 percentile
  * wide bound : per-row best over 0.30..1.00 amax in 0.01 steps, i.e. the
                 unconstrained upper bound any clip-based encoder could reach

If the wide bound only moves the relative L2 error by a fraction of a percent,
then outlier clipping is not where the int8 error budget lives.

Usage:
    .venv/bin/python -m bench.block_swap_int8.encode_error --dit <weights.safetensors>
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from statistics import NormalDist

import torch

from library.runtime.int8_encode import quantize_weight_per_channel_clipped
from library.runtime.int8_linear import INT8_MAX, quantize_weight_per_channel

DEFAULT_DIT = (
    "/home/scv/nvme0n1p1/ComfyUI/models/diffusion_models/anima/anima-base-v1.0-sha.safetensors"
)
WANTED_MARKERS = ("mlp.layer1", "mlp.layer2", "self_attn.qkv_proj", "self_attn.out_proj", "proj.ffn_down")


def relative_error(weight: torch.Tensor, codes: torch.Tensor, scale: torch.Tensor) -> float:
    restored = codes.float() * scale.float()[:, None]
    target = weight.detach().float()
    return float(((restored - target) ** 2).sum() / (target ** 2).sum())


def wide_bound(weight: torch.Tensor, *, lo: float = 0.30, hi: float = 1.00, step: float = 0.01) -> float:
    """Best relative error over an unconstrained per-row amax fraction search."""
    rows = weight.detach().float()
    amax = rows.abs().amax(dim=1)
    best = None
    fractions = [lo + index * step for index in range(int(round((hi - lo) / step)) + 1)]
    for fraction in fractions:
        scale = ((amax * fraction).clamp_min(1e-12)) / INT8_MAX
        codes = (rows / scale[:, None]).round().clamp(-INT8_MAX, INT8_MAX)
        error = ((codes * scale[:, None] - rows) ** 2).sum(dim=1)
        best = error if best is None else torch.minimum(best, error)
    return float(best.sum() / (rows ** 2).sum())


def lloyd_max_bound(weight: torch.Tensor, *, levels: int = 256, rows: int = 64, iters: int = 12) -> float:
    """MSE of an optimal per-row 256-level quantizer (1D Lloyd-Max / k-means).

    This upper-bounds anything a fixed NormalFloat-style codebook (plan 3B) could
    reach, because Lloyd-Max uses the empirically optimal centroids per row.
    """
    data = weight.detach().float()[:rows]
    total_error = 0.0
    total_energy = 0.0
    for row in data:
        values = row
        lo, hi = float(values.min()), float(values.max())
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            continue
        centers = torch.linspace(lo, hi, levels)
        for _ in range(iters):
            index = torch.bucketize(values, centers)
            index = index.clamp(0, levels - 1)
            left = centers[index - 1] if levels > 1 else centers
            # pick the closer of the two neighbouring centres
            take_left = (index > 0) & ((values - left).abs() < (values - centers[index]).abs())
            assign = torch.where(take_left, index - 1, index)
            new_centers = centers.clone()
            sums = torch.zeros(levels, dtype=torch.float32).index_add_(0, assign, values)
            counts = torch.zeros(levels, dtype=torch.float32).index_add_(
                0, assign, torch.ones_like(values)
            )
            occupied = counts > 0
            new_centers[occupied] = sums[occupied] / counts[occupied]
            if torch.equal(new_centers, centers):
                break
            centers = new_centers
        error = ((values - centers[assign]) ** 2).sum()
        total_error += float(error)
        total_energy += float((values ** 2).sum())
    if total_energy <= 0:
        return float("nan")
    return total_error / total_energy


def nf8_relative_error(weight: torch.Tensor) -> float:
    """Relative L2 error of the library NF8 encoder (byte-identical to int8)."""
    from library.runtime.int8_encode import dequantize_nf8, quantize_weight_per_channel_nf8

    rows = weight.detach().float()
    codes, scale = quantize_weight_per_channel_nf8(rows)
    restored = dequantize_nf8(codes, scale)
    return float(((restored - rows) ** 2).sum() / (rows ** 2).sum())


def synthetic_cases() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    cases: dict[str, torch.Tensor] = {}
    rows = torch.randn(64, 512)
    rows[3, 17] = 14.0
    cases["1x14sigma_outlier_w512"] = rows
    rows = torch.randn(64, 512)
    rows[3, :25] = rows[3, :25].sign() * 5.0
    cases["25x5sigma_heavy_tail_w512"] = rows
    cases["pure_gaussian_w512"] = torch.randn(64, 512)
    rows = torch.randn(64, 64)
    rows[3, :3] = 8.0
    cases["3x8sigma_narrow_w64"] = rows
    return cases


def real_cases(dit: Path, limit: int) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    cases: dict[str, torch.Tensor] = {}
    with safe_open(str(dit), framework="pt", device="cpu") as handle:
        keys = [
            key
            for key in handle.keys()
            if key.endswith(".weight") and any(marker in key for marker in WANTED_MARKERS)
        ]
        if not keys:
            keys = [key for key in handle.keys() if key.endswith(".weight")]
        stride = max(1, len(keys) // max(1, limit))
        for key in keys[::stride][:limit]:
            tensor = handle.get_tensor(key)
            if tensor.dim() == 2 and min(tensor.shape) >= 64:
                cases[key] = tensor.float()
    return cases


def measure(name: str, weight: torch.Tensor) -> dict:
    base_codes, base_scale = quantize_weight_per_channel(weight)
    clip_codes, clip_scale = quantize_weight_per_channel_clipped(weight)
    base = relative_error(weight, base_codes, base_scale)
    clip = relative_error(weight, clip_codes, clip_scale)
    wide = wide_bound(weight)
    lloyd = lloyd_max_bound(weight)
    nf8 = nf8_relative_error(weight)
    return {
        "case": name,
        "nf8_rel_error": nf8,
        "nf8_gain_percent": 100.0 * (1.0 - nf8 / base),
        "lloyd_max_rel_error": lloyd,
        "lloyd_gain_percent": 100.0 * (1.0 - lloyd / base),
        "shape": list(weight.shape),
        "baseline_rel_error": base,
        "clip_grid_rel_error": clip,
        "wide_bound_rel_error": wide,
        "clip_gain_percent": 100.0 * (1.0 - clip / base),
        "wide_gain_percent": 100.0 * (1.0 - wide / base),
        "rows_where_clip_wins": int((clip_scale < base_scale).sum()),
        "rows": int(weight.shape[0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dit", default=DEFAULT_DIT)
    parser.add_argument("--real-limit", type=int, default=4)
    parser.add_argument("--root", default="output/bench/block_swap_int8")
    args = parser.parse_args()

    rows = [measure(name, weight) for name, weight in synthetic_cases().items()]
    dit = Path(args.dit)
    if dit.exists():
        for name, weight in real_cases(dit, args.real_limit).items():
            rows.append(measure(name, weight))
    else:
        print(f"[encode-error] real weights not found, synthetic only: {dit}")

    out_dir = Path(args.root)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-encode-error.json"
    result.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2), encoding="utf-8")

    print(f"{'case':40s} {'base':>10s} {'clip%':>6s} {'wide%':>6s} {'nf8%':>6s} {'lloyd%':>7s}")
    for row in rows:
        print(
            f"{row['case'][:40]:40s} {row['baseline_rel_error']:10.3e} "
            f"{row['clip_gain_percent']:6.2f} {row['wide_gain_percent']:6.2f} "
            f"{row['nf8_gain_percent']:6.2f} {row['lloyd_gain_percent']:7.2f}"
        )
    print(f"[encode-error] -> {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
