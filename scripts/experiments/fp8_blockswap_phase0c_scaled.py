#!/usr/bin/env python
"""Phase 0c: block swap 传输压缩格式的离线数值对比。

背景：raw fp8_e4m3 传输（relative_l2 p95 约 8.8%）与保守 per-tensor scaled FP8
（约 2.66%）都没过 2% 门槛（docs/findings/anima_fp8_blockswap_transfer_report.md）。
本脚本在真实 Anima DiT 权重上回答下一轮问题：

  1. per-row（输出通道）scaled FP8 能否压到 2% 以下？
  2. 同等 1 字节带宽的 int8 + per-row scale 是否数值上更优？

只做离线统计，不改训练路径。所有方案的反量化都等价于
``dequant = quant.to(float32) * scale``，与未来 H2D 后 GPU 端 mul 的实现一致。

用法：
  .venv/bin/python scripts/experiments/fp8_blockswap_phase0c_scaled.py \
      --model /path/to/anima-preview3-base.safetensors \
      --out-dir /tmp/anima-fp8-blockswap
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)  # 448.0
INT8_MAX = 127.0
BLOCK_KEY_RE = re.compile(r"^net\.blocks\.(\d+)\.(.+)$")


@dataclass
class TensorStats:
    """单个张量在单个方案下的量化误差统计。"""

    scheme: str
    block_idx: int
    name: str
    numel: int
    ndim: int
    quantized: bool  # False 表示该方案下豁免（保持 bf16）
    payload_bytes: int  # 传输负载：量化字节 + scale 字节（豁免时为 bf16 字节）
    relative_l2: float
    mean_abs_error: float
    max_abs_error: float
    would_saturate: int  # scale 后仍超出格式上限的元素数


def _row_view(w: torch.Tensor) -> torch.Tensor:
    """把 2D+ 张量看成 (out_rows, -1)，per-row scale 沿 dim0。"""
    return w.reshape(w.shape[0], -1)


def _scale_bytes(num_scales: int) -> int:
    return num_scales * 4  # scale 以 float32 存储


def _quant_error(
    w: torch.Tensor,
    scale: torch.Tensor,
    fmt: str,
) -> tuple[torch.Tensor, int]:
    """按给定 scale 量化/反量化，返回 (dequant, would_saturate)。

    w 与 scale 已广播对齐；fmt 为 "fp8" 或 "int8"。
    """
    target = w / scale
    if fmt == "fp8":
        saturate = int((target.abs() > FP8_MAX).sum().item())
        deq = target.to(torch.float8_e4m3fn).to(torch.float32) * scale
    elif fmt == "int8":
        saturate = int((target.abs().round() > INT8_MAX).sum().item())
        q = target.round().clamp_(-INT8_MAX, INT8_MAX)
        deq = q * scale
    else:
        raise ValueError(fmt)
    return deq, saturate


def evaluate_tensor(
    scheme: str, block_idx: int, name: str, w_bf16: torch.Tensor
) -> TensorStats:
    """对单个张量执行某方案并统计误差。

    方案约定：
    - raw_fp8 / fp8_per_tensor / int8_per_tensor：量化所有张量（与上一轮口径一致）。
    - *_per_row：仅量化 2D+ 张量；1D（norm）豁免保持 bf16，因为字节占比可忽略。
    """
    w = w_bf16.to(torch.float32)
    numel = w.numel()
    bf16_bytes = numel * 2
    base = dict(
        scheme=scheme,
        block_idx=block_idx,
        name=name,
        numel=numel,
        ndim=w.dim(),
    )

    per_row = scheme.endswith("_per_row")
    if per_row and w.dim() < 2:
        return TensorStats(
            **base,
            quantized=False,
            payload_bytes=bf16_bytes,
            relative_l2=0.0,
            mean_abs_error=0.0,
            max_abs_error=0.0,
            would_saturate=0,
        )

    if scheme == "raw_fp8":
        deq = w.to(torch.float8_e4m3fn).to(torch.float32)
        saturate = int((w.abs() > FP8_MAX).sum().item())
        payload = numel * 1
    else:
        fmt = "fp8" if "fp8" in scheme else "int8"
        # fp8 用 amax*pad 映射到格式上限留 1% 余量防边界饱和；int8 直接 amax/127。
        if fmt == "fp8":
            pad = 1.05 if scheme == "fp8_per_tensor" else 1.01
            limit = FP8_MAX / pad
        else:
            limit = INT8_MAX
        if per_row:
            rows = _row_view(w)
            amax = rows.abs().amax(dim=1, keepdim=True)
            scale = (amax / limit).clamp_min(1e-12)
            deq2d, saturate = _quant_error(rows, scale, fmt)
            deq = deq2d.reshape(w.shape)
            payload = numel * 1 + _scale_bytes(rows.shape[0])
        else:
            amax = w.abs().amax()
            scale = (amax / limit).clamp_min(1e-12)
            deq, saturate = _quant_error(w, scale, fmt)
            payload = numel * 1 + _scale_bytes(1)

    diff = deq - w
    denom = float(w.norm().item())
    return TensorStats(
        **base,
        quantized=True,
        payload_bytes=payload,
        relative_l2=float(diff.norm().item()) / denom if denom > 0 else 0.0,
        mean_abs_error=float(diff.abs().mean().item()),
        max_abs_error=float(diff.abs().max().item()),
        would_saturate=saturate,
    )


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, math.ceil(q / 100.0 * len(s)) - 1))
    return s[idx]


def summarize(scheme: str, stats: list[TensorStats], bf16_total: int) -> dict:
    quantized = [s for s in stats if s.quantized]
    rel = [s.relative_l2 for s in quantized]
    by_block: dict[int, list[float]] = {}
    for s in quantized:
        by_block.setdefault(s.block_idx, []).append(s.relative_l2)
    block_means = [sum(v) / len(v) for v in by_block.values()]
    payload = sum(s.payload_bytes for s in stats)
    return {
        "scheme": scheme,
        "tensors_quantized": len(quantized),
        "tensors_exempt": len(stats) - len(quantized),
        "relative_l2_p50": percentile(rel, 50),
        "relative_l2_p95": percentile(rel, 95),
        "relative_l2_max": max(rel) if rel else 0.0,
        "block_relative_l2_p50": percentile(block_means, 50),
        "block_relative_l2_p95": percentile(block_means, 95),
        "block_relative_l2_max": max(block_means) if block_means else 0.0,
        "mean_abs_error_p95": percentile([s.mean_abs_error for s in quantized], 95),
        "max_abs_error_p95": percentile([s.max_abs_error for s in quantized], 95),
        "would_saturate_tensors": sum(1 for s in quantized if s.would_saturate),
        "payload_bytes": payload,
        "payload_gib": payload / 1024**3,
        "payload_ratio_vs_bf16": payload / bf16_total if bf16_total else 0.0,
    }


SCHEMES = [
    "raw_fp8",  # 校准：应复现报告的 ~8.8% p95
    "fp8_per_tensor",  # 校准：应接近报告保守 scaled 的 ~2.66%
    "fp8_per_row",  # 本轮假设 1：报告建议的更细粒度 scaling
    "int8_per_tensor",  # 对照
    "int8_per_row",  # 本轮假设 2：同带宽备选格式
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="/home/scv/nvme0n1p1/ComfyUI/models/diffusion_models/anima/anima-preview3-base.safetensors",
    )
    parser.add_argument("--out-dir", default="/tmp/anima-fp8-blockswap")
    parser.add_argument("--gate-pct", type=float, default=2.0, help="通过门槛（%%）")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / "phase0c_scaled_formats.jsonl"
    summary_path = out_dir / "phase0c_scaled_formats_summary.json"

    t0 = time.perf_counter()
    all_stats: dict[str, list[TensorStats]] = {s: [] for s in SCHEMES}
    bf16_total = 0
    tensor_count = 0

    with safe_open(args.model, framework="pt", device="cpu") as f, detail_path.open(
        "w", encoding="utf-8"
    ) as detail:
        block_keys = sorted(
            (k for k in f.keys() if BLOCK_KEY_RE.match(k)),
            key=lambda k: (
                int(BLOCK_KEY_RE.match(k).group(1)),
                BLOCK_KEY_RE.match(k).group(2),
            ),
        )
        for key in block_keys:
            m = BLOCK_KEY_RE.match(key)
            block_idx, name = int(m.group(1)), m.group(2)
            w = f.get_tensor(key)
            bf16_total += w.numel() * 2
            tensor_count += 1
            for scheme in SCHEMES:
                st = evaluate_tensor(scheme, block_idx, name, w)
                all_stats[scheme].append(st)
                detail.write(json.dumps(st.__dict__, ensure_ascii=False) + "\n")
            del w

    summaries = [summarize(s, all_stats[s], bf16_total) for s in SCHEMES]
    elapsed = time.perf_counter() - t0

    print(f"模型: {args.model}")
    print(
        f"枚举: {tensor_count} 个 swappable frozen 张量, "
        f"bf16 总量 {bf16_total / 1024**3:.4f} GiB, 耗时 {elapsed:.1f}s"
    )
    gate = args.gate_pct / 100.0
    header = (
        f"{'scheme':18s} {'t_p50':>7s} {'t_p95':>7s} {'t_max':>7s} "
        f"{'blk_p95':>8s} {'sat':>4s} {'payload':>9s} {'ratio':>6s} {'gate<2%':>8s}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        ok = s["relative_l2_p95"] < gate and s["block_relative_l2_p95"] < gate
        print(
            f"{s['scheme']:18s} "
            f"{s['relative_l2_p50'] * 100:6.3f}% {s['relative_l2_p95'] * 100:6.3f}% "
            f"{s['relative_l2_max'] * 100:6.3f}% {s['block_relative_l2_p95'] * 100:7.3f}% "
            f"{s['would_saturate_tensors']:4d} {s['payload_gib']:8.4f}G "
            f"{s['payload_ratio_vs_bf16']:6.3f} {'PASS' if ok else 'FAIL':>8s}"
        )

    summary_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "tensor_count": tensor_count,
                "bf16_total_bytes": bf16_total,
                "gate_pct": args.gate_pct,
                "schemes": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"明细: {detail_path}")
    print(f"汇总: {summary_path}")


if __name__ == "__main__":
    main()
