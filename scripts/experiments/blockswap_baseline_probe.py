#!/usr/bin/env python
"""块交换「计算 vs 传输」基线探针。

在真实 Anima DiT 块尺寸上测量三块指标，作为块交换优化方向的量化参照：

1. `block_fwd_ms`：单个 DiT block 前向计算时间（决定传输能否被计算隐藏）。
2. `h2d_ms`：一个 block 的 frozen 权重 pinned CPU→GPU H2D 时间（bf16 与 int8/fp8 各一份）。
3. `overlap_ratio`：`block_fwd_ms / h2d_ms`。
   - >= 1：bf16 传输可被单块计算完全隐藏，预取深度收益有限。
   - < 1：传输超过单块计算，需要更大预取深度（方向1）或更小传输量（方向4 int8）。

这是基准确认工具，不修改任何运行时代码。结果落盘 JSON 供 docs/findings 引用。

用法：
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/experiments/blockswap_baseline_probe.py
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/experiments/blockswap_baseline_probe.py --repeats 40
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def _median_block_fwd_ms(seq_len: int, repeats: int) -> float:
    """测真实 Block 在 native-flatten 形状下的前向时间。"""
    from library.anima.models import Block
    from networks.attention_dispatch import AttentionParams

    dev, dt = "cuda", torch.bfloat16
    blk = Block(x_dim=1536, context_dim=1536, num_heads=12, mlp_ratio=4.0).to(dev, dt).eval()
    attn_params = AttentionParams.create_attention_params("torch")
    x = torch.randn(1, 1, seq_len, 1, 1536, device=dev, dtype=dt)
    emb = torch.randn(1, 1, 1536, device=dev, dtype=dt)
    ctx = torch.randn(1, 512, 1536, device=dev, dtype=dt)
    with torch.no_grad():
        for _ in range(5):
            _ = blk(x, emb, ctx, attn_params, None, None, False)
        torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = blk(x, emb, ctx, attn_params, None, None, False)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def _median_h2d_ms(payload_bytes: int, dtype: torch.dtype, repeats: int) -> float:
    """对给定字节数/类型做 pinned H2D 拷贝，返回中位毫秒。"""
    elem = torch.tensor([], dtype=dtype).element_size()
    numel = payload_bytes // elem
    src = torch.zeros(numel, dtype=dtype).pin_memory()
    dst = torch.empty(numel, dtype=dtype, device="cuda")
    times: list[float] = []
    for i in range(repeats + 5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        if i >= 5:
            times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-mib", type=float, default=132.0)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--out", default="/tmp/anima-blockswap-baseline/baseline_probe.json")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "需要 CUDA 设备"
    gpu = torch.cuda.get_device_name(0)
    block_bytes = int(args.block_mib * 1024 * 1024)

    block_fwd_ms = _median_block_fwd_ms(args.seq_len, args.repeats)
    bf16_h2d_ms = _median_h2d_ms(block_bytes, torch.bfloat16, args.repeats)
    byte8_h2d_ms = _median_h2d_ms(block_bytes // 2, torch.uint8, args.repeats)

    payload = {
        "gpu": gpu,
        "block_mib": args.block_mib,
        "seq_len": args.seq_len,
        "block_fwd_ms": round(block_fwd_ms, 3),
        "bf16_h2d_ms": round(bf16_h2d_ms, 3),
        "byte8_h2d_ms": round(byte8_h2d_ms, 3),
        "overlap_ratio_bf16": round(block_fwd_ms / bf16_h2d_ms, 3),
        "overlap_ratio_byte8": round(block_fwd_ms / byte8_h2d_ms, 3),
        "interpretation": (
            "overlap_ratio>=1 → bf16 传输可隐藏在单块计算内；"
            "<1 → 需更大预取深度(方向1)或更小传输量(方向4 int8)。"
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
