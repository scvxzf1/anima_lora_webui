#!/usr/bin/env python
"""Block swap H2D 传输微基准（带宽比例验证）。

测量与 block swap 等价的 pinned CPU -> GPU 拷贝：
  - bf16 负载（132 MiB，对应一个 DiT block 的 frozen 权重）
  - 1 字节负载（66 MiB，对应 fp8/int8 量化后的同一 block）

只做同 dtype DMA 拷贝，不做 GPU 端反量化，因此在任何 CUDA 设备上都能跑
（包括无新内核支持的旧卡）。反量化(mul)开销需在目标卡（3080 Ti）上另测，
预期为亚毫秒级且与 H2D 重叠。

用法：
  .venv/bin/python scripts/experiments/fp8_blockswap_h2d_microbench.py --repeats 50
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def bench_h2d(payload_bytes: int, dtype: torch.dtype, repeats: int) -> list[float]:
    """对给定字节数/类型做 pinned H2D 拷贝，返回每次毫秒数。"""
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
        ms = (time.perf_counter() - t0) * 1000.0
        if i >= 5:  # 丢弃 warmup
            times.append(ms)
    return times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-mib", type=float, default=132.0)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--out", default="/tmp/anima-fp8-blockswap/h2d_microbench_smoke.json")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "需要 CUDA 设备"
    gpu = torch.cuda.get_device_name(0)
    block_bytes = int(args.block_mib * 1024 * 1024)

    groups = {
        "bf16_h2d": (block_bytes, torch.bfloat16),
        "byte8_h2d": (block_bytes // 2, torch.uint8),  # fp8/int8 同为 1 字节负载
    }
    results = {}
    for name, (nbytes, dtype) in groups.items():
        times = bench_h2d(nbytes, dtype, args.repeats)
        results[name] = {
            "payload_mib": nbytes / 1024 / 1024,
            "p50_ms": statistics.median(times),
            "p95_ms": sorted(times)[max(0, int(len(times) * 0.95) - 1)],
            "min_ms": min(times),
            "max_ms": max(times),
        }

    ratio = results["byte8_h2d"]["p50_ms"] / results["bf16_h2d"]["p50_ms"]
    out = {"gpu": gpu, "block_mib": args.block_mib, "results": results, "byte8_vs_bf16_p50_ratio": ratio}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
