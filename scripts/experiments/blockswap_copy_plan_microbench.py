#!/usr/bin/env python
"""Block swap copy-plan 微基准。

比较三种等价的 pinned CPU -> GPU 恢复路径：

1. `loop_copy`：逐 tensor `dst.copy_(src, non_blocking=True)`
2. `foreach_copy`：`torch._foreach_copy_(dst_list, src_list)`
3. `slab_copy`：把同一 block 的权重打包到一个连续 CPU slab，再整体 H2D 到 GPU slab，
   最后仅在 GPU 上切 view 绑定回原 shape

目标不是替代正式训练，而是快速判断“减少小 H2D 次数”这半段值不值得接入 runtime。
默认直接使用 Anima preview3 block0 的真实 frozen 权重形状，总载荷约 132 MiB。

用法：
  .venv/bin/python scripts/experiments/blockswap_copy_plan_microbench.py
  .venv/bin/python scripts/experiments/blockswap_copy_plan_microbench.py --repeats 50
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from safetensors import safe_open


DEFAULT_MODEL = (
    "/home/scv/nvme0n1p1/ComfyUI/models/diffusion_models/anima/"
    "anima-preview3-base.safetensors"
)


def _load_block_shapes(model_path: str, block_idx: int) -> list[tuple[str, tuple[int, ...]]]:
    prefix = f"net.blocks.{block_idx}."
    items: list[tuple[str, tuple[int, ...]]] = []
    with safe_open(model_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(prefix) and key.endswith(".weight"):
                shape = tuple(int(x) for x in f.get_slice(key).get_shape())
                items.append((key, shape))
    if not items:
        raise RuntimeError(f"no weights found for {prefix} in {model_path}")
    items.sort()
    return items


def _alloc_sources(shapes: list[tuple[str, tuple[int, ...]]]) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    for _, shape in shapes:
        tensors.append(torch.randn(shape, dtype=torch.bfloat16).pin_memory())
    return tensors


def _alloc_dests_like(srcs: list[torch.Tensor]) -> list[torch.Tensor]:
    return [torch.empty_like(src, device="cuda") for src in srcs]


def _pack_cpu_slab(srcs: list[torch.Tensor]) -> tuple[torch.Tensor, list[tuple[int, int, tuple[int, ...]]]]:
    meta: list[tuple[int, int, tuple[int, ...]]] = []
    offset = 0
    flat_srcs: list[torch.Tensor] = []
    for src in srcs:
        flat = src.view(-1)
        flat_srcs.append(flat)
        numel = int(flat.numel())
        meta.append((offset, numel, tuple(src.shape)))
        offset += numel
    slab = torch.empty(offset, dtype=torch.bfloat16).pin_memory()
    cursor = 0
    for flat in flat_srcs:
        numel = int(flat.numel())
        slab[cursor : cursor + numel].copy_(flat, non_blocking=False)
        cursor += numel
    return slab, meta


def _bind_gpu_views(
    slab: torch.Tensor, meta: list[tuple[int, int, tuple[int, ...]]]
) -> list[torch.Tensor]:
    views: list[torch.Tensor] = []
    for offset, numel, shape in meta:
        view = slab.narrow(0, offset, numel).view(shape)
        views.append(view)
    return views


def _bench_loop_copy(srcs: list[torch.Tensor], repeats: int) -> list[float]:
    dsts = _alloc_dests_like(srcs)
    times: list[float] = []
    for idx in range(repeats + 5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for dst, src in zip(dsts, srcs):
            dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        if idx >= 5:
            times.append(ms)
    return times


def _bench_foreach_copy(srcs: list[torch.Tensor], repeats: int) -> list[float]:
    dsts = _alloc_dests_like(srcs)
    times: list[float] = []
    for idx in range(repeats + 5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        torch._foreach_copy_(dsts, srcs)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        if idx >= 5:
            times.append(ms)
    return times


def _bench_slab_copy(srcs: list[torch.Tensor], repeats: int) -> list[float]:
    cpu_slab, meta = _pack_cpu_slab(srcs)
    gpu_slab = torch.empty_like(cpu_slab, device="cuda")
    times: list[float] = []
    for idx in range(repeats + 5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gpu_slab.copy_(cpu_slab, non_blocking=True)
        torch.cuda.synchronize()
        _ = _bind_gpu_views(gpu_slab, meta)
        ms = (time.perf_counter() - t0) * 1000.0
        if idx >= 5:
            times.append(ms)
    return times


def _summarize(times: list[float]) -> dict[str, float]:
    vals = sorted(times)
    return {
        "count": len(vals),
        "p50_ms": statistics.median(vals),
        "p95_ms": vals[max(0, int(len(vals) * 0.95) - 1)],
        "min_ms": vals[0],
        "max_ms": vals[-1],
        "avg_ms": sum(vals) / len(vals),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--block-idx", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--out",
        default="/tmp/anima-blockswap-copy-plan/microbench.json",
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "需要 CUDA 设备"
    shapes = _load_block_shapes(args.model, args.block_idx)
    srcs = _alloc_sources(shapes)
    total_bytes = sum(int(src.numel() * src.element_size()) for src in srcs)

    results = {
        "loop_copy": _summarize(_bench_loop_copy(srcs, args.repeats)),
        "foreach_copy": _summarize(_bench_foreach_copy(srcs, args.repeats)),
        "slab_copy": _summarize(_bench_slab_copy(srcs, args.repeats)),
    }
    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "block_idx": args.block_idx,
        "tensor_count": len(srcs),
        "total_bytes": total_bytes,
        "total_mib": total_bytes / 1024 / 1024,
        "results": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
