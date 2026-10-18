"""Pure H2D copy micro-benchmark for the block-swap restore path.

Answers one question without loading any model: for a realistic block payload,
is the measured restore cost limited by the PCIe link or by per-tensor copy
overhead?

It mirrors what the offloader does: pinned/unpinned CPU source, a dedicated copy
stream, non_blocking copies, CUDA-event timing, median over N iterations. The
payload is split into 1..N chunks to expose per-copy overhead.

Usage:
    .venv/bin/python -m bench.block_swap_pipeline.copy_bench
    .venv/bin/python -m bench.block_swap_pipeline.copy_bench --sizes-mb 100,50 --chunks 1,8,24
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def _bench_one(size_bytes: int, chunks: int, pin: bool, iters: int, device: torch.device) -> dict:
    element = 2  # bf16 payload
    per_chunk = max(element, (size_bytes // chunks) // element * element)
    cpu_tensors = []
    for _ in range(chunks):
        t = torch.empty(per_chunk // element, dtype=torch.bfloat16, device="cpu")
        if pin:
            t = t.pin_memory()
        cpu_tensors.append(t)
    gpu_tensors = [torch.empty_like(t, device=device) for t in cpu_tensors]
    stream = torch.cuda.Stream(device=device)

    timings = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        with torch.cuda.stream(stream):
            start.record(stream)
            for dst, src in zip(gpu_tensors, cpu_tensors):
                dst.copy_(src, non_blocking=True)
            end.record(stream)
        stream.synchronize()
        timings.append(start.elapsed_time(end))

    moved = per_chunk * chunks
    median_ms = statistics.median(timings)
    return {
        "size_mb": round(size_bytes / 1e6, 1),
        "moved_mb": round(moved / 1e6, 1),
        "chunks": chunks,
        "pinned": pin,
        "median_ms": round(median_ms, 3),
        "gb_per_s": round((moved / 1e9) / (median_ms / 1000.0), 2),
        "per_chunk_us": round(median_ms * 1000.0 / chunks, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes-mb", default="100,50,200")
    parser.add_argument("--chunks", default="1,8,24")
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--root", default="output/bench/block_swap_pipeline")
    args = parser.parse_args()

    device = torch.device("cuda", args.gpu)
    torch.cuda.set_device(device)
    print(json.dumps({
        "gpu": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
    }), flush=True)

    rows = []
    for size_mb in (int(s) for s in args.sizes_mb.split(",") if s.strip()):
        for chunks in (int(c) for c in args.chunks.split(",") if c.strip()):
            for pin in (True, False):
                row = _bench_one(size_mb * 1_000_000, chunks, pin, args.iters, device)
                rows.append(row)
                print(json.dumps(row), flush=True)

    out_dir = Path(args.root)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    result = out_dir / f"{stamp}-copy-bench.json"
    result.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2), encoding="utf-8")

    print("", flush=True)
    print("size_mb  chunks  pinned  median_ms  GB/s   us/chunk", flush=True)
    for r in rows:
        print(f"{r['size_mb']:>7}  {r['chunks']:>6}  {str(r['pinned']):>6}  "
              f"{r['median_ms']:>9}  {r['gb_per_s']:>5}  {r['per_chunk_us']:>8}", flush=True)
    print(f"[copy-bench] -> {result}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
