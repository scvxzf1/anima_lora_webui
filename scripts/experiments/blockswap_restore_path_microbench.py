#!/usr/bin/env python
"""Block swap restore-path 微基准。

比较 cached CUDA restore 路径里三种组件级实现：

1. `restore_loop`：逐 tensor `dst.copy_(src, non_blocking=True)`
2. `restore_foreach`：`torch._foreach_copy_(dst_list, src_list, non_blocking=True)`
3. `restore_slab`：假设 block slot 的 GPU weight storage 和 CPU master 都已 slab 化，
   每次 restore 只做一次 `gpu_slab.copy_(cpu_slab, non_blocking=True)`

与 `blockswap_copy_plan_microbench.py` 不同，这个脚本不仅测纯 H2D DMA，
还把 offloader restore 路径里的：

- `module_to_cpu.weight.data = source_master`
- `record_stream(stream)`
- 收集 `cuda_dsts / cpu_srcs / cuda_bindings`
- 最终 `module_to_cuda.weight.data = cuda_data_view`

这些 host 侧工作一起纳入时序。目标是回答：
“即使纯 DMA 几乎不变，`_foreach_copy_` 能否明显减少 restore 路径的 host issue 成本？”
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


class _WeightRef:
    def __init__(self, data: torch.Tensor) -> None:
        self.data = data


class _FakeModule:
    def __init__(self, data: torch.Tensor) -> None:
        self.weight = _WeightRef(data)


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


def _build_restore_fixture(
    shapes: list[tuple[str, tuple[int, ...]]],
) -> tuple[
    list[tuple[str, torch.Tensor, torch.Tensor]],
    dict[str, _FakeModule],
    dict[str, _FakeModule],
    dict[str, torch.Tensor],
]:
    jobs: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    modules_to_cpu: dict[str, _FakeModule] = {}
    modules_to_cuda: dict[str, _FakeModule] = {}
    gpu_views: dict[str, torch.Tensor] = {}
    for name, shape in shapes:
        gpu_view = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
        source_master = torch.randn(shape, dtype=torch.bfloat16).pin_memory()
        target_master = torch.randn(shape, dtype=torch.bfloat16).pin_memory()
        modules_to_cpu[name] = _FakeModule(gpu_view)
        modules_to_cuda[name] = _FakeModule(torch.empty(shape, dtype=torch.bfloat16, device="cuda"))
        gpu_views[name] = gpu_view
        jobs.append((name, source_master, target_master))
    return jobs, modules_to_cpu, modules_to_cuda, gpu_views


def _pack_cpu_slab(
    jobs: list[tuple[str, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, dict[str, tuple[int, int, tuple[int, ...]]]]:
    meta: dict[str, tuple[int, int, tuple[int, ...]]] = {}
    offset = 0
    for name, _, target_master in jobs:
        flat = target_master.view(-1)
        numel = int(flat.numel())
        meta[name] = (offset, numel, tuple(target_master.shape))
        offset += numel
    slab = torch.empty(offset, dtype=torch.bfloat16).pin_memory()
    for name, _, target_master in jobs:
        start, numel, _ = meta[name]
        slab[start : start + numel].copy_(target_master.view(-1), non_blocking=False)
    return slab, meta


def _make_gpu_views_from_slab(
    slab: torch.Tensor, meta: dict[str, tuple[int, int, tuple[int, ...]]]
) -> dict[str, torch.Tensor]:
    views: dict[str, torch.Tensor] = {}
    for name, (start, numel, shape) in meta.items():
        views[name] = slab.narrow(0, start, numel).view(shape)
    return views


def _summarize(vals: list[float]) -> dict[str, float]:
    ordered = sorted(vals)
    return {
        "count": len(ordered),
        "avg_ms": sum(ordered) / len(ordered),
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[max(0, int(len(ordered) * 0.95) - 1)],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _run_variant(
    *,
    jobs: list[tuple[str, torch.Tensor, torch.Tensor]],
    modules_to_cpu: dict[str, _FakeModule],
    modules_to_cuda: dict[str, _FakeModule],
    gpu_views: dict[str, torch.Tensor],
    stream: torch.cuda.Stream,
    use_foreach: bool,
) -> tuple[float, float, float]:
    for name, _, _ in jobs:
        modules_to_cpu[name].weight.data = gpu_views[name]

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    host_t0 = time.perf_counter()
    with torch.cuda.stream(stream):
        start_evt.record(stream)
        cuda_dsts: list[torch.Tensor] = []
        cpu_srcs: list[torch.Tensor] = []
        cuda_bindings: list[tuple[_FakeModule, torch.Tensor]] = []
        for name, source_master, target_master in jobs:
            module_to_cpu = modules_to_cpu[name]
            module_to_cuda = modules_to_cuda[name]
            cuda_data_view = module_to_cpu.weight.data
            module_to_cpu.weight.data = source_master
            cuda_data_view.record_stream(stream)
            cuda_dsts.append(cuda_data_view)
            cpu_srcs.append(target_master)
            cuda_bindings.append((module_to_cuda, cuda_data_view))
        if use_foreach:
            torch._foreach_copy_(cuda_dsts, cpu_srcs, non_blocking=True)
        else:
            for cuda_data_view, target_master in zip(cuda_dsts, cpu_srcs):
                cuda_data_view.copy_(target_master, non_blocking=True)
        for module_to_cuda, cuda_data_view in cuda_bindings:
            module_to_cuda.weight.data = cuda_data_view
        end_evt.record(stream)
    host_issue_ms = (time.perf_counter() - host_t0) * 1000.0
    end_evt.synchronize()
    ready_ms = (time.perf_counter() - host_t0) * 1000.0
    gpu_copy_ms = float(start_evt.elapsed_time(end_evt))
    return host_issue_ms, ready_ms, gpu_copy_ms


def _bench_restore_path(
    shapes: list[tuple[str, tuple[int, ...]]], repeats: int
) -> dict[str, dict[str, dict[str, float]]]:
    if getattr(torch, "_foreach_copy_", None) is None:
        raise RuntimeError("torch._foreach_copy_ 不可用，无法比较 restore_foreach")

    jobs, modules_to_cpu, modules_to_cuda, gpu_views = _build_restore_fixture(shapes)
    cpu_slab, slab_meta = _pack_cpu_slab(jobs)
    gpu_slab = torch.empty_like(cpu_slab, device="cuda")
    gpu_slab_views = _make_gpu_views_from_slab(gpu_slab, slab_meta)
    stream = torch.cuda.Stream()

    metrics: dict[str, dict[str, list[float]]] = {
        "restore_loop": {
            "host_issue_ms": [],
            "ready_ms": [],
            "gpu_copy_ms": [],
        },
        "restore_foreach": {
            "host_issue_ms": [],
            "ready_ms": [],
            "gpu_copy_ms": [],
        },
        "restore_slab": {
            "host_issue_ms": [],
            "ready_ms": [],
            "gpu_copy_ms": [],
        },
    }

    variants = [
        ("restore_loop", False),
        ("restore_foreach", True),
        ("restore_slab", None),
    ]
    for idx in range(repeats + 6):
        order = variants if idx % 2 == 0 else list(reversed(variants))
        for label, use_foreach in order:
            if label == "restore_slab":
                for name, source_master, _ in jobs:
                    modules_to_cpu[name].weight.data = gpu_slab_views[name]
                    modules_to_cuda[name].weight.data = gpu_slab_views[name]
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                host_t0 = time.perf_counter()
                with torch.cuda.stream(stream):
                    start_evt.record(stream)
                    for name, source_master, _ in jobs:
                        modules_to_cpu[name].weight.data = source_master
                    gpu_slab.record_stream(stream)
                    gpu_slab.copy_(cpu_slab, non_blocking=True)
                    for name, _, _ in jobs:
                        modules_to_cuda[name].weight.data = gpu_slab_views[name]
                    end_evt.record(stream)
                host_issue_ms = (time.perf_counter() - host_t0) * 1000.0
                end_evt.synchronize()
                ready_ms = (time.perf_counter() - host_t0) * 1000.0
                gpu_copy_ms = float(start_evt.elapsed_time(end_evt))
            else:
                host_issue_ms, ready_ms, gpu_copy_ms = _run_variant(
                    jobs=jobs,
                    modules_to_cpu=modules_to_cpu,
                    modules_to_cuda=modules_to_cuda,
                    gpu_views=gpu_views,
                    stream=stream,
                    use_foreach=bool(use_foreach),
                )
            if idx >= 6:
                metrics[label]["host_issue_ms"].append(host_issue_ms)
                metrics[label]["ready_ms"].append(ready_ms)
                metrics[label]["gpu_copy_ms"].append(gpu_copy_ms)

    return {
        label: {metric: _summarize(vals) for metric, vals in label_metrics.items()}
        for label, label_metrics in metrics.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--block-idx", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument(
        "--out",
        default="/tmp/anima-blockswap-copy-plan/restore_path_microbench.json",
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "需要 CUDA 设备"
    shapes = _load_block_shapes(args.model, args.block_idx)
    total_bytes = 0
    for _, shape in shapes:
        numel = 1
        for dim in shape:
            numel *= dim
        total_bytes += numel * torch.tensor([], dtype=torch.bfloat16).element_size()

    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "block_idx": args.block_idx,
        "tensor_count": len(shapes),
        "total_bytes": total_bytes,
        "total_mib": total_bytes / 1024 / 1024,
        "results": _bench_restore_path(shapes, args.repeats),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
