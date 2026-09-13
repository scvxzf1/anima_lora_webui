"""Measure actual-size NCCL collectives, PP round trips and host transfers."""

from __future__ import annotations

import argparse
from datetime import timedelta
from functools import partial
import os
from pathlib import Path
import statistics
import subprocess
from time import perf_counter

import torch
import torch.distributed as dist

from .common import gather_objects, hardware_record, write_json


def exchange(tensor, buffer, rank):
    if rank == 0:
        operations = ((dist.isend, tensor), (dist.irecv, buffer))
    else:
        operations = ((dist.irecv, buffer), (dist.isend, tensor))
    for operation, value in operations:
        requests = dist.batch_isend_irecv([dist.P2POp(operation, value, 1 - rank)])
        for request in requests:
            request.wait()


def measure(operation, device, warmup, repeats):
    samples = []
    for index in range(warmup + repeats):
        dist.barrier()
        torch.cuda.synchronize(device)
        start = perf_counter()
        operation()
        torch.cuda.synchronize(device)
        elapsed = perf_counter() - start
        if index >= warmup:
            samples.append(elapsed)
    return {
        "median_seconds": statistics.median(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "samples_seconds": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sizes-mib", type=float, nargs="+", default=[0.0625, 1, 16.40625, 36, 64]
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 1 or any(size <= 0 for size in args.sizes_mib):
        parser.error("sizes, repeats and warmup must be positive")
    if args.output.exists():
        parser.error("refusing to overwrite an existing transport result")
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    dist.init_process_group("nccl", timeout=timedelta(seconds=90), device_id=device)
    if dist.get_world_size() != 2:
        raise ValueError("transport probe needs exactly two workers")
    group = dist.new_group(backend="gloo")
    records = []
    try:
        for size in args.sizes_mib:
            count = int(size * (1 << 20) // 2)
            tensor = torch.zeros(count, dtype=torch.bfloat16, device=device)
            buffer = torch.empty_like(tensor)
            host = torch.zeros(count, dtype=torch.bfloat16, pin_memory=True)
            operations = {
                "all_reduce": partial(dist.all_reduce, tensor),
                "pp_roundtrip": partial(exchange, tensor, buffer, rank),
                "h2d_concurrent_ranks": partial(tensor.copy_, host, non_blocking=True),
                "d2h_concurrent_ranks": partial(host.copy_, tensor, non_blocking=True),
            }
            for name, operation in operations.items():
                record = measure(operation, device, args.warmup, args.repeats)
                record.update({"operation": name, "tensor_bytes": count * 2})
                records.append(record)
                if rank == 0:
                    print(
                        f"{name} {size:g} MiB: {record['median_seconds'] * 1000:.3f} ms",
                        flush=True,
                    )
            del tensor, buffer, host, operations
        result = {
            "hardware": hardware_record(rank, int(device.index), device),
            "records": records,
        }
        ranks = gather_objects(result, rank=rank, group=group)
        if rank == 0:
            topology = subprocess.run(
                ["nvidia-smi", "topo", "-m"],
                text=True,
                capture_output=True,
                check=False,
            )
            write_json(
                args.output,
                {
                    "ranks": ranks,
                    "topology": topology.stdout,
                    "torch_version": torch.__version__,
                    "nccl_env": {
                        key: value
                        for key, value in os.environ.items()
                        if key.startswith("NCCL_")
                    },
                    "notes": "CUDA-synchronized wall time. PP roundtrip is two serialized tensor transfers. Host transfers run concurrently on both ranks; no link speed emulation.",
                },
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
