"""Real CUDA allocator OOM and fresh-process recovery on a bounded toy workload."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from library.training.adaptive_runtime.process import IsolatedRunner
from library.training.adaptive_runtime.retry import MemoryPlan, RetryLimits, run_recovery
from library.training.auto_block_swap.process import write_result


def worker(request: Path):
    import torch

    payload = json.loads(request.read_text())
    plan = MemoryPlan(**payload["plan"])
    properties = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(128 * 1024**2 / properties.total_memory)
    torch.manual_seed(20260921)
    result = {"status": "error", "pid": os.getpid(), "gpu": properties.name,
              "gpu_uuid": str(properties.uuid), "torch": torch.__version__,
              "allocator_limit_bytes": 128 * 1024**2, "optimizer_started": False,
              "stage": "model_load", "updates": [],
              "effective_batch": plan.micro_batch * plan.accumulation}
    try:
        model = torch.nn.Linear(1024, 128).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for step in range(3):
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for _ in range(plan.accumulation):
                result["stage"] = "forward"
                x = torch.randn(plan.micro_batch, 4096, 1024, device="cuda")
                loss = model(x).square().mean() / plan.accumulation
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss")
                result["stage"] = "backward"
                loss.backward()
                loss_sum += float(loss.detach())
                del loss, x
            if not all(torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("Nonfinite gradient")
            result["stage"] = "optimizer"
            result["optimizer_started"] = True
            optimizer.step()
            torch.cuda.synchronize()
            result["updates"].append({"step": step + 1, "loss": loss_sum,
                                      "seconds": time.perf_counter() - started})
        result["status"] = "ok"
    except torch.cuda.OutOfMemoryError as exc:
        result.update(status="cuda_oom", error=str(exc))
    except Exception as exc:
        result.update(status="error", error_type=type(exc).__name__, error=str(exc))
    finally:
        result["peak_allocated"] = torch.cuda.max_memory_allocated()
        write_result(request.parent / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if args.output is None:
        parser.error("--output is required")
    args.output.mkdir(parents=True, exist_ok=False)
    runner = IsolatedRunner(args.output, lambda request: [
        sys.executable, "-m", "bench.adaptive_runtime.probe_oom", "--worker", str(request),
    ], timeout=120)
    def record(report):
        write_result(args.output / "summary.json", {
            "scope": "real_cuda_oom_toy_training_not_dit", **report,
        })
    report = run_recovery(
        MemoryPlan(gradient_checkpointing=True, micro_batch=8),
        RetryLimits(max_blocks=0, max_attempts=4, allow_batch_change=True),
        runner, record=record,
    )
    if report["status"] != "ok" or len(report["attempts"]) < 2:
        raise RuntimeError("Expected a real OOM followed by successful recovery")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
