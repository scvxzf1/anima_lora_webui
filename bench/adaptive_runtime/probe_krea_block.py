"""Real-weight, synthetic-activation Krea block calibration (not full training)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from safetensors import safe_open

from library.models.krea2_raw.dit import PositionalEncoding, SingleStreamBlock
from library.training.adaptive_runtime.precision import calibrate_unit, preferred_candidate
from library.training.auto_block_swap.process import write_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=128)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(20260921)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prefix = f"blocks.{args.block}."
    with torch.device("meta"):
        block = SingleStreamBlock(6144, 48, 4, kvheads=12)
    with safe_open(args.weights, framework="pt", device="cpu") as file:
        weights = {key.removeprefix(prefix): file.get_tensor(key)
                   for key in file.keys() if key.startswith(prefix)}
    block.load_state_dict(weights, strict=True, assign=True)
    block.requires_grad_(False)
    block.eval()
    del weights
    position = torch.zeros(1, args.tokens, 3)
    position[..., 1] = torch.arange(args.tokens)
    freqs = PositionalEncoding(6144, [32, 48, 48], theta=1000)(position)
    cases = [((torch.randn(1, args.tokens, 6144) * scale,
               torch.randn(1, 1, 36864) * 0.1, freqs), {})
             for scale in (0.02, 1.0, 8.0)]
    properties = torch.cuda.get_device_properties(0)
    report = {
        "scope": "real_single_block_synthetic_activations_not_full_training",
        "torch": torch.__version__, "gpu": properties.name,
        "capability": list(torch.cuda.get_device_capability()),
        "gpu_uuid": str(properties.uuid), "total_memory": properties.total_memory,
        "weights": str(args.weights.resolve()), "weights_bytes": args.weights.stat().st_size,
        "block": args.block, "tokens": args.tokens,
        "tf32": False, "status": "running",
    }
    write_result(args.output, report)
    started = time.perf_counter()
    try:
        report["calibration"] = calibrate_unit(
            block, cases, device="cuda:0",
            candidate=preferred_candidate(torch.cuda.get_device_capability()),
            preserve_input_indices=(2,),
        )
        torch.cuda.synchronize()
        report["status"] = "ok"
    except Exception as exc:
        report.update(status="error", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report.update(elapsed_seconds=time.perf_counter() - started,
                      peak_allocated=torch.cuda.max_memory_allocated(),
                      peak_reserved=torch.cuda.max_memory_reserved())
        write_result(args.output, report)
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
