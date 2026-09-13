"""Diagnose order-dependent slab equality without relaxing numeric assertions."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from library.runtime.offloading import ModelOffloader
from library.training.auto_block_swap.process import write_result
from tests.test_block_swapping import (
    _TinyBlock,
    test_bf16_slab_forward_only_wraparound_matches_baseline,
)
from tests.test_dynamic_block_swap_runtime import (
    test_nf4_reconfigure_preserves_cpu_masters_and_forward,
)


def trace_layers():
    torch.manual_seed(20260828)
    baseline = nn.ModuleList([_TinyBlock().to("cuda", torch.bfloat16) for _ in range(30)])
    swapped = nn.ModuleList([_TinyBlock().to("cuda", torch.bfloat16) for _ in range(30)])
    swapped.load_state_dict(baseline.state_dict())
    offloader = ModelOffloader(
        swapped, 26, torch.device("cuda"), supports_backward=False,
        transfer_dtype="bf16", restore_mode="slab",
    )
    x = torch.randn(1, 2, device="cuda", dtype=torch.bfloat16)
    expected = []
    actual = []
    try:
        offloader.prepare_block_devices_before_forward(swapped, free_cache=False)
        hidden = x.clone()
        for block in baseline:
            hidden = block(hidden)
            expected.append(hidden.detach().clone())
        for cycle in range(3):
            hidden = x.clone()
            for index, block in enumerate(swapped):
                offloader.wait_for_block(index)
                weight = block.base.weight.detach().clone()
                hidden = block(hidden)
                actual.append((cycle, index, weight, hidden.detach().clone()))
                offloader.submit_move_blocks(swapped, index)
        offloader.set_forward_only(True)
        torch.cuda.synchronize()
        differences = []
        for cycle, index, weight, output in actual:
            if not torch.equal(output, expected[index]):
                differences.append({
                    "cycle": cycle, "layer": index,
                    "weights_equal": torch.equal(weight, baseline[index].base.weight),
                    "max_abs": float((output - expected[index]).abs().max()),
                    "expected": expected[index].tolist(), "actual": output.tolist(),
                })
        return differences
    finally:
        offloader.set_forward_only(True)
        offloader.thread_pool.shutdown(wait=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-compiled", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warm_compiled:
        for device in ("cuda", "cuda:0"):
            for compiled in (False, True):
                test_nf4_reconfigure_preserves_cpu_masters_and_forward(compiled, device)
    results = []
    for repeat in range(3):
        try:
            test_bf16_slab_forward_only_wraparound_matches_baseline()
            results.append({"repeat": repeat, "exact": True})
        except AssertionError as exc:
            results.append({"repeat": repeat, "failure": str(exc)})
    differences = trace_layers()
    write_result(args.output, {"repeats": results, "layer_differences": differences})
    print(json.dumps({"repeats": results, "layer_differences": len(differences)}), flush=True)


if __name__ == "__main__":
    main()
