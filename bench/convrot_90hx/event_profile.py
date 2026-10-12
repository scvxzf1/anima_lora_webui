"""CUDA Event diagnostics for CMP devices unsupported by CUPTI.

Nested scopes overlap. Never sum them into a promised full-step speedup.
"""

import argparse
from collections import defaultdict
from pathlib import Path
import time
import torch
from .model import build_model, drain
from .inputs import load_input
from .runner import DEFAULT_MODEL, step
from .manifest import ROOT, GPU_UUID, validate_device, software_manifest, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--mode", choices=["w8a16", "w8a8"], required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    device = validate_device(GPU_UUID)
    model, network, initial, metadata = build_model(
        Path(DEFAULT_MODEL), args.mode, 12, 0
    )
    tensors, inputs = load_input(
        ROOT / "output/ab-anima-20261002/cache",
        ROOT / "output/ab-anima-20261002/images-1024",
        0,
        0,
    )
    batch = {k: v.to("cuda") for k, v in tensors.items()}
    optimizer = torch.optim.AdamW(
        [p for p in network.parameters() if p.requires_grad], lr=2e-5, weight_decay=0.01
    )
    for _ in range(3):
        step(model, network, optimizer, batch, 12, diagnostics=False)
    drain(model)
    from library.runtime.convrot import fused

    records = defaultdict(list)

    def wrap(name, fn):
        def timed(*a, **kw):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            cpu = time.perf_counter()
            start.record()
            result = fn(*a, **kw)
            end.record()
            records[name].append((start, end, time.perf_counter() - cpu))
            return result

        return timed

    originals = {}
    for name in ["group_rht", "quantize_activation_absmax_int8", "int8_mm_scaled"]:
        if hasattr(fused, name):
            originals[name] = getattr(fused, name)
            setattr(fused, name, wrap(name, originals[name]))
    cls = fused._FusedW8A8Fn if args.mode == "w8a8" else fused._FusedW8A16Fn
    backward = cls.backward
    cls.backward = staticmethod(wrap("full_dX_including_rht", backward))
    try:
        started = time.perf_counter()
        for _ in range(3):
            step(model, network, optimizer, batch, 12, diagnostics=False)
        drain(model)
        elapsed = time.perf_counter() - started
        summary = {
            name: {
                "calls": len(items),
                "event_total_ms": sum(a.elapsed_time(b) for a, b, _ in items),
                "cpu_submission_total_ms": sum(c for _, _, c in items) * 1000,
            }
            for name, items in records.items()
        }
        write_json(
            args.out,
            {
                "status": "PASS",
                "device": device,
                "manifest": software_manifest(),
                "model": metadata,
                "inputs": inputs,
                "steps": 3,
                "wall_seconds": elapsed,
                "timing_valid_for_ranking": False,
                "scope": "CUDA Events on current stream; nested RHT/dX scopes overlap; CPU submission includes enqueue/waits; CUPTI unsupported on CMP90HX",
                "components": summary,
            },
        )
        print(summary, flush=True)
    finally:
        cls.backward = staticmethod(backward)
        for name, fn in originals.items():
            setattr(fused, name, fn)


if __name__ == "__main__":
    main()
