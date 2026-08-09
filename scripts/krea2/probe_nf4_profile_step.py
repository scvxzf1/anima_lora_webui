#!/usr/bin/env python3
"""Profile one eager and one compiled Krea-2 NF4 training step on PG199."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("K2_PROFILE_GPU", "1"))

import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_nf4_ablation import LR, make_network, load_dit  # noqa: E402
from probe_nf4_compile_buckets import _make_inputs, _step  # noqa: E402


def _profile_step(dit, network, optimizer, inputs, device, dtype) -> dict:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        metric = _step(dit, network, optimizer, inputs, device, dtype)
    events = []
    for event in prof.key_averages():
        self_cuda_us = float(getattr(event, "self_device_time_total", 0.0) or 0.0)
        if self_cuda_us <= 0:
            continue
        events.append(
            {
                "name": event.key,
                "self_cuda_ms": self_cuda_us / 1000.0,
                "cuda_ms": float(
                    getattr(event, "device_time_total", 0.0) or 0.0
                ) / 1000.0,
                "count": int(event.count),
            }
        )
    events.sort(key=lambda item: item["self_cuda_ms"], reverse=True)
    return {"metric": metric, "top_cuda": events[:25]}


def main() -> int:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    nf4_path = os.environ.get(
        "K2_PROFILE_NF4_PATH",
        str(ROOT / "models" / "diffusion_models" / "krea2_raw_nf4.safetensors"),
    )
    out_path = Path(
        os.environ.get("K2_PROFILE_OUT", "/tmp/krea2_nf4_profile_step.json")
    )

    dit, source = load_dit(True, nf4_path, torch.device("cpu"), dtype)
    for parameter in dit.parameters():
        parameter.requires_grad_(False)
    network = make_network(dit)
    dit = dit.to(device)
    network = network.to(device).to(dtype)
    dit.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(network.parameters(), lr=LR, weight_decay=0.0)
    inputs = _make_inputs(1008, 1024, device, dtype)

    _step(dit, network, optimizer, inputs, device, dtype)
    eager = _profile_step(dit, network, optimizer, inputs, device, dtype)

    dit.compile_blocks(backend="inductor", compile_block_scope="resident")
    _step(dit, network, optimizer, inputs, device, dtype)
    compiled = _profile_step(dit, network, optimizer, inputs, device, dtype)

    result = {
        "gpu": torch.cuda.get_device_name(),
        "nf4_source": source,
        "shape": [1008, 1024],
        "eager": eager,
        "compiled": compiled,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
