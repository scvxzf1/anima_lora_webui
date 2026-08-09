#!/usr/bin/env python3
"""Krea-2 NF4 compile multi-bucket specialization probe.

Loads the DiT once, then alternates representatives of the 4032 and 4200
image-token families. Synthetic latents/text embeddings isolate DiT compile
and training behavior from TE/VAE loading.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("K2_BUCKET_GPU", "1"))

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from library.models.krea2_raw.family import Krea2TextEmbedding, forward_for_loss  # noqa: E402
from probe_nf4_ablation import LR, make_network, load_dit  # noqa: E402
from probe_train import FIXED_SIGMA  # noqa: E402

CASES = (
    ("tokens4032_a", "tokens4032", (1008, 1024)),
    ("tokens4200_a", "tokens4200", (960, 1120)),
    ("tokens4032_b", "tokens4032", (896, 1152)),
    ("tokens4200_b", "tokens4200", (1120, 960)),
)


def _make_inputs(
    width: int, height: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, Krea2TextEmbedding, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(width * 10_000 + height)
    latent = torch.randn(
        1, 16, 1, height // 8, width // 8,
        device=device, dtype=dtype, generator=generator,
    )
    noise = torch.randn(latent.shape, device=device, dtype=dtype, generator=generator)
    hiddens = torch.randn(
        1, 512, 12, 2560, device=device, dtype=dtype, generator=generator
    )
    mask = torch.zeros(1, 512, device=device, dtype=torch.bool)
    mask[:, :32] = True
    return latent, Krea2TextEmbedding(hiddens, mask), noise, noise - latent


def _step(
    dit,
    network,
    optimizer,
    inputs: tuple[torch.Tensor, Krea2TextEmbedding, torch.Tensor, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    latent, text_emb, noise, target = inputs
    sigma = torch.full((1,), FIXED_SIGMA, device=device, dtype=dtype)
    noisy = (1.0 - sigma) * latent + sigma * noise
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    velocity = forward_for_loss(dit, noisy, text_emb, sigma)
    loss = torch.nn.functional.mse_loss(velocity, target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    return {
        "step_s": time.perf_counter() - started,
        "loss": float(loss.detach()),
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def main() -> int:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    nf4_path = os.environ.get(
        "K2_BUCKET_NF4_PATH",
        str(ROOT / "models" / "diffusion_models" / "krea2_raw_nf4.safetensors"),
    )
    out_path = Path(
        os.environ.get("K2_BUCKET_OUT", "/tmp/krea2_nf4_compile_buckets.json")
    )

    dit, source = load_dit(True, nf4_path, torch.device("cpu"), dtype)
    for parameter in dit.parameters():
        parameter.requires_grad_(False)
    network = make_network(dit)
    dit = dit.to(device)
    network = network.to(device).to(dtype)
    dit.enable_gradient_checkpointing()
    dit.compile_blocks(backend="inductor", compile_block_scope="resident")
    optimizer = torch.optim.AdamW(network.parameters(), lr=LR, weight_decay=0.0)

    inputs = {
        label: _make_inputs(width, height, device, dtype)
        for label, _family, (width, height) in CASES
    }
    family_by_label = {label: family for label, family, _size in CASES}
    sequence = (
        "tokens4032_a",
        "tokens4200_a",
        "tokens4032_b",
        "tokens4200_b",
        "tokens4032_a",
        "tokens4200_a",
    )
    records = []
    for visit, label in enumerate(sequence):
        metric = _step(dit, network, optimizer, inputs[label], device, dtype)
        metric.update(
            {"visit": visit, "case": label, "family": family_by_label[label]}
        )
        records.append(metric)
        print(
            f"visit={visit} case={label} step={metric['step_s']:.3f}s "
            f"peak={metric['peak_gb']:.2f}GB loss={metric['loss']:.6f}",
            flush=True,
        )

    result = {
        "gpu": torch.cuda.get_device_name(),
        "nf4_source": source,
        "sequence": list(sequence),
        "cases": {
            label: {"family": family, "size": list(size)}
            for label, family, size in CASES
        },
        "records": records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
