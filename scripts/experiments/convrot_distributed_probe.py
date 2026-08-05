#!/usr/bin/env python
"""Multi-process distributed hot test: ConvRot W8A16/W8A8 × Accelerate data-parallel.

Validates the frozen quantized-base × manual adapter gradient-sync interaction
that the single-GPU probes cannot reach. Launch with torchrun (one process per
GPU)::

    .venv/bin/torchrun --standalone --nproc_per_node=2 \\
        scripts/experiments/convrot_distributed_probe.py \\
        --mode w8a16 --steps 4 --rank-dim 32 --scope all \\
        --json-out output/tests/convrot_distributed_w8a16.json

Checks (mirrors the ConvRot × distributed audit):
1. end-to-end runs on 2 ranks with no meta/NCCL error;
2. quantized payloads (``w_q``/``w_scale``) land on each rank's *own* device;
3. after ``synchronize_optimizer_gradients`` the adapter params agree bitwise
   across ranks (mean-reduced grads + identical AdamW step);
4. math stays close to a single-process reference (no sync) when every rank is
   fed the *same* cached sample (equivalent to global batch = world × repeat);
5. W8A8 exercises ``torch._int_mm`` on a non-default device (rank1 → cuda:1).

Reports the four required axes per run: math offset, speed (sec/step), VRAM
(peak GB per rank), and the optimization/notes column. Rank 0 writes JSON.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch.nn import functional as F

from library.runtime.convrot.apply import apply_convrot_to_lora_network
from library.runtime.block_swap_payload import block_swap_payload_residency
from library.training.gradient_sync import (
    prepare_network_for_manual_gradient_sync,
    synchronize_optimizer_gradients,
)
from scripts.experiments.convrot_short_train_probe import (
    _create_lora,
    _load_batches,
)
from scripts.experiments.int8_linear_equivalence_probe import (
    DEFAULT_DATA_DIR,
    DEFAULT_DIT_PATH,
)


def _cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def _payload_on_local_device(residency: dict[str, Any], local_index: int) -> bool:
    """True when every resident payload byte sits on this rank's cuda device."""
    by_dev = residency.get("bytes_by_device") or {}
    if not by_dev:
        return True  # no managed payload (e.g. bf16) — nothing to misplace
    expected = f"cuda:{local_index}"
    return set(by_dev.keys()) == {expected}


def run_distributed(
    *,
    mode: str | None,
    steps: int,
    rank_dim: int,
    alpha: float,
    scope: str,
    group_size: int,
    lr: float,
    seed: int,
    batch_size: int,
    sync_each_step: bool,
    dit_path: Path,
    data_dir: Path,
    attn_mode: str,
) -> dict[str, Any]:
    from accelerate import Accelerator
    from library.anima.weights import load_anima_model

    accelerator = Accelerator(cpu=False, mixed_precision="no")
    local_index = int(os.environ.get("LOCAL_RANK", accelerator.local_process_index))
    device = torch.device(f"cuda:{local_index}")
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    world = int(accelerator.num_processes)
    rank = int(accelerator.process_index)

    # Same cached batches on every rank (feed-rank0 mode ⇒ deterministic, and
    # equivalent to a global batch that repeats one sample world times).
    batches = _load_batches(
        data_dir,
        n_batches=max(steps, 4),
        batch_size=batch_size,
        seed=seed,
        device=torch.device("cpu"),
        dtype=dtype,
    )

    torch.zeros(1, device=device)
    torch.cuda.reset_peak_memory_stats(device)

    anima = load_anima_model(
        device=device,
        dit_path=str(dit_path),
        attn_mode=attn_mode,
        loading_device=device,
        dit_weight_dtype=dtype,
    )
    anima.to(device=device, dtype=dtype).requires_grad_(False)
    anima.reset_mod_guidance()
    anima.enable_gradient_checkpointing()

    network = _create_lora(
        anima, seed=seed + 101, device=device, dtype=dtype,
        rank=rank_dim, alpha=alpha, scope=scope,
    )
    patched = 0
    if mode is not None:
        result = apply_convrot_to_lora_network(
            network, mode=mode, scope=scope, group_size=group_size, unet=anima,
        )
        patched = result.patched_count

    params = [p for p in network.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.01)

    # Real distributed wiring: places network, broadcasts params/buffers from
    # rank0 once, and (async) registers gradient hooks. Identical to train loop.
    network = prepare_network_for_manual_gradient_sync(accelerator, network, optim)

    residency = block_swap_payload_residency(anima)
    payload_ok = _payload_on_local_device(residency, local_index)

    losses: list[float] = []
    t0 = time.time()
    for step in range(steps):
        model_inputs_cpu, target_cpu, _meta = batches[step % len(batches)]
        x = model_inputs_cpu[0].to(device=device, dtype=dtype)
        timesteps = model_inputs_cpu[1].to(device=device, dtype=dtype)
        context = model_inputs_cpu[2].to(device=device, dtype=dtype)
        padding_mask = model_inputs_cpu[3].to(device=device, dtype=dtype)
        target = target_cpu.to(device=device, dtype=dtype)

        optim.zero_grad(set_to_none=True)
        out = anima(x, timesteps, context, padding_mask=padding_mask)
        loss = F.mse_loss(out.float(), target.float())
        loss.backward()
        if sync_each_step and world > 1:
            synchronize_optimizer_gradients(accelerator, optim)
        optim.step()
        losses.append(float(loss.detach().item()))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.time() - t0
    peak = int(torch.cuda.max_memory_allocated(device))

    # Cross-rank param agreement: gather max |p_rank - p_ref| over all params.
    # After identical (mean-reduced) grads + identical AdamW the params should
    # match bitwise (max diff == 0); any nonzero diff flags a sync bug.
    param_max_abs = 0.0
    cross_rank_max_diff = 0.0
    if world > 1:
        with torch.no_grad():
            for p in params:
                param_max_abs = max(param_max_abs, float(p.detach().abs().max()))
                ref = p.detach().clone()
                dist.broadcast(ref, src=0)
                cross_rank_max_diff = max(
                    cross_rank_max_diff, float((p.detach() - ref).abs().max())
                )

    return {
        "mode": mode or "bf16",
        "world_size": world,
        "rank": rank,
        "local_index": local_index,
        "device": str(device),
        "patched": patched,
        "steps": steps,
        "losses": losses,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "sec_per_step": elapsed / max(steps, 1),
        "elapsed_sec": elapsed,
        "peak_gb": peak / (1024**3),
        "payload_residency": residency,
        "payload_on_local_device": payload_ok,
        "cross_rank_param_max_diff": cross_rank_max_diff,
        "param_max_abs": param_max_abs,
        "sync_each_step": sync_each_step,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["bf16", "w8a16", "w8a8"], default="w8a16")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scope", default="all")
    parser.add_argument("--group-size", type=int, default=256)
    parser.add_argument("--rank-dim", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--attn-mode", default="torch")
    parser.add_argument("--dit-path", type=Path, default=DEFAULT_DIT_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--sync-each-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mean-reduce adapter grads each step (distributed semantics).",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    mode = None if args.mode == "bf16" else args.mode
    result = run_distributed(
        mode=mode,
        steps=args.steps,
        rank_dim=args.rank_dim,
        alpha=args.alpha,
        scope=args.scope,
        group_size=args.group_size,
        lr=args.lr,
        seed=args.seed,
        batch_size=args.batch_size,
        sync_each_step=args.sync_each_step,
        dit_path=args.dit_path,
        data_dir=args.data_dir,
        attn_mode=args.attn_mode,
    )

    rank = result["rank"]
    world = result["world_size"]
    print(
        f"[rank{rank}/{world}] mode={result['mode']} "
        f"loss_first={result['loss_first']:.6f} loss_last={result['loss_last']:.6f} "
        f"sec/step={result['sec_per_step']:.3f} peak={result['peak_gb']:.2f}GB "
        f"payload_ok={result['payload_on_local_device']} "
        f"cross_rank_diff={result['cross_rank_param_max_diff']:.3e}",
        flush=True,
    )

    # Emit one JSON per rank so the driver can compare rank0 vs rank1.
    if args.json_out is not None:
        out = Path(str(args.json_out))
        out.parent.mkdir(parents=True, exist_ok=True)
        per_rank = out.with_name(out.stem + f"_rank{rank}" + out.suffix)
        per_rank.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[rank{rank}] wrote {per_rank}", flush=True)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
