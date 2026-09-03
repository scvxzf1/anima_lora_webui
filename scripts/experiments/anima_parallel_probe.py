#!/usr/bin/env python
"""Run real two-GPU Anima PP2, TP2, or TP2+INT8 communication probes."""

from __future__ import annotations

import argparse
import gc
import sys
import statistics
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from scripts.experiments.anima_parallel.collectives import (  # noqa: E402
    CommunicationStats,
    configure_collectives,
)
from scripts.experiments.anima_parallel.common import (  # noqa: E402
    DEFAULT_DIT,
    DEFAULT_LATENT,
    DEFAULT_TEXT,
    create_mlp_lora,
    gather_objects,
    hardware_record,
    init_distributed,
    load_cached_batches,
    load_model,
    lora_block_index,
    lora_state_for_blocks,
    save_network_state,
    to_device_batch,
    write_json,
)
from scripts.experiments.anima_parallel.pipeline_parallel import (  # noqa: E402
    PipelineCommunication,
    finish_model,
    prepare_block_inputs,
    run_local_blocks,
)
from scripts.experiments.anima_parallel.tensor_parallel import (  # noqa: E402
    consolidate_tp_state,
    parallelize_anima_blocks,
    synchronize_replicated_lora_gradients,
)


def _local_loras(network, owned: set[int], device: torch.device):
    selected = []
    for lora in network.unet_loras:
        if lora_block_index(lora) in owned:
            lora.to(device=device, dtype=torch.bfloat16)
            selected.extend(parameter for parameter in lora.parameters() if parameter.requires_grad)
    return selected


def _cleanup(device: torch.device) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


def _run_pp(args, rank, device, gloo_group, batches):
    model = load_model(args.dit_path, torch.device("cpu"), args.attn_mode)
    network = create_mlp_lora(model, seed=args.seed + 101, rank_dim=args.rank_dim, alpha=args.alpha)
    midpoint = len(model.blocks) // 2
    owned = set(range(0, midpoint) if rank == 0 else range(midpoint, len(model.blocks)))
    model.blocks = torch.nn.ModuleList([block for index, block in enumerate(model.blocks) if index in owned])
    model.to(device=device, dtype=torch.bfloat16).train()
    model.enable_gradient_checkpointing()
    parameters = _local_loras(network, owned, device)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    communication = PipelineCommunication()
    losses: list[float] = []
    step_seconds: list[float] = []
    initial_output = None
    torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()

    for step in range(args.warmup_steps + args.steps):
        step_started = perf_counter()
        inputs, target, _ = to_device_batch(batches[step], device, torch.bfloat16)
        x, timesteps, context, padding_mask = inputs
        optimizer.zero_grad(set_to_none=True)
        dummy, embedding, params, kwargs = prepare_block_inputs(
            model, x, timesteps, context, padding_mask
        )
        if rank == 0:
            hidden = run_local_blocks(model, dummy, embedding, context, params, kwargs)
            communication.transfer(lambda value: dist.send(value, dst=1), hidden.detach())
            grad = torch.empty_like(hidden)
            communication.transfer(lambda value: dist.recv(value, src=1), grad)
            hidden.backward(grad)
        else:
            boundary = torch.empty_like(dummy)
            communication.transfer(lambda value: dist.recv(value, src=0), boundary)
            boundary.requires_grad_(True)
            hidden = run_local_blocks(model, boundary, embedding, context, params, kwargs)
            output = finish_model(model, hidden, embedding, kwargs["adaln_lora_B_T_3D"])
            loss = F.mse_loss(output.float(), target.float())
            if initial_output is None:
                initial_output = output.detach().cpu()
            loss.backward()
            if boundary.grad is None:
                raise RuntimeError("PP stage boundary did not receive a gradient")
            communication.transfer(lambda value: dist.send(value, dst=0), boundary.grad)
            if step >= args.warmup_steps:
                losses.append(float(loss.detach()))
        if step >= args.warmup_steps:
            optimizer.step()
        else:
            optimizer.zero_grad(set_to_none=True)
        dist.barrier()
        torch.cuda.synchronize(device)
        if step >= args.warmup_steps:
            step_seconds.append(perf_counter() - step_started)

    torch.cuda.synchronize(device)
    elapsed = perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    local_state = lora_state_for_blocks(network, owned)
    states = gather_objects(local_state, rank=rank, group=gloo_group)
    if rank == 0:
        merged = {}
        for state in states or []:
            merged.update(state)
        save_network_state(network, merged, args.output_dir / "checkpoints" / "pp2.safetensors", mode="pp2", steps=args.steps)
    if rank == 1 and initial_output is not None:
        torch.save(initial_output, args.output_dir / "pp2_initial_output.pt")
    local = {
        "hardware": hardware_record(rank, int(device.index), device),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "losses": losses,
        "communication": communication.as_dict(),
    }
    gathered = gather_objects(local, rank=rank, group=gloo_group)
    if rank == 0:
        rank1_losses = (gathered or [{}, {}])[1].get("losses", [])
        return {
            "mode": "pp2",
            "schedule": "single_microbatch_fill_drain",
            "batch_size_per_microbatch": 1,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "elapsed_seconds": elapsed,
            "seconds_per_step": statistics.mean(step_seconds),
            "median_seconds_per_step": statistics.median(step_seconds),
            "step_seconds": step_seconds,
            "loss_first": rank1_losses[0] if rank1_losses else None,
            "loss_last": rank1_losses[-1] if rank1_losses else None,
            "ranks": gathered,
        }
    return None


def _run_tp(args, rank, device, gloo_group, batches):
    mode = "int8" if args.mode == "tp2_int8" else "bf16"
    stats = CommunicationStats()
    configure_collectives(mode, stats)
    model = load_model(args.dit_path, torch.device("cpu"), args.attn_mode)
    network = create_mlp_lora(model, seed=args.seed + 101, rank_dim=args.rank_dim, alpha=args.alpha)
    specs = parallelize_anima_blocks(model, network, rank=rank, world=2)
    model.to(device=device, dtype=torch.bfloat16).train()
    network.to(device=device, dtype=torch.bfloat16).train()
    model.enable_gradient_checkpointing()
    parameters = [parameter for parameter in network.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    losses: list[float] = []
    step_seconds: list[float] = []
    initial_output = None
    torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    for step in range(args.warmup_steps + args.steps):
        step_started = perf_counter()
        inputs, target, _ = to_device_batch(batches[step], device, torch.bfloat16)
        optimizer.zero_grad(set_to_none=True)
        output = model(*inputs[:3], padding_mask=inputs[3])
        loss = F.mse_loss(output.float(), target.float())
        if initial_output is None:
            initial_output = output.detach().cpu()
        loss.backward()
        synchronize_replicated_lora_gradients(network, specs)
        if step >= args.warmup_steps:
            optimizer.step()
            losses.append(float(loss.detach()))
        dist.barrier()
        torch.cuda.synchronize(device)
        if step >= args.warmup_steps:
            step_seconds.append(perf_counter() - step_started)
    stats.finalize(device)
    torch.cuda.synchronize(device)
    elapsed = perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    torch.save(initial_output, args.output_dir / f"{args.mode}_initial_output_rank{rank}.pt")
    local_state = {key: value.detach().cpu().contiguous() for key, value in network.state_dict().items()}
    states = gather_objects(local_state, rank=rank, group=gloo_group)
    if rank == 0:
        merged = consolidate_tp_state(states or [], specs)
        save_network_state(
            network,
            merged,
            args.output_dir / "checkpoints" / f"{args.mode}.safetensors",
            mode=args.mode,
            steps=args.steps,
        )
    local = {
        "hardware": hardware_record(rank, int(device.index), device),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "losses": losses,
        "communication": stats.as_dict(),
    }
    gathered = gather_objects(local, rank=rank, group=gloo_group)
    if rank == 0:
        return {
            "mode": args.mode,
            "transport": mode,
            "batch_size": 1,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "elapsed_seconds": elapsed,
            "seconds_per_step": statistics.mean(step_seconds),
            "median_seconds_per_step": statistics.median(step_seconds),
            "step_seconds": step_seconds,
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "ranks": gathered,
        }
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pp2", "tp2", "tp2_int8"), required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--rank-dim", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--attn-mode", choices=("flash", "torch"), default="flash")
    parser.add_argument("--dit-path", type=Path, default=DEFAULT_DIT)
    parser.add_argument("--latent-path", type=Path, default=DEFAULT_LATENT)
    parser.add_argument("--text-path", type=Path, default=DEFAULT_TEXT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rank, local_rank, _world, device, gloo_group = init_distributed()
    batches = load_cached_batches(
        args.latent_path,
        args.text_path,
        count=args.steps + args.warmup_steps,
        seed=args.seed,
        dtype=torch.bfloat16,
    )
    try:
        result = (
            _run_pp(args, rank, device, gloo_group, batches)
            if args.mode == "pp2"
            else _run_tp(args, rank, device, gloo_group, batches)
        )
        if rank == 0 and result is not None:
            write_json(args.output_dir / f"{args.mode}.json", result)
            print(f"[{args.mode}] {result['seconds_per_step']:.3f} sec/step", flush=True)
    finally:
        _cleanup(device)
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
