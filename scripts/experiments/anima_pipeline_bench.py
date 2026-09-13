"""Fixed-effective-batch Anima PP experiment; production PP remains disabled."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics
import subprocess

import torch
import torch.distributed as dist

from scripts.experiments.anima_parallel.benchmark import (
    Experiment,
    pack_batches,
    timed_step,
)
from scripts.experiments.anima_parallel.checkpoint import (
    experiment_contract,
    load_distributed_checkpoint,
    save_stage_checkpoint,
)
from scripts.experiments.anima_parallel.common import (
    DEFAULT_DIT,
    DEFAULT_LATENT,
    DEFAULT_TEXT,
    gather_objects,
    hardware_record,
    load_cached_batches,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single", "pp", "tp"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dit-path", type=Path, default=DEFAULT_DIT)
    parser.add_argument("--latent-path", type=Path, default=DEFAULT_LATENT)
    parser.add_argument("--text-path", type=Path, default=DEFAULT_TEXT)
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument("--split", type=int, default=20)
    parser.add_argument("--schedule", choices=("1f1b", "gpipe"), default="1f1b")
    parser.add_argument(
        "--checkpoint", choices=("full", "off", "every_other"), default="full"
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=114)
    parser.add_argument("--rank-dim", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--attn-mode", choices=("flash", "torch"), default="flash")
    parser.add_argument("--verify-resume", action="store_true")
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--deterministic-attention", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--capture-initial", action="store_true")
    args = parser.parse_args()
    args.runtime_version = 2
    if args.steps < 1 or args.warmup_steps < 1 or args.microbatches < 1:
        parser.error("steps, warmup-steps and microbatches must be positive")
    if args.resume_dir is not None and args.mode != "pp":
        parser.error("--resume-dir is currently supported only for PP")
    return args


def hardware_snapshot():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu,power.draw,pcie.link.gen.current,pcie.link.width.current",
                "--format=csv",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if result.returncode:
            return f"unavailable: exit {result.returncode}: {result.stderr.strip()}"
        return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"unavailable: {type(error).__name__}: {error}"


def export_adapter(path, state, args, completed_steps):
    from networks.lora_save import save_network_weights

    save_network_weights(
        state,
        file=str(path),
        dtype=torch.bfloat16,
        save_variant="standard",
        metadata={
            "ss_network_module": "networks.lora_anima",
            "ss_network_dim": str(args.rank_dim),
            "ss_network_alpha": str(args.alpha),
            "ss_steps": str(completed_steps),
            "ss_output_name": path.stem,
            "ss_model_family": "anima",
            "ss_use_moe_style": "False",
            "ss_route_per_layer": "False",
            "ss_router_source": "none",
            "ss_parallel_probe_mode": args.mode,
        },
    )


def run(args, rank, device, group):
    distributed = args.mode != "single"
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    if distributed:
        dist.barrier()
    initial_hardware = hardware_snapshot()
    examples = load_cached_batches(
        args.latent_path,
        args.text_path,
        count=1,
        seed=args.seed,
        dtype=torch.bfloat16,
    )
    experiment = Experiment(args, rank, device, examples[0])
    contract = experiment_contract(args)
    write_json(args.output_dir / f"contract-rank{rank}.json", contract)
    start_step = (
        load_distributed_checkpoint(
            args.resume_dir,
            experiment.module,
            experiment.optimizer,
            contract=contract,
            group=group,
        )
        if args.resume_dir is not None
        else 0
    )
    batches = load_cached_batches(
        args.latent_path,
        args.text_path,
        count=(args.steps + args.warmup_steps) * args.microbatches,
        seed=args.seed + start_step * args.microbatches * 17,
        dtype=torch.bfloat16,
    )
    inputs, target = pack_batches(batches[: args.microbatches], device)
    resume = (
        experiment.verify_resume(args.output_dir, inputs, target, contract)
        if args.verify_resume
        else None
    )
    records, communication, peak_allocated, peak_reserved = run_steps(
        args, experiment, batches
    )
    local = {
        "hardware": hardware_record(rank, int(device.index), device),
        "initial_hardware": initial_hardware,
        "final_hardware": hardware_snapshot(),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "registered_parameter_bytes": sum(
            p.numel() * p.element_size() for p in experiment.module.parameters()
        ),
        "communication_measured_steps": communication,
        "resume_check": resume,
        "steps": records,
        "latent_shape": list(examples[0][0][0].shape),
        "context_shape": list(examples[0][0][2].shape),
    }
    completed_steps = start_step + args.steps
    if args.mode == "pp":
        save_stage_checkpoint(
            args.output_dir / f"stage-rank{rank}.pt",
            experiment.module,
            experiment.optimizer,
            step=completed_steps,
            contract=contract,
        )
        local["topology"] = experiment.module.topology()
    states = (
        gather_objects(experiment.adapter_state(), rank=rank, group=group)
        if distributed
        else [experiment.adapter_state()]
    )
    ranks = gather_objects(local, rank=rank, group=group) if distributed else [local]
    if rank == 0:
        write_result(
            args, experiment, states, ranks, records, start_step, completed_steps
        )


def run_steps(args, experiment, batches):
    rank, device = experiment.rank, experiment.device
    distributed = args.mode != "single"
    records = []
    communication_before = {}
    profiler = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        )
        if args.profile
        else nullcontext()
    )
    with profiler as profile:
        for step in range(args.warmup_steps + args.steps):
            offset = step * args.microbatches
            inputs, target = pack_batches(
                batches[offset : offset + args.microbatches], device
            )
            if step == args.warmup_steps:
                torch.cuda.reset_peak_memory_stats(device)
                communication_before = experiment.communication()
            record = timed_step(
                experiment,
                inputs,
                target,
                distributed=distributed,
                update=step >= args.warmup_steps,
                capture=args.capture_initial and step == 0,
            )
            if args.capture_initial and step == 0:
                payload = {
                    "output": getattr(experiment, "captured_output", None),
                    "gradients": {key: parameter.grad.detach().cpu() for key, parameter in
                                  experiment.adapters.named_parameters() if parameter.grad is not None},
                }
                if payload["output"] is not None:
                    payload["output"] = payload["output"].cpu()
                    del experiment.captured_output
                torch.save(payload, args.output_dir / f"initial-rank{rank}.pt")
            if step >= args.warmup_steps:
                records.append(record)
            if rank == 0:
                print(
                    json.dumps({"step": step - args.warmup_steps + 1, **record}),
                    flush=True,
                )
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    if args.profile:
        profile.export_chrome_trace(str(args.output_dir / f"trace-rank{rank}.json"))
    communication = {
        key: value - communication_before.get(key, 0)
        for key, value in experiment.communication().items()
    }
    return records, communication, peak_allocated, peak_reserved


def write_result(args, experiment, states, ranks, records, start_step, completed_steps):
    if args.mode == "tp":
        from scripts.experiments.anima_parallel.tensor_parallel import (
            consolidate_tp_state,
        )

        merged = consolidate_tp_state(states, experiment.tp_specs)
    else:
        merged = {}
        for state in states:
            if set(merged).intersection(state):
                raise RuntimeError("overlapping PP adapter ownership")
            merged.update(state)
    export_adapter(
        args.output_dir / "adapter.safetensors", merged, args, completed_steps
    )
    seconds = statistics.mean(record["seconds"] for record in records)
    result = {
        "schema_version": 1,
        "mode": args.mode,
        "schedule": "fill_drain" if args.microbatches == 1 else args.schedule,
        "microbatch_size": 1,
        "effective_batch": args.microbatches,
        "start_step": start_step,
        "completed_steps": completed_steps,
        "seconds_per_optimizer_step": seconds,
        "samples_per_second": args.microbatches / seconds,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "torch_version": torch.__version__,
        "runtime_version": args.runtime_version,
        "ranks": ranks,
        "measurement": "CUDA-synchronized compute/update time; max across ranks; batch transfer, reporting collectives and checkpoint IO excluded",
    }
    write_json(args.output_dir / "result.json", result)
    print(
        json.dumps(
            {
                "result": str(args.output_dir / "result.json"),
                "seconds": seconds,
                "samples_per_second": args.microbatches / seconds,
            }
        ),
        flush=True,
    )


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    if args.deterministic_attention:
        from functools import partial
        from networks import attention_dispatch

        if args.attn_mode == "flash":
            attention_dispatch.flash_attn_func = partial(
                attention_dispatch.flash_attn_func, deterministic=True
            )
            attention_dispatch.flash_attn_varlen_func = partial(
                attention_dispatch.flash_attn_varlen_func, deterministic=True
            )
    group = None
    if args.mode != "single":
        dist.init_process_group(
            "nccl", timeout=timedelta(seconds=120), device_id=device
        )
        if dist.get_world_size() != 2:
            raise ValueError("distributed probe requires exactly two workers")
        group = dist.new_group(backend="gloo")
    elif int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("single mode requires exactly one process")
    try:
        run(args, rank, device, group)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
