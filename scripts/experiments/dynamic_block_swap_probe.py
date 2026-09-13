"""Bounded formal training acceptance for live AUTO, using an existing fixture."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import time

from library.training.auto_block_swap.process import write_result


def pressure_worker(connection):
    import torch

    torch.cuda.init()
    warmup = torch.zeros(1, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    del warmup
    torch.cuda.empty_cache()
    held = None
    connection.send({"ready": True})
    try:
        while True:
            size = connection.recv()
            if size is None:
                break
            held = None
            torch.cuda.empty_cache()
            if size:
                held = torch.empty(size, dtype=torch.uint8, device="cuda")
                held.zero_()
                torch.cuda.synchronize()
            connection.send({"held_bytes": size})
    finally:
        del held
        connection.close()


class PressureSchedule:
    def __init__(self, connection, output):
        self.connection = connection
        self.output = output
        self.started = None
        self.released = False

    def before(self, state):
        import torch

        controller = getattr(state, "auto_swap_controller", None)
        if controller is None or controller.microsteps:
            return
        step = state.global_step
        size = None
        if self.started is None and controller.policy.current < controller.maximum:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info()
            budget = free + torch.cuda.memory_reserved()
            safe_peak = controller.overhead + controller._resident_bytes(
                min(controller.maximum, controller.policy.current + 1)
            )
            size = int(
                budget
                - safe_peak
                - controller.reserve
                - min(controller.block_bytes) / 4
            )
            if not 0 < size < free - 512 * 1024**2:
                raise RuntimeError("Cannot safely establish the pressure experiment")
            self.started = step
        elif (
            self.started is not None and not self.released and step >= self.started + 4
        ):
            size = 0
            self.released = True
        if size is None:
            return
        self.connection.send(size)
        if not self.connection.poll(60):
            raise TimeoutError("Pressure worker did not acknowledge")
        event = {"step": step, **self.connection.recv()}
        with self.output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
        print(f"Pressure event: {event}", flush=True)


def training_args(options):
    from scripts.experiments.auto_block_swap_probe import prepare_args

    args = prepare_args(options.fixture)
    args.auto_block_swap = options.mode == "dynamic"
    args.auto_block_swap_mode = "dynamic"
    args.auto_block_swap_interval = options.interval
    args.blocks_to_swap = options.blocks
    args.output_dir = str(options.output)
    args.logging_dir = None
    args.log_with = None
    args.output_name = "dynamic-acceptance"
    args.max_train_steps = options.steps
    args.max_train_epochs = None
    args.save_every_n_steps = None
    args.save_every_n_epochs = None
    args.save_state = False
    args.save_state_on_train_end = False
    args.resume = None
    args.gradient_accumulation_steps = options.accumulation
    args.block_swap_profile_jsonl = "off"
    return args


def verify_checkpoint(options):
    from safetensors import safe_open
    import torch

    path = options.output / "dynamic-acceptance.safetensors"
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        steps = int((checkpoint.metadata() or {}).get("ss_steps", -1))
        finite = all(
            torch.isfinite(checkpoint.get_tensor(k)).all().item()
            for k in checkpoint.keys()
        )
    if steps != options.steps or not finite:
        raise RuntimeError("Checkpoint step/finite acceptance failed")
    return steps, finite


def run(options):
    import torch
    from train import AnimaTrainer
    from library.training import loop
    from scripts.experiments.dynamic_block_swap_measurement import UpdateRecorder

    if not (options.fixture / "config.toml").is_file():
        raise ValueError("An existing isolated AUTO fixture is required")
    options.output.mkdir(parents=True, exist_ok=False)
    args = training_args(options)
    if options.memory_history:
        torch.cuda.memory._record_memory_history(max_entries=10000)
    write_result(
        options.output / "request.json",
        vars(options)
        | {
            "fixture": str(options.fixture),
            "output": str(options.output),
        },
    )
    ctx = multiprocessing.get_context("spawn")
    connection, child = ctx.Pipe()
    worker = None
    schedule = None
    original_step = loop._run_step
    started = time.monotonic()
    try:
        if options.pressure:
            worker = ctx.Process(target=pressure_worker, args=(child,))
            worker.start()
            if not connection.poll(60) or not connection.recv().get("ready"):
                raise RuntimeError("Pressure worker failed to initialize")
            schedule = PressureSchedule(connection, options.output / "pressure.jsonl")
        loop._run_step = UpdateRecorder(options, args, original_step, schedule)
        AnimaTrainer().train(args)
        steps, finite = verify_checkpoint(options)
        if schedule is not None and not schedule.released:
            raise RuntimeError(
                "Dynamic run did not complete the pressure/release scenario"
            )
        write_result(
            options.output / "completion.json",
            {
                "status": "completed",
                "steps": steps,
                "finite": finite,
                "seconds": time.monotonic() - started,
                "final_blocks": args.blocks_to_swap,
            },
        )
    finally:
        loop._run_step = original_step
        if worker is not None:
            if worker.is_alive():
                try:
                    connection.send(None)
                except (BrokenPipeError, EOFError, OSError):
                    pass
            worker.join(timeout=15)
            if worker.is_alive():
                worker.terminate()
                worker.join()
        connection.close()
        child.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=["dynamic", "fixed"], default="dynamic")
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--interval", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=26)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--pressure", action="store_true")
    parser.add_argument("--memory-history", action="store_true")
    options = parser.parse_args()
    if not 1 <= options.steps <= 128 or not 1 <= options.accumulation <= 4:
        parser.error("Bounded acceptance requires 1..128 steps, accumulation 1..4")
    if options.pressure and options.mode != "dynamic":
        parser.error("Pressure schedule requires dynamic mode")
    options.fixture = options.fixture.resolve()
    options.output = options.output.resolve()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault(
        "TORCHINDUCTOR_CACHE_DIR", str(options.fixture / "compiler-cache")
    )
    run(options)


if __name__ == "__main__":
    main()
