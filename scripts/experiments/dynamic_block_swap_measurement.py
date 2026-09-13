"""Whole-update timing and storage diagnostics for the bounded live probe."""

from __future__ import annotations

import json
import time

from library.training.auto_block_swap.process import write_result


def offloader_snapshot(model):
    offloader = model.offloader
    return {
        "blocks": offloader.blocks_to_swap,
        "forward_only": offloader.forward_only,
        "futures": len(offloader.futures),
        "master_types": sorted({
            type(master).__name__
            for block in offloader._cpu_weight_masters for master in block.values()
        }),
        "gpu_weights_by_block": [
            sum(p.numel() * p.element_size() for p in block.parameters()
                if not p.requires_grad and p.device.type == "cuda")
            for block in model.blocks
        ],
        "master_gpu_bytes": sum(
            master.params4bit.numel() * master.params4bit.element_size()
            for block in offloader._cpu_weight_masters for master in block.values()
            if hasattr(master, "params4bit") and master.params4bit.device.type == "cuda"
        ),
        "plan_job_counts": [
            list(map(len, plan)) for plan in list(offloader._swap_plan_cache.values())[:2]
        ],
    }


class UpdateRecorder:
    def __init__(self, options, args, original_step, schedule):
        self.options = options
        self.args = args
        self.original_step = original_step
        self.schedule = schedule
        self.shapes = []
        self.started = None

    def __call__(self, trainer, state, batch):
        import torch

        if self.schedule is not None:
            self.schedule.before(state)
        if not self.shapes:
            self.started = time.perf_counter()
        self.shapes.append(list(batch["latents"].shape))
        loss = self.original_step(trainer, state, batch)
        if not state.accelerator.sync_gradients:
            return loss
        torch.cuda.synchronize()
        row = {
            "step": state.global_step + 1,
            "seconds": time.perf_counter() - self.started,
            "blocks": self.args.blocks_to_swap,
            "shapes": sorted(self.shapes),
            "microsteps": len(self.shapes),
            "loss": float(loss.detach()),
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "allocated": torch.cuda.memory_allocated(),
            "dynamo_unique_graphs": torch._dynamo.utils.counters["stats"]["unique_graphs"],
        }
        if state.global_step < 2:
            row["offloader"] = offloader_snapshot(state.accelerator.unwrap_model(state.unet))
            if self.options.memory_history:
                write_result(
                    self.options.output / f"memory-step-{state.global_step + 1}.json",
                    {"active": [
                        block for segment in torch.cuda.memory_snapshot()
                        for block in segment["blocks"] if block["state"] == "active_allocated"
                    ]},
                )
        with (self.options.output / "updates.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        self.shapes.clear()
        return loss
