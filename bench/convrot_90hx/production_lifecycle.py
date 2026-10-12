"""Instrument the production CLI lifecycle; synchronization is diagnostic only."""

import argparse
import json
import runpy
import sys
from pathlib import Path

import torch

from .manifest import GPU_UUID, ROOT, validate_device
from .model import drain
from library.runtime.block_swap_payload import block_swap_payload_residency


def install(trace):
    from library.anima import training
    from library.training import loop
    from library.runtime.offloading import ModelOffloader

    context = {"model": None, "step": 0}

    def point(label, **extra):
        model = context["model"]
        if model is not None:
            drain(model)
        torch.cuda.synchronize()
        record = {
            "phase": label, "step": context["step"],
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "managed_payload": block_swap_payload_residency(model) if model else {},
            **extra,
        }
        with trace.open("a") as file:
            file.write(json.dumps(record) + "\n")

    original_step = loop._run_step
    def completed_step(trainer, state, batch):
        context.update(model=state.unet, step=state.global_step + 1)
        result = original_step(trainer, state, batch)
        if result.grad_fn is not None or result.requires_grad:
            raise RuntimeError("Completed production step still retains its loss graph")
        point("completed_step_graph_released", loss=float(result))
        return result
    loop._run_step = completed_step

    original_live = training.decode_samples_for_live_preview
    def live(accelerator, args, vae, **kwargs):
        model = kwargs.get("dit")
        context["model"] = model
        original_to = model.to
        def model_to(*args, **kwargs):
            result = original_to(*args, **kwargs)
            if args and str(args[0]) == "cpu":
                point("model_cpu_park_complete")
            return result
        model.to = model_to
        point("decode_boundary_before_park")
        try:
            return original_live(accelerator, args, vae, **kwargs)
        finally:
            model.to = original_to
            point("decode_boundary_training_restored")
    training.decode_samples_for_live_preview = live

    original_decode = training.decode_pending_samples
    def decode(accelerator, args, vae):
        original_to = vae.to
        def vae_to(*args, **kwargs):
            result = original_to(*args, **kwargs)
            point("vae_device_transition", vae_device=str(vae.device))
            return result
        vae.to = vae_to
        point("vae_decode_before_load")
        try:
            return original_decode(accelerator, args, vae)
        finally:
            vae.to = original_to
            point("vae_decode_finished")
    training.decode_pending_samples = decode

    for method, label in [("park_blocks_on_cpu", "offloader_cpu_park_complete"),
                          ("restore_parked_blocks", "offloader_training_restore_complete")]:
        original = getattr(ModelOffloader, method)
        def wrapped(self, blocks, original=original, label=label):
            result = original(self, blocks)
            point(label, changed=result)
            return result
        setattr(ModelOffloader, method, wrapped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()
    if args.trace.exists():
        raise FileExistsError(args.trace)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    validate_device(GPU_UUID)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    install(args.trace)
    sys.argv = ["train.py", "--method", "lora", "--preset", "default",
                "--config_file", str(args.config)]
    runpy.run_path(str(ROOT / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
