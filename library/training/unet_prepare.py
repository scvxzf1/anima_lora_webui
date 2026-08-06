"""UNet accelerator prepare / block-swap helpers for AnimaTrainer."""

from __future__ import annotations

import argparse

import torch
from accelerate import Accelerator

from library.training.contexts import TrainCtx


def prepare_unet_with_accelerator(
    trainer,
    args: argparse.Namespace,
    accelerator: Accelerator,
    unet: torch.nn.Module,
) -> torch.nn.Module:
    # Re-apply with unsloth_offload if needed (after base has already enabled it).
    if trainer._use_unsloth_offload_checkpointing and args.gradient_checkpointing:
        unet.enable_gradient_checkpointing(unsloth_offload=True)

    if not trainer.is_swapping_blocks:
        return accelerator.prepare(unet)

    model = unet
    model = accelerator.prepare(
        model, device_placement=[not trainer.is_swapping_blocks]
    )
    accelerator.unwrap_model(model).move_to_device_except_swap_blocks(
        accelerator.device
    )
    # free_cache=False: steady-state training forward shapes are constant, so
    # the per-step empty_cache()+gc.collect() only makes the caching allocator
    # re-grow the same blocks each step (~1GB nvidia-smi swing on a 5060 Ti).
    accelerator.unwrap_model(model).prepare_block_swap_before_forward(free_cache=False)

    return model


def on_validation_step_end(trainer, ctx: TrainCtx, batch) -> None:
    if trainer.is_swapping_blocks:
        # prepare for next forward: because backward pass is not called, we need to prepare it here
        ctx.accelerator.unwrap_model(ctx.unet).prepare_block_swap_before_forward(
            free_cache=False
        )
