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
    accelerator.unwrap_model(model).prepare_block_swap_before_forward()

    return model


def on_validation_step_end(trainer, ctx: TrainCtx, batch) -> None:
    if trainer.is_swapping_blocks:
        # prepare for next forward: because backward pass is not called, we need to prepare it here
        ctx.accelerator.unwrap_model(ctx.unet).prepare_block_swap_before_forward()
