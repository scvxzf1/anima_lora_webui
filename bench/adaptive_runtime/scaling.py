"""Opt-in AMP scaling for comparison probes and explicit checkpoint experiments."""

import math

import torch


def validate_scaling(args):
    if not math.isfinite(args.loss_scale) or args.loss_scale < 1:
        raise ValueError("Loss scale must be finite and >= 1")
    if getattr(args, "scaled_checkpoint", False):
        if (args.loss_scale == 1 or args.precision != "fp16-islands"
                or not args.checkpoint_every_step or not args.inputs
                or args.disposable_probe or args.capture_training or args.capture_linear
                or getattr(args, "record_replay", False)
                or getattr(args, "replay_reference", None)):
            raise ValueError("Scaled checkpoint experiment requires real-input FP16 checkpoints without capture/replay")
        return
    if args.loss_scale != 1 and (
        args.precision != "fp16-islands" or not args.disposable_probe
        or not args.capture_training or args.resume or args.checkpoint_every_step
    ):
        raise ValueError("Scaling experiment requires disposable FP16 training capture without checkpoints")


def make_scaler(scale, *, device="cuda"):
    return torch.amp.GradScaler(device, enabled=scale != 1, init_scale=scale,
                                growth_interval=2**31 - 1)


def backward_unscaled(loss, noisy, optimizer, scaler):
    scale = scaler.get_scale()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    # GradScaler only unscales optimizer parameters, not the diagnostic input leaf.
    if noisy.grad is not None and scale != 1:
        noisy.grad.div_(scale)
