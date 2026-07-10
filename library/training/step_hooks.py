"""Per-step adapter hooks dispatched by AnimaTrainer."""

from __future__ import annotations

from library.training.contexts import TrainCtx
from library.training.method_adapter import StepCtx


def on_step_start(trainer, ctx: TrainCtx, batch, *, is_train: bool = True) -> None:
    if not trainer._adapters:
        return
    step_ctx = StepCtx(
        args=ctx.args,
        accelerator=ctx.accelerator,
        network=ctx.network,
        weight_dtype=ctx.weight_dtype,
    )
    for adapter in trainer._adapters:
        adapter.on_step_start(step_ctx, batch, is_train=is_train)


def run_after_backward(trainer, ctx: TrainCtx) -> None:
    """Dispatch the post-backward hook to adapters (between
    ``accelerator.backward`` and gradient clipping)."""
    if not trainer._adapters:
        return
    step_ctx = StepCtx(
        args=ctx.args,
        accelerator=ctx.accelerator,
        network=ctx.network,
        weight_dtype=ctx.weight_dtype,
    )
    for adapter in trainer._adapters:
        adapter.after_backward(step_ctx)
