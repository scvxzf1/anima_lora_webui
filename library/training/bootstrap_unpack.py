"""Unpack TrainingBootstrap result dataclasses for AnimaTrainer facades."""

from __future__ import annotations

from typing import Any, Optional


def unpack_network_result(result) -> Optional[tuple[Any, dict, bool, bool]]:
    if result is None:
        return None
    return (
        result.network,
        result.net_kwargs,
        result.train_unet,
        result.train_text_encoder,
    )


def unpack_optimizer_result(result) -> tuple:
    return (
        result.optimizer,
        result.optimizer_name,
        result.optimizer_args,
        result.optimizer_train_fn,
        result.optimizer_eval_fn,
        result.text_encoder_lr,
        result.lr_descriptions,
        result.train_dataloader,
        result.val_dataloader,
        result.lr_scheduler,
    )


def unpack_accelerator_result(result) -> tuple:
    return (
        result.network,
        result.optimizer,
        result.train_dataloader,
        result.val_dataloader,
        result.lr_scheduler,
        result.training_model,
        result.unet,
        result.text_encoders,
        result.text_encoder,
        result.unet_weight_dtype,
    )


def unpack_dataset_result(result) -> tuple:
    return (
        result.train_dataset_group,
        result.val_dataset_group,
        result.current_epoch,
        result.current_step,
        result.collator,
        result.use_user_config,
        result.use_dreambooth_method,
    )
