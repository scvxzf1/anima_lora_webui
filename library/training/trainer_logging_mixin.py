"""Logging wrappers shared by AnimaTrainer."""

from __future__ import annotations

import argparse

from accelerate import Accelerator

from library.training.log_dispatch import dispatch_logs


class TrainerLoggingMixin:
    def generate_step_logs(
        self,
        args: argparse.Namespace,
        current_loss,
        avr_loss,
        lr_scheduler,
        lr_descriptions,
        optimizer=None,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
        mean_grad_norm=None,
        mean_combined_norm=None,
    ):
        from library.training.step_logs import generate_step_logs as _generate_step_logs

        return _generate_step_logs(
            self,
            args,
            current_loss,
            avr_loss,
            lr_scheduler,
            lr_descriptions,
            optimizer=optimizer,
            keys_scaled=keys_scaled,
            mean_norm=mean_norm,
            maximum_norm=maximum_norm,
            mean_grad_norm=mean_grad_norm,
            mean_combined_norm=mean_combined_norm,
        )

    def step_logging(
        self, accelerator: Accelerator, logs: dict, global_step: int, epoch: int
    ):
        dispatch_logs(
            accelerator,
            logs,
            global_step,
            global_step,
            epoch,
            progress_sink=getattr(self, "progress_sink", None),
        )

    def epoch_logging(
        self, accelerator: Accelerator, logs: dict, global_step: int, epoch: int
    ):
        dispatch_logs(
            accelerator,
            logs,
            epoch,
            global_step,
            epoch,
            progress_sink=getattr(self, "progress_sink", None),
        )

    def val_logging(
        self,
        accelerator: Accelerator,
        logs: dict,
        global_step: int,
        epoch: int,
        val_step: int,
    ):
        dispatch_logs(
            accelerator,
            logs,
            global_step + val_step,
            global_step,
            epoch,
            val_step,
            progress_sink=getattr(self, "progress_sink", None),
        )
