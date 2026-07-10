"""Step log construction helpers extracted from AnimaTrainer.

Keeps loss / LR / VR / prior log keys identical to the previous train.py method.
"""

from __future__ import annotations

import argparse

from library.training.optimizers import is_prodigy_plus_schedulefree_type


def generate_step_logs(
    trainer,
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
    logs = {"loss/current": current_loss, "loss/average": avr_loss}

    if keys_scaled is not None:
        logs["max_norm/keys_scaled"] = keys_scaled
        logs["max_norm/max_key_norm"] = maximum_norm
    if mean_norm is not None:
        logs["norm/avg_key_norm"] = mean_norm
    if mean_grad_norm is not None:
        logs["norm/avg_grad_norm"] = mean_grad_norm
    if mean_combined_norm is not None:
        logs["norm/avg_combined_norm"] = mean_combined_norm

    if float(getattr(args, "vr_loss_weight", 0.0) or 0.0) > 0.0:
        lambda_ema = trainer._state.vr.get("lambda_ema")
        lambda_batch = trainer._state.vr.get("lambda_batch")
        if isinstance(lambda_ema, float):
            logs["vr/lambda_ema"] = lambda_ema
        if isinstance(lambda_batch, float):
            logs["vr/lambda_batch"] = lambda_batch
    if float(getattr(args, "prior_preservation_weight", 0.0) or 0.0) > 0.0:
        logs["prior_preservation/weight"] = float(args.prior_preservation_weight)
    if float(getattr(args, "inverted_mask_prior_weight", 0.0) or 0.0) > 0.0:
        logs["inverted_mask_prior/weight"] = float(args.inverted_mask_prior_weight)

    def prodigy_plus_effective_lr(group):
        d = group.get("d")
        lr = group.get("effective_lr", group.get("lr"))
        if d is None or lr is None:
            return None
        return d * lr

    is_prodigy_plus = is_prodigy_plus_schedulefree_type(args)
    lrs = lr_scheduler.get_last_lr()
    for i, lr in enumerate(lrs):
        if lr_descriptions is not None:
            lr_desc = lr_descriptions[i]
        else:
            idx = i - (0 if args.network_train_unet_only else -1)
            if idx == -1:
                lr_desc = "textencoder"
            else:
                if len(lrs) > 2:
                    lr_desc = f"group{idx}"
                else:
                    lr_desc = "unet"

        logs[f"lr/{lr_desc}"] = lr

        if (
            args.optimizer_type.lower().startswith("DAdapt".lower())
            or args.optimizer_type.lower() == "Prodigy".lower()
        ):
            # tracking d*lr value
            logs[f"lr/d*lr/{lr_desc}"] = (
                lr_scheduler.optimizers[-1].param_groups[i]["d"]
                * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
            )
        if is_prodigy_plus and optimizer is not None:
            effective_lr = prodigy_plus_effective_lr(optimizer.param_groups[i])
            if effective_lr is not None:
                logs[f"lr/d*lr/{lr_desc}"] = effective_lr
                if i == 0:
                    logs["lr/d*lr"] = effective_lr
    else:
        idx = 0
        if not args.network_train_unet_only:
            logs["lr/textencoder"] = float(lrs[0])
            idx = 1

        for i in range(idx, len(lrs)):
            logs[f"lr/group{i}"] = float(lrs[i])
            if (
                args.optimizer_type.lower().startswith("DAdapt".lower())
                or args.optimizer_type.lower() == "Prodigy".lower()
            ):
                logs[f"lr/d*lr/group{i}"] = (
                    lr_scheduler.optimizers[-1].param_groups[i]["d"]
                    * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
                )
            if is_prodigy_plus and optimizer is not None:
                effective_lr = prodigy_plus_effective_lr(optimizer.param_groups[i])
                if effective_lr is not None:
                    logs[f"lr/d*lr/group{i}"] = effective_lr

    return logs

