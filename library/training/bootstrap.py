"""Internal training bootstrap helpers.

This module owns the pre-loop assembly steps that are large enough to test
independently but are still part of the trainer's private implementation:
dataset/collator creation, network construction, optimizer/dataloader setup,
and accelerator preparation.
"""

from __future__ import annotations

import importlib
import logging
import math
import os
from dataclasses import dataclass
from multiprocessing import Value
from pathlib import Path
import sys
from typing import Any, Optional

import torch

from library.config import loader as config_util
from library.config.io import load_dataset_config_from_base
from library.config.loader import BlueprintGenerator, ConfigSanitizer
from library.datasets import (
    collator_class,
    load_arbitrary_dataset,
)
from library.runtime.accelerator import patch_accelerator_for_fp16_training
from library.training.optimizers import get_optimizer, get_optimizer_train_eval_fn
from library.training.schedulers import get_scheduler_fix
from networks import all_network_kwargs

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

# Network-module-consumed flags (networks.lora_anima / networks.methods.*).
# Source of truth is the registry in networks/__init__.py.
NETWORK_KWARG_ALLOWLIST: tuple[str, ...] = all_network_kwargs()

# Top-level training args that still flow through net_kwargs because a network
# module consumes them. Keep this list small and explicit.
EXTRA_FORWARDED_TOP_LEVEL_ARGS: tuple[str, ...] = (
    "gradient_accumulation_steps",
)


@dataclass(frozen=True)
class DatasetBuildResult:
    train_dataset_group: Any
    val_dataset_group: Any
    current_epoch: Any
    current_step: Any
    collator: Any
    use_user_config: bool
    use_dreambooth_method: bool


@dataclass(frozen=True)
class NetworkBuildResult:
    network: Any
    net_kwargs: dict[str, str]
    train_unet: bool
    train_text_encoder: bool


@dataclass(frozen=True)
class OptimizerBuildResult:
    optimizer: Any
    optimizer_name: str
    optimizer_args: Any
    optimizer_train_fn: Any
    optimizer_eval_fn: Any
    text_encoder_lr: Any
    lr_descriptions: Optional[list]
    train_dataloader: Any
    val_dataloader: Any
    lr_scheduler: Any


@dataclass(frozen=True)
class AcceleratorPrepareResult:
    network: Any
    optimizer: Any
    train_dataloader: Any
    val_dataloader: Any
    lr_scheduler: Any
    training_model: Any
    unet: Any
    text_encoders: list
    text_encoder: Any
    unet_weight_dtype: torch.dtype


class TrainingBootstrap:
    """Private assembly service used by ``AnimaTrainer``.

    The trainer still owns policy and lifecycle orchestration. This class only
    collects the bulky setup steps so tests can inject a fake bootstrap or call
    individual assembly units without driving a full training run.
    """

    @staticmethod
    def apply_train_batch_size_to_user_config(user_config: dict, args) -> None:
        train_batch_size = getattr(args, "train_batch_size", None)
        try:
            train_batch_size = int(train_batch_size)
        except (TypeError, ValueError):
            return
        if train_batch_size <= 1:
            return

        changed = 0
        for dataset_config in user_config.get("datasets", []):
            if not isinstance(dataset_config, dict):
                continue
            if dataset_config.get("batch_size") != train_batch_size:
                dataset_config["batch_size"] = train_batch_size
                changed += 1

        if changed:
            logger.info(
                "Applied train_batch_size=%s to %s dataset batch_size setting(s)",
                train_batch_size,
                changed,
            )

    @staticmethod
    def build_net_kwargs(args) -> dict[str, str]:
        net_kwargs = {}
        if args.network_args is not None:
            for net_arg in args.network_args:
                key, value = net_arg.split("=", 1)
                net_kwargs[key] = value

        for key in NETWORK_KWARG_ALLOWLIST + EXTRA_FORWARDED_TOP_LEVEL_ARGS:
            if (
                key not in net_kwargs
                and hasattr(args, key)
                and getattr(args, key) is not None
            ):
                net_kwargs[key] = str(getattr(args, key))
        return net_kwargs

    def prepare_dataset(self, trainer, args) -> DatasetBuildResult:
        """Build train/val dataset groups and the collator shared by loaders."""
        use_dreambooth_method = args.in_json is None
        use_user_config = args.dataset_config is not None

        if args.dataset_class is None:
            blueprint_generator = BlueprintGenerator(
                ConfigSanitizer(support_dropout=True)
            )
            if use_user_config:
                logger.info(f"Loading dataset config from {args.dataset_config}")
                user_config = config_util.load_user_config(args.dataset_config)
                ignored = ["train_data_dir", "reg_data_dir", "in_json"]
                if any(getattr(args, attr) is not None for attr in ignored):
                    logger.warning(
                        "ignoring the following options because config file is found: {0}".format(
                            ", ".join(ignored)
                        )
                    )
            else:
                base_ds = load_dataset_config_from_base(
                    overrides=vars(args),
                    method=getattr(args, "method", None),
                    methods_subdir=getattr(args, "methods_subdir", None) or "methods",
                )
                if base_ds is not None:
                    logger.info("Loading dataset config from configs/base.toml")
                    user_config = base_ds
                    use_user_config = True
                elif use_dreambooth_method:
                    logger.info("Using DreamBooth method.")
                    user_config = {
                        "datasets": [
                            {
                                "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                    args.train_data_dir, args.reg_data_dir
                                )
                            }
                        ]
                    }
                else:
                    logger.info("Training with captions.")
                    user_config = {
                        "datasets": [
                            {
                                "subsets": [
                                    {
                                        "image_dir": args.train_data_dir,
                                        "metadata_file": args.in_json,
                                    }
                                ]
                            }
                        ]
                    }

            sample_ratio = getattr(args, "sample_ratio", None)
            if sample_ratio is not None:
                for ds in user_config.get("datasets", []):
                    for sub in ds.get("subsets", []):
                        sub["sample_ratio"] = sample_ratio
                logger.info(f"Applied --sample_ratio={sample_ratio} to all subsets")

            self.apply_train_batch_size_to_user_config(user_config, args)

            blueprint = blueprint_generator.generate(user_config, args)
            train_dataset_group, val_dataset_group = (
                config_util.generate_dataset_group_by_blueprint(
                    blueprint.dataset_group,
                    constant_token_buckets=True,
                )
            )

            rates = [
                subset.caption_dropout_rate
                for ds in train_dataset_group.datasets
                for subset in ds.subsets
            ]
            trainer._state.caption_dropout_enabled = bool(rates) and any(
                r > 0 for r in rates
            )
            if trainer._state.caption_dropout_enabled:
                logger.info(f"caption dropout ENABLED -- per-subset rates: {rates}")
            else:
                logger.info("caption dropout DISABLED (rate=0.0 on all subsets)")
        else:
            train_dataset_group = load_arbitrary_dataset(args)
            val_dataset_group = None

        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        ds_for_collator = (
            train_dataset_group if args.max_data_loader_n_workers == 0 else None
        )
        collator = collator_class(current_epoch, current_step, ds_for_collator)

        return DatasetBuildResult(
            train_dataset_group=train_dataset_group,
            val_dataset_group=val_dataset_group,
            current_epoch=current_epoch,
            current_step=current_step,
            collator=collator,
            use_user_config=use_user_config,
            use_dreambooth_method=use_dreambooth_method,
        )

    def create_and_apply_network(
        self,
        trainer,
        args,
        accelerator,
        vae,
        text_encoder,
        unet,
        text_encoders,
        weight_dtype,
    ) -> NetworkBuildResult | None:
        """Import network module, merge base weights, build LoRA, apply it."""
        sys.path.append(str(ROOT))
        accelerator.print("import network module:", args.network_module)
        network_module = importlib.import_module(args.network_module)

        if args.base_weights is not None:
            for i, weight_path in enumerate(args.base_weights):
                if (
                    args.base_weights_multiplier is None
                    or len(args.base_weights_multiplier) <= i
                ):
                    multiplier = 1.0
                else:
                    multiplier = args.base_weights_multiplier[i]

                accelerator.print(
                    f"merging module: {weight_path} with multiplier {multiplier}"
                )

                module, weights_sd = network_module.create_network_from_weights(
                    multiplier, weight_path, vae, text_encoder, unet, for_inference=True
                )
                module.merge_to(
                    text_encoder,
                    unet,
                    weights_sd,
                    weight_dtype,
                    accelerator.device if args.lowram else "cpu",
                )

            accelerator.print(f"all weights merged: {', '.join(args.base_weights)}")

        net_kwargs = self.build_net_kwargs(args)

        if args.dim_from_weights:
            network, _ = network_module.create_network_from_weights(
                1, args.network_weights, vae, text_encoder, unet, **net_kwargs
            )
        else:
            if "dropout" not in net_kwargs:
                net_kwargs["dropout"] = args.network_dropout

            network = network_module.create_network(
                1.0,
                args.network_dim,
                args.network_alpha,
                vae,
                text_encoder,
                unet,
                neuron_dropout=args.network_dropout,
                **net_kwargs,
            )
        if network is None:
            return None

        if hasattr(network, "prepare_network"):
            network.prepare_network(args)
        if args.scale_weight_norms and not hasattr(
            network, "apply_max_norm_regularization"
        ):
            logger.warning(
                "warning: scale_weight_norms is specified but the network does not support it"
            )
            args.scale_weight_norms = False

        trainer.post_process_network(args, accelerator, network, text_encoders, unet)

        train_unet = not args.network_train_text_encoder_only
        train_text_encoder = trainer.is_train_text_encoder(args)
        network.apply_to(text_encoder, unet, train_text_encoder, train_unet)

        if args.network_weights is not None:
            info = network.load_weights(args.network_weights)
            accelerator.print(
                f"load network weights from {args.network_weights}: {info}"
            )

        if args.gradient_checkpointing:
            if args.cpu_offload_checkpointing:
                unet.enable_gradient_checkpointing(cpu_offload=True)
            else:
                unet.enable_gradient_checkpointing()

            for t_enc, flag in zip(
                text_encoders,
                trainer.get_text_encoders_train_flags(args, text_encoders),
            ):
                if flag and t_enc.supports_gradient_checkpointing:
                    t_enc.gradient_checkpointing_enable()
            network.enable_gradient_checkpointing()

        return NetworkBuildResult(
            network=network,
            net_kwargs=net_kwargs,
            train_unet=train_unet,
            train_text_encoder=train_text_encoder,
        )

    def setup_optimizer_and_dataloader(
        self,
        args,
        accelerator,
        network,
        train_dataset_group,
        val_dataset_group,
        collator,
    ) -> OptimizerBuildResult:
        """Build optimizer, dataloaders, and LR scheduler."""
        accelerator.print("prepare optimizer, data loader etc.")

        support_multiple_lrs = hasattr(
            network, "prepare_optimizer_params_with_multiple_te_lrs"
        )
        if support_multiple_lrs:
            text_encoder_lr = args.text_encoder_lr
        else:
            if (
                args.text_encoder_lr is None
                or isinstance(args.text_encoder_lr, float)
                or isinstance(args.text_encoder_lr, int)
            ):
                text_encoder_lr = args.text_encoder_lr
            else:
                text_encoder_lr = (
                    None if len(args.text_encoder_lr) == 0 else args.text_encoder_lr[0]
                )
        try:
            if support_multiple_lrs:
                results = network.prepare_optimizer_params_with_multiple_te_lrs(
                    text_encoder_lr, args.unet_lr, args.learning_rate
                )
            else:
                results = network.prepare_optimizer_params(
                    text_encoder_lr, args.unet_lr, args.learning_rate
                )
            if isinstance(results, tuple):
                trainable_params = results[0]
                lr_descriptions = results[1]
            else:
                trainable_params = results
                lr_descriptions = None
        except TypeError:
            trainable_params = network.prepare_optimizer_params(
                text_encoder_lr, args.unet_lr
            )
            lr_descriptions = None

        optimizer_name, optimizer_args, optimizer = get_optimizer(
            args, trainable_params
        )
        optimizer_train_fn, optimizer_eval_fn = get_optimizer_train_eval_fn(
            optimizer, args
        )

        train_dataset_group.set_current_strategies()
        if val_dataset_group is not None:
            val_dataset_group.set_current_strategies()

        n_workers = min(args.max_data_loader_n_workers, os.cpu_count())
        persistent_workers = args.persistent_data_loader_workers and n_workers > 0

        dataloader_kwargs = {
            "batch_size": 1,
            "collate_fn": collator,
            "num_workers": n_workers,
            "persistent_workers": persistent_workers,
            "pin_memory": args.dataloader_pin_memory,
        }
        if n_workers > 0:
            dataloader_kwargs["prefetch_factor"] = args.dataloader_prefetch_factor

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            shuffle=True,
            **dataloader_kwargs,
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset_group if val_dataset_group is not None else [],
            shuffle=False,
            **dataloader_kwargs,
        )

        if args.max_train_epochs is not None:
            args.max_train_steps = args.max_train_epochs * math.ceil(
                len(train_dataloader)
                / accelerator.num_processes
                / args.gradient_accumulation_steps
            )
            accelerator.print(
                f"override steps. steps for {args.max_train_epochs} epochs is"
            )

        train_dataset_group.set_max_train_steps(args.max_train_steps)
        lr_scheduler = get_scheduler_fix(args, optimizer, accelerator.num_processes)

        return OptimizerBuildResult(
            optimizer=optimizer,
            optimizer_name=optimizer_name,
            optimizer_args=optimizer_args,
            optimizer_train_fn=optimizer_train_fn,
            optimizer_eval_fn=optimizer_eval_fn,
            text_encoder_lr=text_encoder_lr,
            lr_descriptions=lr_descriptions,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            lr_scheduler=lr_scheduler,
        )

    def prepare_with_accelerator(
        self,
        trainer,
        args,
        accelerator,
        network,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
        unet,
        text_encoders,
        text_encoder,
        vae,
        vae_dtype,
        weight_dtype,
        train_unet,
        train_text_encoder,
        cache_latents,
    ) -> AcceleratorPrepareResult:
        """Cast dtypes, run accelerator.prepare, and flip train/eval modes."""
        if args.full_fp16:
            assert args.mixed_precision == "fp16", (
                "full_fp16 requires mixed precision='fp16'"
            )
            accelerator.print("enable full fp16 training.")
            network.to(weight_dtype)
        elif args.full_bf16:
            assert args.mixed_precision == "bf16", (
                "full_bf16 requires mixed precision='bf16'"
            )
            accelerator.print("enable full bf16 training.")
            network.to(weight_dtype)

        unet_weight_dtype = te_weight_dtype = weight_dtype

        unet.requires_grad_(False)
        if trainer.cast_unet(args):
            unet.to(dtype=unet_weight_dtype)
        for t_enc in text_encoders:
            if t_enc is None:
                continue
            t_enc.requires_grad_(False)
            if t_enc.device.type != "cpu" and trainer.cast_text_encoder(args):
                t_enc.to(dtype=te_weight_dtype)

        if train_unet:
            unet = trainer.prepare_unet_with_accelerator(args, accelerator, unet)
        else:
            unet.to(
                accelerator.device,
                dtype=unet_weight_dtype if trainer.cast_unet(args) else None,
            )
        if train_text_encoder:
            text_encoders = [
                (accelerator.prepare(t_enc) if flag else t_enc)
                for t_enc, flag in zip(
                    text_encoders,
                    trainer.get_text_encoders_train_flags(args, text_encoders),
                )
            ]
            if len(text_encoders) > 1:
                text_encoder = text_encoders
            else:
                text_encoder = text_encoders[0]

        network, optimizer, train_dataloader, val_dataloader, lr_scheduler = (
            accelerator.prepare(
                network, optimizer, train_dataloader, val_dataloader, lr_scheduler
            )
        )
        training_model = network

        if args.gradient_checkpointing:
            unet.train()
            for i, (t_enc, flag) in enumerate(
                zip(
                    text_encoders,
                    trainer.get_text_encoders_train_flags(args, text_encoders),
                )
            ):
                if t_enc is None:
                    continue
                t_enc.train()
                if flag:
                    trainer.prepare_text_encoder_grad_ckpt_workaround(i, t_enc)
        else:
            unet.eval()
            for t_enc in text_encoders:
                if t_enc is None:
                    continue
                t_enc.eval()

        accelerator.unwrap_model(network).prepare_grad_etc(text_encoder, unet)

        if not cache_latents:
            vae.requires_grad_(False)
            vae.eval()
            vae.to(accelerator.device, dtype=vae_dtype)

        if args.full_fp16:
            patch_accelerator_for_fp16_training(accelerator)

        return AcceleratorPrepareResult(
            network=network,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            lr_scheduler=lr_scheduler,
            training_model=training_model,
            unet=unet,
            text_encoders=text_encoders,
            text_encoder=text_encoder,
            unet_weight_dtype=unet_weight_dtype,
        )
