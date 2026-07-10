"""Dataset/network/optimizer/accelerator bootstrap wrappers for AnimaTrainer."""

from __future__ import annotations

from typing import Optional

import torch

from library.training.bootstrap import TrainingBootstrap


class TrainerBootstrapMixin:
    @staticmethod
    def _parse_profile_steps(args) -> tuple[int, int] | None:
        from library.training.rng_state import parse_profile_steps

        return parse_profile_steps(args)

    @staticmethod
    def _switch_rng_state(
        seed: int,
    ) -> tuple[torch.ByteTensor, Optional[torch.ByteTensor], tuple]:
        from library.training.rng_state import switch_rng_state

        return switch_rng_state(seed)

    @staticmethod
    def _restore_rng_state(
        rng_states: tuple[torch.ByteTensor, Optional[torch.ByteTensor], tuple],
    ):
        from library.training.rng_state import restore_rng_state

        return restore_rng_state(rng_states)

    @staticmethod
    def _apply_train_batch_size_to_user_config(user_config: dict, args) -> None:
        TrainingBootstrap.apply_train_batch_size_to_user_config(user_config, args)

    def _prepare_dataset(self, args):
        """Build train/val dataset groups and the collator shared by both loaders."""
        from library.training.bootstrap_unpack import unpack_dataset_result

        return unpack_dataset_result(self.bootstrap.prepare_dataset(self, args))

    def _create_and_apply_network(
        self,
        args,
        accelerator,
        vae,
        text_encoder,
        unet,
        text_encoders,
        weight_dtype,
    ):
        """Import network module, merge base weights, build LoRA, apply to the model."""
        from library.training.bootstrap_unpack import unpack_network_result

        return unpack_network_result(
            self.bootstrap.create_and_apply_network(
                self,
                args,
                accelerator,
                vae,
                text_encoder,
                unet,
                text_encoders,
                weight_dtype,
            )
        )

    def _setup_optimizer_and_dataloader(
        self,
        args,
        accelerator,
        network,
        train_dataset_group,
        val_dataset_group,
        collator,
    ):
        """Build optimizer, dataloaders, and LR scheduler; finalize max_train_steps."""
        from library.training.bootstrap_unpack import unpack_optimizer_result

        return unpack_optimizer_result(
            self.bootstrap.setup_optimizer_and_dataloader(
                args,
                accelerator,
                network,
                train_dataset_group,
                val_dataset_group,
                collator,
            )
        )

    def _prepare_with_accelerator(
        self,
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
    ):
        """Cast model dtypes, run accelerator.prepare, flip train/eval, optional torch.compile."""
        from library.training.bootstrap_unpack import unpack_accelerator_result

        return unpack_accelerator_result(
            self.bootstrap.prepare_with_accelerator(
                self,
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
            )
        )

    def train(self, args):
        from library.training.train_session import run_training_session

        run_training_session(self, args)
