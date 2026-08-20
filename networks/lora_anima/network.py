# LoRANetwork: the module-assembly / training-orchestration core of the LoRA
# adapter stack for Anima. Targets DiT blocks (and optionally text-encoder
# attention) with pluggable per-module classes supplied by a NetworkSpec.

import logging
from typing import Dict, List, Optional, Union

import torch

from library.log import setup_logging
from library.training.metrics import MetricContext
from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_anima.persistence import (
    load_lora_network_weights,
    reabsorb_baked_inv_scale,
    save_lora_network_weights,
    strip_orig_mod_keys,
)
from networks.lora_anima import (
    application,
    builders,
    merge as merge_ops,
    optimizer_groups,
    regularization,
    router_stats,
    routing_state,
)
from networks.lora_anima.routers import (
    CROSSATTN_EMB_DIM,
    ContentRouter,
    FreqRouter,
    GlobalRouter,
)

setup_logging()
logger = logging.getLogger(__name__)


class LoRANetwork(torch.nn.Module):
    # Target modules: DiT blocks, embedders, final layer. embedders and final layer are excluded by default.
    ANIMA_TARGET_REPLACE_MODULE = [
        "Block",
        "PatchEmbed",
        "TimestepEmbedding",
        "FinalLayer",
    ]
    # Target modules: LLM Adapter blocks
    ANIMA_ADAPTER_TARGET_REPLACE_MODULE = ["LLMAdapterTransformerBlock"]
    # Target modules for text encoder (Qwen3)
    TEXT_ENCODER_TARGET_REPLACE_MODULE = [
        "Qwen3Attention",
        "Qwen3MLP",
        "Qwen3SdpaAttention",
        "Qwen3FlashAttention2",
    ]

    LORA_PREFIX_ANIMA = "lora_unet"  # ComfyUI compatible
    LORA_PREFIX_TEXT_ENCODER = "lora_te"  # Qwen3

    def __init__(
        self,
        text_encoders: list,
        unet,
        cfg: LoRANetworkCfg,
        *,
        multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self._anima_checkpoint_layout = getattr(
            unet, "_anima_checkpoint_layout", None
        )
        self._anima_base_sha256 = getattr(unet, "_anima_base_sha256", None)

        # Mutable runtime state — explicitly NOT in cfg. ``set_multiplier`` and
        # ``set_loraplus_lr_ratio`` write these post-construction; per-step
        # diagnostics (hit counters, σ caches) accumulate during training.
        self.multiplier = multiplier
        self.loraplus_lr_ratio = None
        self.loraplus_unet_lr_ratio = None
        self.loraplus_text_encoder_lr_ratio = None
        self._channel_scale_misses: List[str] = []
        self._channel_scale_hits: int = 0
        self._sigma_router_hits: int = 0
        self._hydra_router_hits: int = 0
        self._hydra_router_misses: int = 0
        self._last_sigma: Optional[torch.Tensor] = None
        # Hydra up-weight grad-norm snapshot (T-LoRA / σ-bucket conflict
        # diagnostic). Filled by ``capture_up_grad_stats`` between backward
        # and ``optimizer.zero_grad``; consumed by the ``hydra_up_grad``
        # metric. Values stay on-device until ``get_up_grad_stats`` runs the
        # D2H — capture happens every sync step but the metric only reads on
        # log steps, so the sync was the per-step bottleneck.
        self._last_up_grad_stats: Dict[str, object] = {}
        # Per-step cache for ``get_router_stats`` — both the progress-bar
        # postfix and the metrics layer call it on log steps. Cleared in
        # ``clear_step_caches`` so the next forward recomputes.
        self._router_stats_cache: Optional[Dict[str, object]] = None
        # Separate cache for the chimera dual-pool router stats — different
        # reduction (mean gates per pool, not argmax-histogram) and different
        # entropy normalization (per-pool log(K_pool)). Same lifecycle.
        self._chimera_router_stats_cache: Optional[Dict[str, object]] = None

        # Family-aware target containers (Krea-2-Raw migration). cfg fields
        # default None → anima class-attribute defaults, behavior unchanged.
        unet_targets = (
            cfg.unet_target_replace_modules
            if cfg.unet_target_replace_modules is not None
            else LoRANetwork.ANIMA_TARGET_REPLACE_MODULE
        )
        text_encoder_targets = (
            cfg.text_encoder_target_replace_modules
            if cfg.text_encoder_target_replace_modules is not None
            else LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE
        )
        builders.initialize_network_components(
            self,
            text_encoders,
            unet,
            cfg=cfg,
            multiplier=multiplier,
            unet_target_replace_modules=unet_targets,
            adapter_target_replace_modules=LoRANetwork.ANIMA_ADAPTER_TARGET_REPLACE_MODULE,
            text_encoder_target_replace_modules=text_encoder_targets,
            router_class=GlobalRouter,
            freq_router_class=FreqRouter,
            content_router_class=ContentRouter,
            crossattn_emb_dim=CROSSATTN_EMB_DIM,
            logger=logger,
        )

    def _wire_shared_sigma_buffers(self) -> None:
        return routing_state.wire_shared_sigma_buffers(self)

    def _wire_shared_fei_buffers(self) -> None:
        return routing_state.wire_shared_fei_buffers(self)

    def _wire_shared_routing_buffers(self) -> None:
        return routing_state.wire_shared_routing_buffers(self)

    def _wire_shared_content_routing_buffers(self) -> None:
        return routing_state.wire_shared_content_routing_buffers(self)

    def _wire_shared_freq_routing_buffers(self) -> None:
        return routing_state.wire_shared_freq_routing_buffers(self)

    def prepare_network(self, args):
        return application.prepare_network(self, args, logger=logger)

    def set_multiplier(self, multiplier):
        return application.set_multiplier(self, multiplier)

    def set_enabled(self, is_enabled):
        return application.set_enabled(self, is_enabled)

    def fuse_weights(self):
        return merge_ops.fuse_weights(self)

    def unfuse_weights(self):
        return merge_ops.unfuse_weights(self)

    def set_timestep_mask(self, timesteps: torch.Tensor, max_timestep: float = 1.0):
        return routing_state.set_timestep_mask(self, timesteps, max_timestep)

    def set_step_index(self, step_index: int) -> None:
        return application.set_step_index(self, step_index)

    def set_reft_timestep_mask(
        self, timesteps: torch.Tensor, max_timestep: float = 1.0
    ):
        return routing_state.set_reft_timestep_mask(self, timesteps, max_timestep)

    def clear_timestep_mask(self):
        return routing_state.clear_timestep_mask(self)

    def set_sigma(self, sigmas: torch.Tensor) -> None:
        return routing_state.set_sigma(self, sigmas)

    def clear_sigma(self) -> None:
        return routing_state.clear_sigma(self)

    def set_fei(self, fei: torch.Tensor) -> None:
        return routing_state.set_fei(self, fei)

    def clear_fei(self) -> None:
        return routing_state.clear_fei(self)

    def set_routing_weights(self, weights: torch.Tensor) -> None:
        return routing_state.set_routing_weights(self, weights)

    def clear_routing_weights(self) -> None:
        return routing_state.clear_routing_weights(self)

    def set_crossattn_routing(self, crossattn_emb: torch.Tensor) -> None:
        return routing_state.set_crossattn_routing(self, crossattn_emb)

    def set_freq_routing_weights(self, weights: torch.Tensor) -> None:
        return routing_state.set_freq_routing_weights(self, weights)

    def clear_freq_routing_weights(self) -> None:
        return routing_state.clear_freq_routing_weights(self)

    def set_content(self, crossattn_emb: torch.Tensor) -> None:
        return routing_state.set_content(self, crossattn_emb)

    def set_content_routing_weights(self, weights: torch.Tensor) -> None:
        return routing_state.set_content_routing_weights(self, weights)

    def clear_content_routing_weights(self) -> None:
        return routing_state.clear_content_routing_weights(self)

    def clear_step_caches(self) -> None:
        return routing_state.clear_step_caches(self)

    def step_balance_loss_warmup(self, global_step: int, max_train_steps: int) -> None:
        return router_stats.step_balance_loss_warmup(self, global_step, max_train_steps)

    @staticmethod
    def _switch_balance(gate: torch.Tensor) -> torch.Tensor:
        return router_stats.switch_balance(gate)

    def get_balance_loss(self) -> torch.Tensor:
        return router_stats.get_balance_loss(self)

    def _get_chimera_balance_loss(self) -> torch.Tensor:
        return router_stats.get_chimera_balance_loss(self)

    def get_router_entropy(self) -> Optional[float]:
        return router_stats.get_router_entropy(self)

    def get_router_stats(
        self,
    ) -> Dict[str, Union[float, List[float], List[List[float]], List[int]]]:
        return router_stats.get_router_stats(self)

    def get_chimera_router_stats(
        self,
    ) -> Dict[str, Union[float, List[float]]]:
        return router_stats.get_chimera_router_stats(self)

    def capture_up_grad_stats(self) -> None:
        return router_stats.capture_up_grad_stats(self)

    def get_up_grad_stats(self) -> Dict[str, List[float]]:
        return router_stats.get_up_grad_stats(self)

    def get_ortho_regularization(self) -> torch.Tensor:
        return router_stats.get_ortho_regularization(self)

    def metrics(self, ctx: MetricContext) -> dict[str, float]:
        return router_stats.metrics(self, ctx)

    @staticmethod
    def _strip_orig_mod_keys(state_dict):
        return strip_orig_mod_keys(state_dict)

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        state_dict = strip_orig_mod_keys(state_dict)
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def load_weights(self, file):
        return load_lora_network_weights(self, file)

    def _reabsorb_baked_inv_scale(self, weights_sd: Dict[str, torch.Tensor]) -> None:
        return reabsorb_baked_inv_scale(self, weights_sd)

    def apply_to(self, text_encoders, unet, apply_text_encoder=True, apply_unet=True):
        return application.apply_to(
            self,
            text_encoders,
            unet,
            apply_text_encoder=apply_text_encoder,
            apply_unet=apply_unet,
            logger=logger,
        )

    def is_mergeable(self):
        return merge_ops.is_mergeable(self)

    def merge_to(self, text_encoders, unet, weights_sd, dtype=None, device=None):
        return merge_ops.merge_lora_weights(
            self,
            text_encoders,
            unet,
            weights_sd,
            dtype=dtype,
            device=device,
        )

    def set_loraplus_lr_ratio(
        self, loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio
    ):
        return optimizer_groups.set_loraplus_lr_ratio(
            self,
            loraplus_lr_ratio,
            loraplus_unet_lr_ratio,
            loraplus_text_encoder_lr_ratio,
        )

    def prepare_optimizer_params_with_multiple_te_lrs(
        self, text_encoder_lr, unet_lr, default_lr
    ):
        return optimizer_groups.prepare_lora_optimizer_params(
            self, text_encoder_lr, unet_lr, default_lr
        )

    def enable_gradient_checkpointing(self):
        pass  # not supported

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        return save_lora_network_weights(self, file, dtype, metadata)

    def backup_weights(self):
        return merge_ops.backup_weights(self)

    def restore_weights(self):
        return merge_ops.restore_weights(self)

    def pre_calculation(self):
        return merge_ops.pre_calculation(self)

    def apply_max_norm_regularization(self, max_norm_value, device):
        return regularization.apply_max_norm_regularization(
            self,
            max_norm_value,
            device,
        )
