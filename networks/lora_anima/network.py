# LoRANetwork: the module-assembly / training-orchestration core of the LoRA
# adapter stack for Anima. Targets DiT blocks (and optionally text-encoder
# attention) with pluggable per-module classes supplied by a NetworkSpec.

import logging
import re
from typing import Dict, List, Optional, Tuple, Union

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
    builders,
    merge as merge_ops,
    optimizer_groups,
    router_stats,
    routing_state,
)
from networks.lora_anima.targeting import compile_lora_target_patterns
from networks.lora_anima.routers import (
    CROSSATTN_EMB_DIM,
    ContentRouter,
    FreqRouter,
    GlobalRouter,
)
from networks.lora_modules import (
    ChimeraHydraInferenceModule,
    ChimeraHydraLoRAModule,
    LoRAModule,
    OrthoHydraLoRAModule,
    OrthoLoRAModule,
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

        # Local aliases for the closure body and the post-closure ReFT block.
        # Reading via `cfg.foo` works too; aliases just keep the diff small.
        module_class = cfg.module_class
        modules_dim = cfg.modules_dim
        dropout = cfg.dropout
        rank_dropout = cfg.rank_dropout
        module_dropout = cfg.module_dropout
        verbose = cfg.verbose
        alpha = cfg.alpha
        lora_dim = cfg.lora_dim
        train_llm_adapter = cfg.train_llm_adapter

        # Unified routing scope. ``cfg.router_targets`` is the single regex
        # that governs which Linears participate in routed adaptation (Hydra
        # MoE leaves + σ-feature concat + FEI-feature concat all share it).
        # From-weights path supplies an explicit name set per router family
        # (different families may have different module memberships in older
        # checkpoints); when present, the explicit set wins over the regex.
        _router_re = (
            re.compile(cfg.router_targets) if cfg.router_targets else None
        )

        self._sigma_router_names = (
            set(cfg.sigma_router_names) if cfg.sigma_router_names else None
        )
        self._sigma_router_re = (
            _router_re
            if (
                cfg.router_source == "sigma"
                and _router_re is not None
                and self._sigma_router_names is None
            )
            else None
        )

        self._fei_router_names = (
            set(cfg.fei_router_names) if cfg.fei_router_names else None
        )
        self._fei_router_re = (
            _router_re
            if (
                cfg.router_source == "fei"
                and _router_re is not None
                and self._fei_router_names is None
            )
            else None
        )
        self._fei_router_hits = 0
        # Modules built with ``use_global_router=True`` (shared_A +
        # ``route_per_layer=False``): the per-layer router is skipped and gates
        # arrive via the network-level ``GlobalRouter``. Counted separately
        # from ``_fei_router_hits`` because the per-layer FEI cat is bypassed.
        self._global_router_hits = 0
        # Retained as a network attr (library/inference/adapters.py reads it
        # via getattr); derived from cfg.router_source.
        self.use_fei_router = cfg.router_source == "fei"
        self.use_sigma_router = cfg.router_source == "sigma"
        # Shared-A Hydra layout + network-level router (FEI-on-Hydra global).
        # Toggle for the per-module construction loop below; lets Hydra /
        # OrthoHydra modules skip ``self.router`` and consume gates from the
        # ``GlobalRouter`` instead. Mirrors the FeRA (independent_A) routing
        # location without changing the underlying Hydra parameter layout.
        self._use_global_router_for_hydra = (
            cfg.use_moe_style == "shared_A"
            and not cfg.route_per_layer
            and cfg.router_source != "none"
        )

        # Per-module HydraLoRA gating. Matching modules get the Hydra class;
        # non-matching modules fall back to plain LoRA / OrthoLoRA so MoE
        # capacity is concentrated where specialization is actually learnable.
        # Fresh path: regex over `original_name`. From-weights path: explicit
        # name set detected from checkpoint keys. Explicit set wins. None on
        # both = apply MoE everywhere (legacy).
        self._hydra_router_names = (
            set(cfg.hydra_router_names) if cfg.hydra_router_names else None
        )
        self._hydra_router_re = (
            _router_re if (_router_re is not None and self._hydra_router_names is None)
            else None
        )

        if modules_dim is not None:
            logger.info("create LoRA network from weights")
        else:
            logger.info(
                f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}"
            )
            logger.info(
                f"neuron dropout: p={dropout}, rank dropout: p={rank_dropout}, module dropout: p={module_dropout}"
            )

        exclude_re_patterns = compile_lora_target_patterns(
            cfg.exclude_patterns,
            logger=logger,
        )
        include_re_patterns = compile_lora_target_patterns(
            cfg.include_patterns,
            logger=logger,
        )

        # create module instances
        def create_modules(
            is_unet: bool,
            text_encoder_idx: Optional[int],
            root_module: torch.nn.Module,
            target_replace_modules: List[str],
            default_dim: Optional[int] = None,
        ) -> Tuple[List[LoRAModule], List[str]]:
            return builders.create_lora_modules(
                self,
                is_unet=is_unet,
                text_encoder_idx=text_encoder_idx,
                root_module=root_module,
                target_replace_modules=target_replace_modules,
                default_dim=default_dim,
                exclude_patterns=exclude_re_patterns,
                include_patterns=include_re_patterns,
                logger=logger,
            )

        # Create LoRA for text encoders (Qwen3 - typically not trained for Anima)
        # Skip for OrthoLoRA since SVD init is expensive and TE modules are discarded in apply_to anyway
        self.text_encoder_loras: List[LoRAModule] = []
        skipped_te = []
        if text_encoders is not None and module_class not in (
            OrthoLoRAModule,
            OrthoHydraLoRAModule,
            ChimeraHydraLoRAModule,
            ChimeraHydraInferenceModule,
        ):
            for i, text_encoder in enumerate(text_encoders):
                if text_encoder is None:
                    continue
                logger.info(f"create LoRA for Text Encoder {i + 1}:")
                te_loras, te_skipped = create_modules(
                    False,
                    i,
                    text_encoder,
                    LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE,
                )
                logger.info(
                    f"create LoRA for Text Encoder {i + 1}: {len(te_loras)} modules."
                )
                self.text_encoder_loras.extend(te_loras)
                skipped_te += te_skipped

        # Create LoRA for DiT blocks
        target_modules = list(LoRANetwork.ANIMA_TARGET_REPLACE_MODULE)
        if train_llm_adapter:
            target_modules.extend(LoRANetwork.ANIMA_ADAPTER_TARGET_REPLACE_MODULE)

        self.unet_loras: List[LoRAModule]
        self.unet_loras, skipped_un = create_modules(True, None, unet, target_modules)

        logger.info(f"create LoRA for Anima DiT: {len(self.unet_loras)} modules.")
        if verbose:
            for lora in self.unet_loras:
                logger.info(f"\t{lora.lora_name:60} {lora.lora_dim}, {lora.alpha}")

        skipped = skipped_te + skipped_un
        if verbose and len(skipped) > 0:
            logger.warning(f"dim (rank) is 0, {len(skipped)} LoRA modules are skipped:")
            for name in skipped:
                logger.info(f"\t{name}")

        if cfg.channel_scales_dict is not None:
            logger.info(
                f"channel_scaling: {self._channel_scale_hits} DiT modules "
                f"received calibration-based input scaling"
            )
            if self._channel_scale_misses:
                logger.warning(
                    f"channel_scaling: {len(self._channel_scale_misses)} DiT modules "
                    f"have no calibration stats (first: {self._channel_scale_misses[:3]}). "
                    f"These will train without input rebalancing — regenerate the vendored "
                    f"calibration with `python bench/channel_stats/analyze_lora_input_channels.py "
                    f"--per_artist --dump_channel_stats networks/calibration/channel_stats.safetensors` "
                    f"if this is unexpected."
                )

        # Create ReFT modules on the DiT residual stream (block outputs), following
        # Wu et al. (2024) §3.3 — one intervention per selected block, not per
        # internal Linear. Selection is controlled by ``reft_layers``.
        self.text_encoder_refts, self.unet_refts = builders.create_reft_modules(
            unet,
            cfg=cfg,
            multiplier=multiplier,
            logger=logger,
        )

        # assertion: no duplicate names
        names = set()
        for lora in (
            self.text_encoder_loras
            + self.unet_loras
            + self.text_encoder_refts
            + self.unet_refts
        ):
            assert lora.lora_name not in names, (
                f"duplicated lora name: {lora.lora_name}"
            )
            names.add(lora.lora_name)

        # Alias each sigma-aware module's ``_sigma`` / ``_sigma_features``
        # buffer to a single network-level shared tensor. ``set_sigma`` then
        # updates the shared tensor in place once and every aliased module
        # buffer sees the new value through shared storage — instead of
        # ~56 per-module ``copy_`` calls per training step.
        self._wire_shared_sigma_buffers()
        self._wire_shared_fei_buffers()
        self._wire_shared_routing_buffers()
        self._wire_shared_freq_routing_buffers()
        self._wire_shared_content_routing_buffers()

        # Build the network-level GlobalRouter when the cfg selects MoE
        # without per-Linear routers. The input dim is derived from the
        # routing signal: ``"fei"`` → ``fei_feature_dim`` simplex,
        # ``"sigma"`` → ``sigma_feature_dim`` sinusoidal features.
        # Routing-aware modules: ``independent_A`` (StackedExperts) always
        # consume the broadcast gates; ``shared_A`` (Hydra / OrthoHydra)
        # consumes them when built with ``use_global_router=True``.
        self.global_router: Optional[GlobalRouter]
        # ``use_crossattn_router`` advertises to the train / inference call
        # sites that they must fire ``set_crossattn_routing`` with the pooled
        # text tensor each forward (parallel to chimera's ``use_content_router``
        # but broadcasting to the standard ``_routing_weights`` slot).
        self.global_router, self.use_crossattn_router = builders.create_global_router(
            cfg,
            router_class=GlobalRouter,
            crossattn_emb_dim=CROSSATTN_EMB_DIM,
            routing_aware_count=len(self._routing_aware_loras),
            logger=logger,
        )

        # ChimeraHydra FreqRouter: one per network, broadcasts ``π_f`` over
        # the freq pool of every chimera module. Input is
        # ``concat(FEI, sinusoidal-σ-features)`` — owned by the freq router
        # exclusively (the per-layer content router never sees σ/FEI). Built
        # only when at least one chimera module was actually constructed; the
        # router_targets regex can narrow the chimera class to a subset of
        # layers (others fall back to OrthoLoRA).
        self.freq_router: Optional[FreqRouter]

        # ChimeraHydra ContentRouter: network-level twin of FreqRouter for
        # the content pool. Built only when ``content_router_source ==
        # "crossattn_emb"`` AND at least one chimera module exists. Per-Linear
        # ``self.router`` is None on those modules in that case — π_c flows
        # exclusively through the broadcast ``_content_routing_weights``
        # slot. ``use_content_router=True`` advertises to the train /
        # inference call sites that they must thread ``crossattn_emb``
        # through ``set_content`` (no-op otherwise).
        self.content_router: Optional[ContentRouter]
        (
            self.freq_router,
            self.content_router,
            chimera_uses_fei_router,
            self.use_content_router,
        ) = builders.create_chimera_routers(
            cfg,
            freq_router_class=FreqRouter,
            content_router_class=ContentRouter,
            crossattn_emb_dim=CROSSATTN_EMB_DIM,
            chimera_count=len(self._chimera_aware_loras),
            logger=logger,
        )
        if chimera_uses_fei_router:
            # Force the per-step conditioning hook to fire set_fei every step.
            self.use_fei_router = True

        self.register_tokens, self.register_injector, self.extra_seq_tokens = (
            builders.create_register_injector(self, unet, logger=logger)
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
        if getattr(args, "lora_fp32_accumulation", False):
            logger.warning(
                "--lora_fp32_accumulation is deprecated and has no effect; "
                "fp32 accumulation is now unconditional in LoRA/Hydra/ReFT "
                "bottleneck matmuls. Remove the flag from your config."
            )

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier
        for reft in self.text_encoder_refts + self.unet_refts:
            reft.multiplier = self.multiplier

    def set_enabled(self, is_enabled):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.enabled = is_enabled

    def fuse_weights(self):
        return merge_ops.fuse_weights(self)

    def unfuse_weights(self):
        return merge_ops.unfuse_weights(self)

    def set_timestep_mask(self, timesteps: torch.Tensor, max_timestep: float = 1.0):
        return routing_state.set_timestep_mask(self, timesteps, max_timestep)

    def set_step_index(self, step_index: int) -> None:
        """Broadcast a hard denoising-step index to step-expert modules."""
        k = int(step_index)
        for lora in self.text_encoder_loras + self.unet_loras:
            set_step = getattr(lora, "set_step", None)
            if set_step is not None:
                set_step(k)

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
        if apply_text_encoder:
            logger.info(
                f"enable LoRA for text encoder: {len(self.text_encoder_loras)} modules"
            )
        else:
            self.text_encoder_loras = []
            self.text_encoder_refts = []

        if apply_unet:
            logger.info(f"enable LoRA for DiT: {len(self.unet_loras)} modules")
        else:
            self.unet_loras = []
            self.unet_refts = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

        # ReFT wraps each selected DiT Block's forward, so the chain is:
        #   Block.__call__ -> ReFT.forward -> original Block.forward
        #   (inside which LoRA-wrapped Linears still fire normally).
        for reft in self.text_encoder_refts + self.unet_refts:
            reft.apply_to()
            self.add_module(reft.lora_name, reft)

        if apply_unet and self.register_injector is not None:
            self.register_injector.apply(unet)

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
        if getattr(self.cfg, "use_dora", False):
            return 0, 0, 0

        downkeys = []
        upkeys = []
        alphakeys = []
        norms = []
        keys_scaled = 0

        state_dict = self.state_dict()
        for key in state_dict.keys():
            if "lora_down" in key and "weight" in key:
                downkeys.append(key)
                upkeys.append(key.replace("lora_down", "lora_up"))
                alphakeys.append(key.replace("lora_down.weight", "alpha"))

        for i in range(len(downkeys)):
            down = state_dict[downkeys[i]].to(device)
            up = state_dict[upkeys[i]].to(device)
            alpha = state_dict[alphakeys[i]].to(device)
            dim = down.shape[0]
            scale = alpha / dim

            if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
                updown = (
                    (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2))
                    .unsqueeze(2)
                    .unsqueeze(3)
                )
            elif up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3):
                updown = torch.nn.functional.conv2d(
                    down.permute(1, 0, 2, 3), up
                ).permute(1, 0, 2, 3)
            else:
                updown = up @ down

            updown *= scale

            norm = updown.norm().clamp(min=max_norm_value / 2)
            desired = torch.clamp(norm, max=max_norm_value)
            ratio = desired.cpu() / norm.cpu()
            sqrt_ratio = ratio**0.5
            if ratio != 1:
                keys_scaled += 1
                state_dict[upkeys[i]] *= sqrt_ratio
                state_dict[downkeys[i]] *= sqrt_ratio
            scalednorm = updown.norm() * ratio
            norms.append(scalednorm.item())

        return keys_scaled, sum(norms) / len(norms), max(norms)
