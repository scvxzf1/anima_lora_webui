"""LoRA module construction helpers for the LoRA-family network facade."""

from __future__ import annotations

import logging
import re
from typing import Pattern, Sequence

import torch

from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_anima.loading import _parse_reft_layers
from networks.lora_anima.module_builders import create_lora_modules
from networks.lora_anima.targeting import compile_lora_target_patterns
from networks.register_injection import RegisterInjector
from networks.lora_modules import (
    ChimeraHydraInferenceModule,
    ChimeraHydraLoRAModule,
    OrthoHydraLoRAModule,
    OrthoLoRAModule,
    ReFTModule,
)


def _initialize_router_scope(network, cfg: LoRANetworkCfg) -> None:
    # Unified routing scope. ``cfg.router_targets`` is the single regex
    # that governs which Linears participate in routed adaptation (Hydra
    # MoE leaves + sigma-feature concat + FEI-feature concat all share it).
    # From-weights path supplies an explicit name set per router family
    # (different families may have different module memberships in older
    # checkpoints); when present, the explicit set wins over the regex.
    router_re = re.compile(cfg.router_targets) if cfg.router_targets else None

    network._sigma_router_names = (
        set(cfg.sigma_router_names) if cfg.sigma_router_names else None
    )
    network._sigma_router_re = (
        router_re
        if (
            cfg.router_source == "sigma"
            and router_re is not None
            and network._sigma_router_names is None
        )
        else None
    )

    network._fei_router_names = (
        set(cfg.fei_router_names) if cfg.fei_router_names else None
    )
    network._fei_router_re = (
        router_re
        if (
            cfg.router_source == "fei"
            and router_re is not None
            and network._fei_router_names is None
        )
        else None
    )
    network._fei_router_hits = 0
    # Modules built with ``use_global_router=True`` (shared_A +
    # ``route_per_layer=False``): the per-layer router is skipped and gates
    # arrive via the network-level ``GlobalRouter``. Counted separately
    # from ``_fei_router_hits`` because the per-layer FEI cat is bypassed.
    network._global_router_hits = 0
    # Retained as a network attr (library/inference/adapters.py reads it
    # via getattr); derived from cfg.router_source.
    network.use_fei_router = cfg.router_source == "fei"
    network.use_sigma_router = cfg.router_source == "sigma"
    # Shared-A Hydra layout + network-level router (FEI-on-Hydra global).
    # Toggle for the per-module construction loop below; lets Hydra /
    # OrthoHydra modules skip ``self.router`` and consume gates from the
    # ``GlobalRouter`` instead. Mirrors the FeRA (independent_A) routing
    # location without changing the underlying Hydra parameter layout.
    network._use_global_router_for_hydra = (
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
    network._hydra_router_names = (
        set(cfg.hydra_router_names) if cfg.hydra_router_names else None
    )
    network._hydra_router_re = (
        router_re if (router_re is not None and network._hydra_router_names is None)
        else None
    )


def _log_network_build_start(cfg: LoRANetworkCfg, logger: logging.Logger) -> None:
    if cfg.modules_dim is not None:
        logger.info("create LoRA network from weights")
        return
    logger.info(
        f"create LoRA network. base dim (rank): {cfg.lora_dim}, alpha: {cfg.alpha}"
    )
    logger.info(
        f"neuron dropout: p={cfg.dropout}, rank dropout: p={cfg.rank_dropout}, "
        f"module dropout: p={cfg.module_dropout}"
    )


def _compile_target_pattern_sets(cfg: LoRANetworkCfg, logger: logging.Logger):
    return (
        compile_lora_target_patterns(cfg.exclude_patterns, logger=logger),
        compile_lora_target_patterns(cfg.include_patterns, logger=logger),
    )


def _create_text_encoder_loras(
    network,
    text_encoders: list,
    *,
    cfg: LoRANetworkCfg,
    target_replace_modules: Sequence[str],
    exclude_patterns: Sequence[Pattern[str]],
    include_patterns: Sequence[Pattern[str]],
    logger: logging.Logger,
):
    network.text_encoder_loras = []
    skipped = []
    if text_encoders is None or cfg.module_class in (
        OrthoLoRAModule,
        OrthoHydraLoRAModule,
        ChimeraHydraLoRAModule,
        ChimeraHydraInferenceModule,
    ):
        return skipped

    # Qwen3 text encoders are usually not trained for Anima; Ortho variants
    # skip this path because SVD init is expensive and apply_to drops TE loras.
    for i, text_encoder in enumerate(text_encoders):
        if text_encoder is None:
            continue
        logger.info(f"create LoRA for Text Encoder {i + 1}:")
        te_loras, te_skipped = create_lora_modules(
            network,
            is_unet=False,
            text_encoder_idx=i,
            root_module=text_encoder,
            target_replace_modules=target_replace_modules,
            exclude_patterns=exclude_patterns,
            include_patterns=include_patterns,
            logger=logger,
        )
        logger.info(f"create LoRA for Text Encoder {i + 1}: {len(te_loras)} modules.")
        network.text_encoder_loras.extend(te_loras)
        skipped += te_skipped
    return skipped


def _create_unet_loras(
    network,
    unet,
    *,
    cfg: LoRANetworkCfg,
    unet_target_replace_modules: Sequence[str],
    adapter_target_replace_modules: Sequence[str],
    exclude_patterns: Sequence[Pattern[str]],
    include_patterns: Sequence[Pattern[str]],
    logger: logging.Logger,
):
    target_modules = list(unet_target_replace_modules)
    if cfg.train_llm_adapter:
        target_modules.extend(adapter_target_replace_modules)

    network.unet_loras, skipped = create_lora_modules(
        network,
        is_unet=True,
        text_encoder_idx=None,
        root_module=unet,
        target_replace_modules=target_modules,
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        logger=logger,
    )

    logger.info(f"create LoRA for Anima DiT: {len(network.unet_loras)} modules.")
    if cfg.verbose:
        for lora in network.unet_loras:
            logger.info(f"\t{lora.lora_name:60} {lora.lora_dim}, {lora.alpha}")
    return skipped


def _create_base_lora_modules(
    network,
    text_encoders: list,
    unet,
    *,
    cfg: LoRANetworkCfg,
    unet_target_replace_modules: Sequence[str],
    adapter_target_replace_modules: Sequence[str],
    text_encoder_target_replace_modules: Sequence[str],
    exclude_patterns: Sequence[Pattern[str]],
    include_patterns: Sequence[Pattern[str]],
    logger: logging.Logger,
):
    skipped_te = _create_text_encoder_loras(
        network,
        text_encoders,
        cfg=cfg,
        target_replace_modules=text_encoder_target_replace_modules,
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        logger=logger,
    )
    skipped_un = _create_unet_loras(
        network,
        unet,
        cfg=cfg,
        unet_target_replace_modules=unet_target_replace_modules,
        adapter_target_replace_modules=adapter_target_replace_modules,
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        logger=logger,
    )
    return skipped_te + skipped_un


def _log_skipped_lora_modules(
    skipped: Sequence[str], *, verbose: bool, logger: logging.Logger
) -> None:
    if not verbose or len(skipped) == 0:
        return
    logger.warning(f"dim (rank) is 0, {len(skipped)} LoRA modules are skipped:")
    for name in skipped:
        logger.info(f"\t{name}")


def _log_channel_scale_summary(
    network, cfg: LoRANetworkCfg, logger: logging.Logger
) -> None:
    if cfg.channel_scales_dict is None:
        return
    logger.info(
        f"channel_scaling: {network._channel_scale_hits} DiT modules "
        f"received calibration-based input scaling"
    )
    if network._channel_scale_misses:
        logger.warning(
            f"channel_scaling: {len(network._channel_scale_misses)} DiT modules "
            f"have no calibration stats (first: {network._channel_scale_misses[:3]}). "
            f"These will train without input rebalancing -- regenerate the vendored "
            f"calibration with `python bench/channel_stats/analyze_lora_input_channels.py "
            f"--per_artist --dump_channel_stats networks/calibration/channel_stats.safetensors` "
            f"if this is unexpected."
        )


def _assert_unique_adapter_names(network) -> None:
    names = set()
    for lora in (
        network.text_encoder_loras
        + network.unet_loras
        + network.text_encoder_refts
        + network.unet_refts
    ):
        assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
        names.add(lora.lora_name)


def _wire_shared_router_buffers(network) -> None:
    network._wire_shared_sigma_buffers()
    network._wire_shared_fei_buffers()
    network._wire_shared_routing_buffers()
    network._wire_shared_freq_routing_buffers()
    network._wire_shared_content_routing_buffers()


def _create_network_router_modules(
    network,
    cfg: LoRANetworkCfg,
    *,
    router_class,
    freq_router_class,
    content_router_class,
    crossattn_emb_dim: int,
    logger: logging.Logger,
) -> None:
    network.global_router, network.use_crossattn_router = create_global_router(
        cfg,
        router_class=router_class,
        crossattn_emb_dim=crossattn_emb_dim,
        routing_aware_count=len(network._routing_aware_loras),
        logger=logger,
    )

    (
        network.freq_router,
        network.content_router,
        chimera_uses_fei_router,
        network.use_content_router,
    ) = create_chimera_routers(
        cfg,
        freq_router_class=freq_router_class,
        content_router_class=content_router_class,
        crossattn_emb_dim=crossattn_emb_dim,
        chimera_count=len(network._chimera_aware_loras),
        logger=logger,
    )
    if chimera_uses_fei_router:
        # Force the per-step conditioning hook to fire set_fei every step.
        network.use_fei_router = True


def initialize_network_components(
    network,
    text_encoders: list,
    unet,
    *,
    cfg: LoRANetworkCfg,
    multiplier: float,
    unet_target_replace_modules: Sequence[str],
    adapter_target_replace_modules: Sequence[str],
    text_encoder_target_replace_modules: Sequence[str],
    router_class,
    freq_router_class,
    content_router_class,
    crossattn_emb_dim: int,
    logger: logging.Logger,
) -> None:
    _initialize_router_scope(network, cfg)
    _log_network_build_start(cfg, logger)
    exclude_re_patterns, include_re_patterns = _compile_target_pattern_sets(
        cfg, logger
    )

    skipped = _create_base_lora_modules(
        network,
        text_encoders,
        unet,
        cfg=cfg,
        unet_target_replace_modules=unet_target_replace_modules,
        adapter_target_replace_modules=adapter_target_replace_modules,
        text_encoder_target_replace_modules=text_encoder_target_replace_modules,
        exclude_patterns=exclude_re_patterns,
        include_patterns=include_re_patterns,
        logger=logger,
    )
    _log_skipped_lora_modules(skipped, verbose=cfg.verbose, logger=logger)
    _log_channel_scale_summary(network, cfg, logger)

    # Create ReFT modules on the DiT residual stream (block outputs), following
    # Wu et al. (2024) §3.3 -- one intervention per selected block, not per
    # internal Linear. Selection is controlled by ``reft_layers``.
    network.text_encoder_refts, network.unet_refts = create_reft_modules(
        unet,
        cfg=cfg,
        multiplier=multiplier,
        logger=logger,
    )

    _assert_unique_adapter_names(network)
    _wire_shared_router_buffers(network)
    _create_network_router_modules(
        network,
        cfg,
        router_class=router_class,
        freq_router_class=freq_router_class,
        content_router_class=content_router_class,
        crossattn_emb_dim=crossattn_emb_dim,
        logger=logger,
    )

    network.register_tokens, network.register_injector, network.extra_seq_tokens = (
        create_register_injector(network, unet, logger=logger)
    )


def create_reft_modules(
    unet,
    *,
    cfg: LoRANetworkCfg,
    multiplier: float,
    logger: logging.Logger,
):
    unet_refts = []
    if not cfg.add_reft:
        return [], unet_refts

    dit_blocks = getattr(unet, "blocks", None)
    if dit_blocks is None or len(dit_blocks) == 0:
        raise ValueError(
            "add_reft=True but DiT has no .blocks attribute to wrap. "
            "Block-level ReFT requires a transformer with a `blocks` ModuleList."
        )
    num_blocks = len(dit_blocks)
    selected_indices = _parse_reft_layers(cfg.reft_layers, num_blocks)

    reft_alpha_value = cfg.reft_alpha if cfg.reft_alpha is not None else cfg.alpha
    for idx in selected_indices:
        block = dit_blocks[idx]
        block_embed_dim = getattr(block, "x_dim", None)
        if block_embed_dim is None:
            raise ValueError(
                f"Block {idx} ({type(block).__name__}) has no `x_dim`; "
                "cannot infer embed_dim for ReFT."
            )
        reft_name = f"reft_unet_blocks_{idx}"
        reft = ReFTModule(
            reft_name,
            block,
            embed_dim=block_embed_dim,
            multiplier=multiplier,
            reft_dim=cfg.reft_dim,
            alpha=reft_alpha_value,
            dropout=cfg.dropout,
            module_dropout=cfg.module_dropout,
        )
        reft.original_name = f"blocks.{idx}"
        unet_refts.append(reft)
    logger.info(
        f"create ReFT for Anima DiT: {len(unet_refts)}/{num_blocks} "
        f"blocks (reft_dim={cfg.reft_dim}, layers={cfg.reft_layers!r})"
    )
    return [], unet_refts


def create_global_router(
    cfg: LoRANetworkCfg,
    *,
    router_class,
    crossattn_emb_dim: int,
    routing_aware_count: int,
    logger: logging.Logger,
):
    if cfg.use_moe_style is False or cfg.route_per_layer:
        return None, False

    router_layer_norm = False
    if cfg.router_source == "fei":
        router_input_dim = int(cfg.fei_feature_dim)
    elif cfg.router_source == "sigma":
        router_input_dim = int(cfg.sigma_feature_dim)
    elif cfg.router_source == "crossattn_emb":
        router_input_dim = int(crossattn_emb_dim)
        router_layer_norm = True
    else:
        router_input_dim = 0
    if router_input_dim <= 0 or cfg.num_experts <= 1:
        return None, False

    global_router = router_class(
        input_dim=router_input_dim,
        num_experts=int(cfg.num_experts),
        hidden_dim=int(cfg.router_hidden_dim),
        tau=float(cfg.router_tau),
        apply_layer_norm=router_layer_norm,
    )
    use_crossattn_router = cfg.router_source == "crossattn_emb"
    logger.info(
        f"GlobalRouter: source={cfg.router_source!r}, "
        f"input_dim={router_input_dim}, "
        f"num_experts={cfg.num_experts}, "
        f"hidden={cfg.router_hidden_dim}, tau={cfg.router_tau:.2f}, "
        f"LN={router_layer_norm}, "
        f"routing-aware modules={routing_aware_count}"
    )
    return global_router, use_crossattn_router


def create_chimera_routers(
    cfg: LoRANetworkCfg,
    *,
    freq_router_class,
    content_router_class,
    crossattn_emb_dim: int,
    chimera_count: int,
    logger: logging.Logger,
):
    freq_router = None
    content_router = None
    use_fei_router = False
    use_content_router = False

    if cfg.use_chimera_hydra and chimera_count:
        freq_input_dim = int(cfg.fei_feature_dim) + int(cfg.sigma_feature_dim)
        if freq_input_dim <= 0:
            raise ValueError(
                "use_chimera_hydra=True requires fei_feature_dim + "
                f"sigma_feature_dim > 0 for the FreqRouter input (got "
                f"FEI={cfg.fei_feature_dim}, sigma={cfg.sigma_feature_dim})."
            )
        freq_router = freq_router_class(
            input_dim=freq_input_dim,
            num_freq_experts=int(cfg.num_experts_freq),
            hidden_dim=int(cfg.router_hidden_dim),
            tau=float(cfg.router_tau),
            init_std=float(cfg.freq_router_init_std),
            fei_dim=int(cfg.fei_feature_dim),
            sigma_dim=int(cfg.sigma_feature_dim),
            apply_layer_norm=bool(cfg.freq_router_layer_norm),
        )
        use_fei_router = True
        logger.info(
            f"ChimeraHydra FreqRouter: input_dim={freq_input_dim} "
            f"(FEI={cfg.fei_feature_dim} + sigma={cfg.sigma_feature_dim}), "
            f"K_f={cfg.num_experts_freq}, hidden={cfg.router_hidden_dim}, "
            f"tau={cfg.router_tau:.2f}, init_std={cfg.freq_router_init_std}, "
            f"LN={freq_router.apply_layer_norm}, "
            f"chimera modules={chimera_count}"
        )

    if (
        cfg.use_chimera_hydra
        and cfg.content_router_source == "crossattn_emb"
        and chimera_count
    ):
        content_router = content_router_class(
            input_dim=int(crossattn_emb_dim),
            num_content_experts=int(cfg.num_experts_content),
            hidden_dim=int(cfg.router_hidden_dim),
            tau=float(cfg.router_tau),
            init_std=float(cfg.content_router_init_std),
            apply_layer_norm=bool(cfg.content_router_layer_norm),
        )
        use_content_router = True
        logger.info(
            f"ChimeraHydra ContentRouter: input_dim={crossattn_emb_dim} "
            f"(pooled crossattn_emb), K_c={cfg.num_experts_content}, "
            f"hidden={cfg.router_hidden_dim}, tau={cfg.router_tau:.2f}, "
            f"init_std={cfg.content_router_init_std}, "
            f"LN={cfg.content_router_layer_norm}, "
            f"chimera modules={chimera_count} "
            "-- per-Linear content router disabled"
        )

    return freq_router, content_router, use_fei_router, use_content_router


def create_register_injector(network, unet, *, logger: logging.Logger):
    cfg = network.cfg
    extra_seq_tokens = int(cfg.num_registers)
    if cfg.num_registers <= 0:
        return None, None, extra_seq_tokens

    n_blocks = len(unet.blocks)
    if not (0 <= cfg.register_insert_block < n_blocks):
        raise ValueError(
            f"register_insert_block must be in [0, {n_blocks}), "
            f"got {cfg.register_insert_block}"
        )
    register_tokens = torch.nn.Parameter(
        torch.randn(cfg.num_registers, int(unet.model_channels))
        * cfg.register_init_std
    )
    register_injector = RegisterInjector(
        num_registers=cfg.num_registers,
        insert_block=cfg.register_insert_block,
        get_scaled_tokens=lambda: register_tokens * network.multiplier,
    )
    logger.info(
        f"Register tokens: K={cfg.num_registers}, "
        f"insert_block={cfg.register_insert_block}, "
        f"lr scale x{cfg.register_lr_scale:g}, "
        f"init_std={cfg.register_init_std:g}. "
        "Checkpoint stays kept-live at inference."
    )
    return register_tokens, register_injector, extra_seq_tokens
