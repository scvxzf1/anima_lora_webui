"""LoRA module construction helpers for the LoRA-family network facade."""

from __future__ import annotations

import logging
from typing import Optional, Pattern, Sequence

import torch

from networks import ModuleCreationContext, NETWORK_REGISTRY
from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_anima.loading import _parse_reft_layers
from networks.lora_anima.targeting import collect_lora_target_candidates
from networks.register_injection import RegisterInjector
from networks.lora_modules import (
    ChimeraHydraInferenceModule,
    ChimeraHydraLoRAModule,
    HydraLoRAModule,
    LoRAModule,
    OrthoHydraLoRAModule,
    OrthoLoRAModule,
    ReFTModule,
    StackedExpertsLoRAModule,
)


def _nominal_spec(module_class):
    return next(
        (
            spec
            for spec in NETWORK_REGISTRY.values()
            if spec.module_class is module_class
        ),
        None,
    )


def _effective_spec(nominal_spec, effective_module_class):
    if (
        nominal_spec is not None
        and effective_module_class is nominal_spec.module_class
    ):
        return nominal_spec
    return next(
        (
            spec
            for spec in NETWORK_REGISTRY.values()
            if spec.module_class is effective_module_class
        ),
        None,
    )


def _resolve_hydra_module_class(network, module_class, lora_name, original_name):
    if network._hydra_router_names is not None:
        hydra_on = lora_name in network._hydra_router_names
    elif network._hydra_router_re is not None:
        hydra_on = bool(network._hydra_router_re.search(original_name))
    else:
        hydra_on = True
    if hydra_on:
        network._hydra_router_hits += 1
        return module_class

    network._hydra_router_misses += 1
    if module_class is HydraLoRAModule:
        return LoRAModule
    if module_class is ChimeraHydraInferenceModule:
        # Load path. Unrouted leg was saved as plain LoRA.
        return LoRAModule
    # Train path (ChimeraHydraLoRAModule) and OrthoHydra: unrouted leg uses
    # the OrthoLoRA Cayley parameterization.
    return OrthoLoRAModule


def _base_extra_kwargs(network, cfg: LoRANetworkCfg, effective_module_class):
    extra_kwargs = {}
    if effective_module_class == OrthoLoRAModule:
        pass
    elif effective_module_class == ChimeraHydraLoRAModule:
        extra_kwargs["num_experts_content"] = cfg.num_experts_content
        extra_kwargs["num_experts_freq"] = cfg.num_experts_freq
        extra_kwargs["centered_gate"] = cfg.chimera_centered_gate
        extra_kwargs["lambda_init"] = cfg.chimera_lambda_init
        if cfg.content_router_source == "crossattn_emb":
            extra_kwargs["use_global_content_router"] = True
    elif effective_module_class == ChimeraHydraInferenceModule:
        extra_kwargs["num_experts_content"] = cfg.num_experts_content
        extra_kwargs["num_experts_freq"] = cfg.num_experts_freq
        extra_kwargs["centered_gate"] = cfg.chimera_centered_gate
        if cfg.content_router_source == "crossattn_emb":
            extra_kwargs["use_global_content_router"] = True
    elif effective_module_class == OrthoHydraLoRAModule:
        extra_kwargs["num_experts"] = cfg.num_experts
        extra_kwargs["centered_gate"] = cfg.ortho_centered_gate
        extra_kwargs["lambda_init"] = cfg.ortho_lambda_init
        if network._use_global_router_for_hydra:
            extra_kwargs["use_global_router"] = True
            network._global_router_hits += 1
    elif effective_module_class == HydraLoRAModule:
        extra_kwargs["num_experts"] = cfg.num_experts
        extra_kwargs["centered_gate"] = cfg.ortho_centered_gate
        if cfg.expert_init_std > 0.0:
            extra_kwargs["expert_init_std"] = cfg.expert_init_std
        if network._use_global_router_for_hydra:
            extra_kwargs["use_global_router"] = True
            network._global_router_hits += 1
        if cfg.use_chimera_hydra:
            # Load path from a distilled chimera checkpoint. HydraLoRAModule
            # narrows its router to K_c and registers _freq_routing_weights.
            extra_kwargs["num_experts_content"] = cfg.num_experts_content
            if cfg.content_router_source == "crossattn_emb":
                extra_kwargs["use_global_content_router"] = True
    elif effective_module_class == StackedExpertsLoRAModule:
        extra_kwargs["num_experts"] = cfg.num_experts
        extra_kwargs["ortho"] = cfg.use_ortho
        if cfg.use_ortho:
            extra_kwargs["ortho_init_std"] = cfg.ortho_init_std
    return extra_kwargs


def _apply_sigma_band_kwargs(cfg, extra_kwargs, effective_module_class, is_unet):
    if (
        cfg.specialize_experts_by_sigma_buckets
        and effective_module_class in (HydraLoRAModule, OrthoHydraLoRAModule)
        and is_unet
    ):
        extra_kwargs["specialize_experts_by_sigma_buckets"] = True
        extra_kwargs["num_sigma_buckets"] = cfg.num_sigma_buckets
        if cfg.sigma_bucket_boundaries is not None:
            extra_kwargs["sigma_bucket_boundaries"] = cfg.sigma_bucket_boundaries


def _apply_sigma_router_kwargs(
    network, cfg, extra_kwargs, effective_module_class, is_unet, lora_name, original_name
):
    if not (
        cfg.router_source == "sigma"
        and effective_module_class in (HydraLoRAModule, OrthoHydraLoRAModule)
        and is_unet
        and not network._use_global_router_for_hydra
    ):
        return
    if network._sigma_router_names is not None:
        enable = lora_name in network._sigma_router_names
    elif network._sigma_router_re is not None:
        enable = bool(network._sigma_router_re.search(original_name))
    else:
        enable = True
    if enable:
        extra_kwargs["sigma_feature_dim"] = cfg.sigma_feature_dim
        network._sigma_router_hits += 1


def _apply_fei_router_kwargs(
    network, cfg, extra_kwargs, effective_module_class, is_unet, lora_name, original_name
):
    if not (
        cfg.router_source == "fei"
        and effective_module_class in (HydraLoRAModule, OrthoHydraLoRAModule)
        and is_unet
        and not network._use_global_router_for_hydra
    ):
        return
    if network._fei_router_names is not None:
        enable_fei = lora_name in network._fei_router_names
    elif network._fei_router_re is not None:
        enable_fei = bool(network._fei_router_re.search(original_name))
    else:
        enable_fei = True
    if enable_fei:
        extra_kwargs["fei_feature_dim"] = cfg.fei_feature_dim
        network._fei_router_hits += 1


def _apply_channel_scale_kwargs(
    network, cfg, extra_kwargs, effective_spec, is_unet, lora_name
):
    if not (
        cfg.channel_scales_dict is not None
        and is_unet
        and not (effective_spec is not None and effective_spec.name == "lokr")
    ):
        return
    channel_scale = cfg.channel_scales_dict.get(lora_name)
    if channel_scale is not None:
        extra_kwargs["channel_scale"] = channel_scale
        network._channel_scale_hits += 1
    else:
        network._channel_scale_misses.append(lora_name)


def create_lora_modules(
    network,
    *,
    is_unet: bool,
    text_encoder_idx: Optional[int],
    root_module: torch.nn.Module,
    target_replace_modules: Sequence[str],
    exclude_patterns: Sequence[Pattern[str]],
    include_patterns: Sequence[Pattern[str]],
    logger: logging.Logger,
    default_dim: Optional[int] = None,
):
    cfg = network.cfg
    module_class = cfg.module_class
    nominal_spec = _nominal_spec(module_class)

    prefix = network.LORA_PREFIX_ANIMA if is_unet else network.LORA_PREFIX_TEXT_ENCODER
    candidates = collect_lora_target_candidates(
        root_module=root_module,
        prefix=prefix,
        target_replace_modules=target_replace_modules,
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        is_unet=is_unet,
        layer_start=cfg.layer_start,
        layer_end=cfg.layer_end,
        modules_dim=cfg.modules_dim,
        modules_alpha=cfg.modules_alpha,
        reg_dims=cfg.reg_dims,
        default_dim=default_dim,
        lora_dim=cfg.lora_dim,
        alpha=cfg.alpha,
        verbose=cfg.verbose,
        logger=logger,
    )

    from tqdm import tqdm

    loras = []
    non_skipped = [
        (item.lora_name, item.child_module, item.dim, item.alpha, item.original_name)
        for item in candidates
        if not item.skipped
    ]
    skipped = [item.lora_name for item in candidates if item.skipped]

    if nominal_spec is not None and nominal_spec.name == "vera":
        linear_candidates = [
            child_module
            for _lora_name, child_module, _dim, _alpha, _original_name in non_skipped
            if isinstance(child_module, torch.nn.Linear)
        ]
        if linear_candidates:
            cfg.plugin_args["_vera_max_in_features"] = max(
                int(module.in_features) for module in linear_candidates
            )
            cfg.plugin_args["_vera_max_out_features"] = max(
                int(module.out_features) for module in linear_candidates
            )

    label = (
        "DiT"
        if is_unet
        else f"TE{text_encoder_idx + 1}"
        if text_encoder_idx is not None
        else "model"
    )
    for lora_name, child_module, dim, alpha_val, original_name in tqdm(
        non_skipped, desc=f"Creating {label} LoRA", leave=False
    ):
        effective_module_class = module_class
        if (
            module_class
            in (
                HydraLoRAModule,
                OrthoHydraLoRAModule,
                ChimeraHydraLoRAModule,
                ChimeraHydraInferenceModule,
            )
            and is_unet
        ):
            effective_module_class = _resolve_hydra_module_class(
                network, module_class, lora_name, original_name
            )

        extra_kwargs = _base_extra_kwargs(network, cfg, effective_module_class)
        if cfg.down_init != "kaiming" and effective_module_class is LoRAModule:
            extra_kwargs["down_init"] = cfg.down_init

        effective_spec = _effective_spec(nominal_spec, effective_module_class)
        if effective_spec is not None and effective_spec.module_kwargs is not None:
            extra_kwargs.update(
                effective_spec.module_kwargs(
                    ModuleCreationContext(
                        cfg=cfg,
                        is_unet=is_unet,
                        lora_name=lora_name,
                        original_name=original_name,
                        child_module=child_module,
                        module_class=effective_module_class,
                    )
                )
            )

        _apply_sigma_band_kwargs(cfg, extra_kwargs, effective_module_class, is_unet)
        _apply_sigma_router_kwargs(
            network,
            cfg,
            extra_kwargs,
            effective_module_class,
            is_unet,
            lora_name,
            original_name,
        )
        _apply_fei_router_kwargs(
            network,
            cfg,
            extra_kwargs,
            effective_module_class,
            is_unet,
            lora_name,
            original_name,
        )
        _apply_channel_scale_kwargs(
            network, cfg, extra_kwargs, effective_spec, is_unet, lora_name
        )

        lora = effective_module_class(
            lora_name,
            child_module,
            network.multiplier,
            dim,
            alpha_val,
            dropout=cfg.dropout,
            rank_dropout=cfg.rank_dropout,
            module_dropout=cfg.module_dropout,
            **extra_kwargs,
        )
        lora.fp32_compute = bool(cfg.lora_fp32_compute)
        lora.original_name = original_name
        loras.append(lora)

    return loras, skipped


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
