"""LoRA module construction helpers for the LoRA-family network facade."""

from __future__ import annotations

import logging
from typing import Optional, Pattern, Sequence

import torch

from networks import ModuleCreationContext, NETWORK_REGISTRY
from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_anima.targeting import collect_lora_target_candidates
from networks.lora_modules import (
    ChimeraHydraInferenceModule,
    ChimeraHydraLoRAModule,
    HydraLoRAModule,
    LoRAModule,
    OrthoHydraLoRAModule,
    OrthoLoRAModule,
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
