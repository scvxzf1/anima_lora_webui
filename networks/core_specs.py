"""Core NetworkSpec registration for built-in LoRA-family variants.

This module owns the built-in lora/dora/ortho/hydra family registrations so
``networks.registry`` can remain importable without eagerly pulling in
``networks.lora_modules``.
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple


def _post_init_hydra(network: Any, kwargs: Mapping[str, Any]) -> None:
    blw = kwargs.get("balance_loss_weight")
    target = float(blw) if blw is not None else 0.01
    warmup = kwargs.get("balance_loss_warmup_ratio")
    warmup_ratio = float(warmup) if warmup is not None else 0.0
    network._balance_loss_target_weight = target
    network._balance_loss_warmup_ratio = warmup_ratio
    network._balance_loss_weight = 0.0 if warmup_ratio > 0.0 else target
    network._use_hydra = True

    cfg_weight = getattr(getattr(network, "cfg", None), "fera_fecl_weight", None)
    if cfg_weight is not None:
        network.fecl_weight = float(cfg_weight)
    else:
        network.fecl_weight = float(kwargs.get("fera_fecl_weight", 0.0) or 0.0)

    cfg = getattr(network, "cfg", None)
    if cfg is not None and getattr(cfg, "use_chimera_hydra", False):
        network._use_chimera_hydra = True
        w_c = cfg.balance_w_content if cfg.balance_w_content is not None else target
        w_f = cfg.balance_w_freq if cfg.balance_w_freq is not None else target
        network._balance_w_content = float(w_c)
        network._balance_w_freq = float(w_f)
    else:
        network._use_chimera_hydra = False


_HYDRA_KWARG_FLAGS: Tuple[str, ...] = (
    "num_experts",
    "balance_loss_weight",
    "balance_loss_warmup_ratio",
    "expert_init_std",
    "ortho_centered_gate",
    "ortho_lambda_init",
    "router_targets",
    "sigma_feature_dim",
    "per_bucket_balance_weight",
    "num_sigma_buckets",
    "specialize_experts_by_sigma_buckets",
    "sigma_bucket_boundaries",
    "fei_feature_dim",
    "fei_sigma_low_div",
)

_CHIMERA_KWARG_FLAGS: Tuple[str, ...] = (
    "use_chimera_hydra",
    "num_experts_content",
    "num_experts_freq",
    "balance_w_content",
    "balance_w_freq",
    "freq_router_init_std",
    "freq_router_layer_norm",
    "network_content_router_lr_scale",
    "network_freq_router_lr_scale",
    "content_router_source",
    "content_router_init_std",
    "content_router_layer_norm",
    "chimera_centered_gate",
    "chimera_lambda_init",
)

_CORE_SPEC_NAMES: Tuple[str, ...] = (
    "lora",
    "dora",
    "ortho",
    "hydra",
    "ortho_hydra",
    "chimera_hydra",
    "stacked_experts_global_fei",
)


def register_core_network_specs() -> None:
    """Register built-in LoRA-family NetworkSpec entries once."""

    from networks.registry import NETWORK_REGISTRY, NetworkSpec, register_network_spec

    if any(name in NETWORK_REGISTRY for name in _CORE_SPEC_NAMES):
        return

    from networks.lora_modules import (
        ChimeraHydraLoRAModule,
        DoRALoRAModule,
        HydraLoRAModule,
        LoRAModule,
        OrthoHydraLoRAModule,
        OrthoLoRAModule,
        StackedExpertsLoRAModule,
    )

    register_network_spec(
        NetworkSpec(name="lora", module_class=LoRAModule, save_variant="standard")
    )
    register_network_spec(
        NetworkSpec(name="dora", module_class=DoRALoRAModule, save_variant="standard")
    )
    register_network_spec(
        NetworkSpec(
            name="ortho",
            module_class=OrthoLoRAModule,
            save_variant="ortho_to_lora",
        )
    )
    register_network_spec(
        NetworkSpec(
            name="hydra",
            module_class=HydraLoRAModule,
            save_variant="hydra_moe",
            kwarg_flags=_HYDRA_KWARG_FLAGS,
            post_init=_post_init_hydra,
        )
    )
    register_network_spec(
        NetworkSpec(
            name="ortho_hydra",
            module_class=OrthoHydraLoRAModule,
            save_variant="ortho_hydra_to_hydra",
            kwarg_flags=_HYDRA_KWARG_FLAGS,
            post_init=_post_init_hydra,
        )
    )
    register_network_spec(
        NetworkSpec(
            name="chimera_hydra",
            module_class=ChimeraHydraLoRAModule,
            save_variant="chimera_hydra_moe",
            kwarg_flags=_HYDRA_KWARG_FLAGS + _CHIMERA_KWARG_FLAGS,
            post_init=_post_init_hydra,
        )
    )
    register_network_spec(
        NetworkSpec(
            name="stacked_experts_global_fei",
            module_class=StackedExpertsLoRAModule,
            save_variant="stacked_experts_global_fei",
            kwarg_flags=_HYDRA_KWARG_FLAGS,
            post_init=_post_init_hydra,
        )
    )


__all__ = [
    "register_core_network_specs",
]
