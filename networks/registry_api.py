"""Leaf registry types and tables for LoRA-family adapters.

This module is intentionally free of plugin/core-spec imports so both
``networks.registry`` and ``networks.core_specs`` can depend on it without
forming a package cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Type

SaveHandler = Callable[[dict[str, Any], str, Any, Optional[dict[str, str]]], bool]


@dataclass(frozen=True)
class ModuleCreationContext:
    """Context handed to ``NetworkSpec.module_kwargs`` for one adapted module."""

    cfg: Any
    is_unet: bool
    lora_name: str
    original_name: str
    child_module: Any
    module_class: Type


@dataclass(frozen=True)
class WeightDetectionContext:
    """Checkpoint sniffing context for plugin specs."""

    key: str
    value: Any
    lora_name: str
    metadata: Mapping[str, str]
    modules_dim: dict[str, int]
    modules_alpha: dict[str, Any]
    plain_module_names: set[str]
    state: dict[str, Any]


@dataclass(frozen=True)
class ContinueWeightDetectionContext:
    """Lightweight WebUI detection context for continue-training weights."""

    keys: list[str]
    metadata: Mapping[str, str]
    lowered_keys: list[str]


@dataclass(frozen=True)
class NetworkSpec:
    """Descriptor for one LoRA-family adapter variant.

    ``module_class`` is the only required implementation object. Plugin hooks
    are optional and let a variant own its selector, constructor extras,
    checkpoint sniffing and save path without putting variant-specific imports
    in the core factory.
    """

    name: str
    module_class: Type
    save_variant: str = "standard"
    kwarg_flags: Tuple[str, ...] = ()
    post_init: Optional[Callable[[Any, Mapping[str, Any]], None]] = None
    selector: Optional[Callable[[Mapping[str, Any]], bool]] = None
    validate: Optional[Callable[[Mapping[str, Any]], None]] = None
    module_kwargs: Optional[Callable[[ModuleCreationContext], dict[str, Any]]] = None
    detect_from_weights: Optional[Callable[[WeightDetectionContext], bool]] = None
    preprocess_weights: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None
    finish_weight_detection: Optional[
        Callable[[dict[str, Any], dict[str, int], dict[str, Any]], dict[str, Any]]
    ] = None
    continue_weight_kind: Optional[
        Callable[[ContinueWeightDetectionContext], Optional[str]]
    ] = None


NETWORK_REGISTRY: Dict[str, NetworkSpec] = {}
SAVE_HANDLERS: dict[str, SaveHandler] = {}


# Kwargs every LoRA-family variant consumes in ``create_network``: core
# targeting knobs + cross-cutting add-ons (ReFT, channel scaling,
# LoRA+, T-LoRA). Variant selectors owned by plugins live in their spec
# ``kwarg_flags`` instead of here.
SHARED_KWARG_FLAGS: Tuple[str, ...] = (
    "dora_wd",
    "train_llm_adapter",
    "exclude_patterns",
    "include_patterns",
    "layer_start",
    "layer_end",
    "rank_dropout",
    "module_dropout",
    "verbose",
    "network_reg_dims",
    "network_reg_lrs",
    "network_reg_alphas",
    "network_router_lr_scale",
    # adaln convenience knobs. Desugar into include_patterns / reg_dims /
    # reg_alphas inside ``LoRANetworkCfg.from_kwargs``; listed here so a
    # top-level TOML key reaches ``net_kwargs`` at all (bootstrap forwards
    # only what this allowlist names).
    "train_adaln",
    "adaln_rank",
    "adaln_alpha",
    "loraplus_lr_ratio",
    "loraplus_unet_lr_ratio",
    "loraplus_text_encoder_lr_ratio",
    "use_timestep_mask",
    "min_rank",
    "alpha_rank_scale",
    "channel_scaling_alpha",
    "lora_fp32_compute",
    "down_init",
    "use_custom_down_autograd",
    "use_ortho",
    "ortho_init_std",
    "use_moe_style",
    "route_per_layer",
    "router_source",
    "router_hidden_dim",
    "router_tau",
    "fera_fecl_weight",
    "fera_num_bands",
    "add_reft",
    "reft_dim",
    "reft_alpha",
    "reft_layers",
    "num_registers",
    "register_insert_block",
    "register_lr_scale",
    "register_init_std",
)


def register_network_spec(spec: NetworkSpec) -> NetworkSpec:
    if spec.name in NETWORK_REGISTRY:
        raise ValueError(f"NetworkSpec {spec.name!r} is already registered")
    NETWORK_REGISTRY[spec.name] = spec
    return spec


def register_save_handler(name: str, handler: SaveHandler) -> SaveHandler:
    if name in SAVE_HANDLERS:
        raise ValueError(f"save handler {name!r} is already registered")
    SAVE_HANDLERS[name] = handler
    return handler


__all__ = [
    "ContinueWeightDetectionContext",
    "ModuleCreationContext",
    "NETWORK_REGISTRY",
    "NetworkSpec",
    "SAVE_HANDLERS",
    "SHARED_KWARG_FLAGS",
    "SaveHandler",
    "WeightDetectionContext",
    "register_network_spec",
    "register_save_handler",
]
