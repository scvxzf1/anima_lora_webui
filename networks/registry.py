"""NetworkSpec registry for LoRA-family adapter dispatch.

Core code owns the registry and built-in LoRA-family specs. Optional variants
register themselves as plugins, so the factory/network/save path depends on
the ``NetworkSpec`` contract instead of concrete variant modules.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import logging
import pkgutil
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Type

logger = logging.getLogger(__name__)

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
    "network_router_lr_scale",
    "loraplus_lr_ratio",
    "loraplus_unet_lr_ratio",
    "loraplus_text_encoder_lr_ratio",
    "use_timestep_mask",
    "min_rank",
    "alpha_rank_scale",
    "channel_scaling_alpha",
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


def all_network_kwargs() -> Tuple[str, ...]:
    """Return the union of shared + per-variant kwargs, sorted."""

    ensure_builtin_plugins_loaded()
    merged: set[str] = set(SHARED_KWARG_FLAGS)
    for spec in NETWORK_REGISTRY.values():
        merged.update(spec.kwarg_flags)
    return tuple(sorted(merged))


def _parse_bool_flag(kwargs: Mapping[str, Any], key: str) -> bool:
    v = kwargs.get(key, False)
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).lower() == "true"


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


def _register_core_specs() -> None:
    if NETWORK_REGISTRY:
        return

    from .lora_modules import (
        ChimeraHydraLoRAModule,
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


_PLUGINS_LOADED = False


def ensure_builtin_plugins_loaded() -> None:
    """Import bundled plugin packages so they can register their specs."""

    global _PLUGINS_LOADED
    _register_core_specs()
    if _PLUGINS_LOADED:
        return
    _PLUGINS_LOADED = True
    try:
        plugins_pkg = importlib.import_module(f"{__package__}.plugins")
    except ModuleNotFoundError:
        return

    for module_info in pkgutil.iter_modules(
        plugins_pkg.__path__, plugins_pkg.__name__ + "."
    ):
        if module_info.name.rsplit(".", 1)[-1].startswith("_"):
            continue
        try:
            importlib.import_module(module_info.name)
        except Exception:
            logger.exception("Failed to import network plugin %s", module_info.name)
            raise


def resolve_network_spec(kwargs: Mapping[str, Any]) -> NetworkSpec:
    """Resolve which NetworkSpec to instantiate from create_network kwargs."""

    ensure_builtin_plugins_loaded()

    use_ortho = _parse_bool_flag(kwargs, "use_ortho")
    use_chimera = _parse_bool_flag(kwargs, "use_chimera_hydra")

    for spec in NETWORK_REGISTRY.values():
        if spec.name in {
            "lora",
            "ortho",
            "hydra",
            "ortho_hydra",
            "chimera_hydra",
            "stacked_experts_global_fei",
        }:
            continue
        if spec.validate is not None:
            spec.validate(kwargs)
        if spec.selector is not None and spec.selector(kwargs):
            return spec

    if use_chimera:
        return NETWORK_REGISTRY["chimera_hydra"]

    raw_moe = kwargs.get("use_moe_style")
    if isinstance(raw_moe, str):
        moe_style = raw_moe.strip()
        if moe_style.lower() in ("false", "none", ""):
            moe_style = ""
    elif raw_moe is False or raw_moe is None:
        moe_style = ""
    else:
        raise ValueError(
            f"use_moe_style={raw_moe!r}: expected False, 'shared_A', or 'independent_A'."
        )
    if moe_style not in ("", "shared_A", "independent_A"):
        raise ValueError(
            f"use_moe_style={raw_moe!r}: expected False, 'shared_A', or 'independent_A'."
        )

    if moe_style == "independent_A":
        return NETWORK_REGISTRY["stacked_experts_global_fei"]
    if moe_style == "shared_A":
        return (
            NETWORK_REGISTRY["ortho_hydra"] if use_ortho else NETWORK_REGISTRY["hydra"]
        )
    if use_ortho:
        return NETWORK_REGISTRY["ortho"]
    return NETWORK_REGISTRY["lora"]


def detect_network_spec_from_weights(
    weights_sd: Mapping[str, Any],
    metadata: Mapping[str, str],
    modules_dim: dict[str, int],
    modules_alpha: dict[str, Any],
    plain_module_names: set[str],
) -> dict[str, Any]:
    """Run plugin checkpoint sniffers and return their accumulated state."""

    ensure_builtin_plugins_loaded()
    state: dict[str, Any] = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue
        lora_name = key.split(".")[0]
        ctx = WeightDetectionContext(
            key=key,
            value=value,
            lora_name=lora_name,
            metadata=metadata,
            modules_dim=modules_dim,
            modules_alpha=modules_alpha,
            plain_module_names=plain_module_names,
            state=state,
        )
        for spec in NETWORK_REGISTRY.values():
            if spec.detect_from_weights is not None:
                spec.detect_from_weights(ctx)
    for spec in NETWORK_REGISTRY.values():
        if spec.finish_weight_detection is not None:
            state.update(
                spec.finish_weight_detection(state, modules_dim, modules_alpha)
            )
    return state


def preprocess_weights_from_plugins(weights_sd: dict[str, Any]) -> dict[str, Any]:
    """Let plugin variants rewrite checkpoint keys before core sniffing."""

    ensure_builtin_plugins_loaded()
    for spec in NETWORK_REGISTRY.values():
        if spec.preprocess_weights is not None:
            weights_sd = spec.preprocess_weights(weights_sd)
    return weights_sd


def continue_weight_kind_from_plugins(
    keys: list[str],
    metadata: Mapping[str, str],
) -> Optional[str]:
    ensure_builtin_plugins_loaded()
    lowered_keys = [str(key).lower() for key in keys]
    ctx = ContinueWeightDetectionContext(
        keys=keys,
        metadata=metadata,
        lowered_keys=lowered_keys,
    )
    for spec in NETWORK_REGISTRY.values():
        if spec.continue_weight_kind is None:
            continue
        kind = spec.continue_weight_kind(ctx)
        if kind:
            return kind
    return None


ensure_builtin_plugins_loaded()


__all__ = [
    "ContinueWeightDetectionContext",
    "ModuleCreationContext",
    "NETWORK_REGISTRY",
    "NetworkSpec",
    "SAVE_HANDLERS",
    "SHARED_KWARG_FLAGS",
    "WeightDetectionContext",
    "all_network_kwargs",
    "continue_weight_kind_from_plugins",
    "detect_network_spec_from_weights",
    "ensure_builtin_plugins_loaded",
    "preprocess_weights_from_plugins",
    "register_network_spec",
    "register_save_handler",
    "resolve_network_spec",
]
