"""Network registry compatibility facade.

The implementation lives in :mod:`networks.registry`; this package-level
module keeps existing imports such as ``from networks import NETWORK_REGISTRY``
working while bundled adapter variants register through the registry API.
"""

from .registry import (
    ContinueWeightDetectionContext,
    ModuleCreationContext,
    NETWORK_REGISTRY,
    NetworkSpec,
    SAVE_HANDLERS,
    SHARED_KWARG_FLAGS,
    WeightDetectionContext,
    all_network_kwargs,
    continue_weight_kind_from_plugins,
    detect_network_spec_from_weights,
    ensure_builtin_plugins_loaded,
    register_network_spec,
    register_save_handler,
    resolve_network_spec,
)

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
    "register_network_spec",
    "register_save_handler",
    "resolve_network_spec",
]
