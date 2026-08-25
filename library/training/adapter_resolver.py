"""Resolve concrete trainer method adapters.

The adapter protocol lives in ``method_adapter``. This module is the only
training-layer place that knows which concrete network modules provide adapter
hooks, which keeps the protocol independent from ``networks.methods``.
"""

from __future__ import annotations

from library.training.method_adapter import MethodAdapter
from library.models.family_registry import get_model_family_spec


def resolve_adapters(args, network) -> list[MethodAdapter]:
    """Sniff ``args`` + ``network`` and return the adapters that apply.

    Imports each adapter lazily so importing the trainer protocol stays cheap.
    """

    adapters: list[MethodAdapter] = []
    if getattr(args, "use_ip_adapter", False):
        from networks.methods.ip_adapter import IPAdapterMethodAdapter

        adapters.append(IPAdapterMethodAdapter())
    if getattr(args, "use_easycontrol", False):
        from networks.methods.easycontrol import EasyControlMethodAdapter

        adapters.append(EasyControlMethodAdapter())
    if getattr(args, "use_byg", False):
        from networks.methods.byg import BYGMethodAdapter

        adapters.append(BYGMethodAdapter())
    # Soft-tokens contrastive: opt-in via a positive contrastive weight on the
    # built network (the objective leaves no learned params, so it's detected
    # off the network's target weight rather than an args flag).
    if float(getattr(network, "_contrastive_target_weight", 0.0) or 0.0) > 0.0:
        from networks.methods.soft_tokens import SoftTokensMethodAdapter

        adapters.append(SoftTokensMethodAdapter())
    if adapters:
        from library.env import resolve_model_family

        family_spec = get_model_family_spec(resolve_model_family(args))
        if not family_spec.supports_method_adapters:
            names = ", ".join(adapter.name for adapter in adapters)
            raise ValueError(
                f"{family_spec.display_name} training does not support method adapters; "
                f"method adapters are unsupported: {names}"
            )
    return adapters
