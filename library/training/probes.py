"""Training memory/peak probe helpers used during setup."""

from __future__ import annotations

import torch


def maybe_probe(trainer, label: str, **kwargs) -> None:
    probe = getattr(trainer, "memory_probe", None)
    if probe is None:
        return
    probe.snapshot(label, **kwargs)


def maybe_probe_components(trainer, label: str, **kwargs) -> None:
    probe = getattr(trainer, "memory_probe", None)
    if probe is None:
        return
    probe.component_summary(label, **kwargs)


def attach_peak_probe_to_network(network, probe) -> int:
    if probe is None or network is None:
        return 0
    count = 0
    modules = []
    for attr in ("text_encoder_loras", "unet_loras"):
        modules.extend(list(getattr(network, attr, []) or []))
    for module in modules:
        if hasattr(module, "_peak_probe"):
            module._peak_probe = probe
            count += 1
    return count
