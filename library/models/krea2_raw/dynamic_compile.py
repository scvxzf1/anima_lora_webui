"""Cache compiled callables while the resident prefix changes between updates."""

from __future__ import annotations

import torch


def refresh_resident_compile(model) -> None:
    options = getattr(model, "_krea_compile_options", None)
    if options is None:
        return
    resident = len(model.blocks) - model.blocks_to_swap
    for index, block in enumerate(model.blocks):
        base = getattr(block, "_krea_compile_base_forward", block._forward)
        if index >= resident:
            block._forward = base
            continue
        block._krea_compile_base_forward = base
        compiled = getattr(block, "_krea_dynamic_compiled_forward", None)
        if compiled is None:
            compiled = torch.compile(base, **options)
            block._krea_dynamic_compiled_forward = compiled
        block._forward = compiled
