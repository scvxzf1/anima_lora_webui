"""Bundled LoKR plugin registration."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from networks.plugins.lokr.module import LoKrModule
from networks.plugins.lokr.save import save_lokr_weights
from networks.registry import (
    ContinueWeightDetectionContext,
    ModuleCreationContext,
    NetworkSpec,
    WeightDetectionContext,
    register_network_spec,
    register_save_handler,
)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _selector(kwargs: Mapping[str, Any]) -> bool:
    return _truthy(kwargs.get("use_lokr"))


def _validate(kwargs: Mapping[str, Any]) -> None:
    if not _selector(kwargs):
        return
    if (
        _truthy(kwargs.get("use_ortho"))
        or _truthy(kwargs.get("use_chimera_hydra"))
        or kwargs.get("use_moe_style") not in (None, False, "", "false", "False")
    ):
        raise ValueError(
            "use_lokr is mutually exclusive with use_ortho, "
            "use_moe_style, and use_chimera_hydra."
        )


def _module_kwargs(ctx: ModuleCreationContext) -> dict[str, Any]:
    return {"factor": int(ctx.cfg.plugin_args.get("lokr_factor", 8))}


def _detect_from_weights(ctx: WeightDetectionContext) -> bool:
    key = ctx.key
    value = ctx.value
    try:
        network_dim = int(float(ctx.metadata.get("ss_network_dim", "")))
    except (TypeError, ValueError):
        network_dim = None
    ctx.state["lokr_network_dim_meta"] = network_dim
    if key.endswith(".lokr_w1"):
        ctx.state["has_lokr"] = True
        ctx.state.setdefault("lokr_module_names", set()).add(ctx.lora_name)
        if value.dim() == 2:
            ctx.state.setdefault("lokr_factors", set()).add(int(value.size(0)))
        ctx.modules_dim[ctx.lora_name] = network_dim or (
            value.size(0) if value.dim() >= 1 else 1
        )
        ctx.modules_alpha.setdefault(
            ctx.lora_name, torch.tensor(float(ctx.modules_dim[ctx.lora_name]))
        )
        ctx.plain_module_names.add(ctx.lora_name)
        return True
    if key.endswith(".lokr_w2"):
        ctx.state["has_lokr"] = True
        ctx.state.setdefault("lokr_module_names", set()).add(ctx.lora_name)
        ctx.modules_dim.setdefault(
            ctx.lora_name, ctx.state.get("lokr_network_dim_meta") or 1
        )
        ctx.modules_alpha.setdefault(
            ctx.lora_name, torch.tensor(float(ctx.modules_dim[ctx.lora_name]))
        )
        ctx.plain_module_names.add(ctx.lora_name)
        return True
    return False


def _finish_weight_detection(
    state: dict[str, Any],
    modules_dim: dict[str, int],
    modules_alpha: dict[str, Any],
) -> dict[str, Any]:
    if not state.get("has_lokr"):
        return {}
    if state.get("lokr_network_dim_meta") is None:
        for lora_name in state.get("lokr_module_names", set()):
            alpha_value = modules_alpha.get(lora_name)
            if isinstance(alpha_value, torch.Tensor) and alpha_value.numel() == 1:
                modules_dim[lora_name] = max(
                    1,
                    int(float(alpha_value.detach().float().cpu().item())),
                )
    factor = next(iter(sorted(state.get("lokr_factors", set()))), 8)
    return {"detected_spec": "lokr", "plugin_args": {"lokr_factor": factor}}


def _continue_weight_kind(ctx: ContinueWeightDetectionContext) -> str | None:
    if any("lokr_w1" in key or "lokr_w2" in key for key in ctx.lowered_keys):
        return "LoKr"
    if str(ctx.metadata.get("ss_network_spec") or "").strip().lower() == "lokr":
        return "LoKr"
    return None


register_save_handler("lokr", save_lokr_weights)
register_network_spec(
    NetworkSpec(
        name="lokr",
        module_class=LoKrModule,
        save_variant="lokr",
        kwarg_flags=("use_lokr", "lokr_factor"),
        selector=_selector,
        validate=_validate,
        module_kwargs=_module_kwargs,
        detect_from_weights=_detect_from_weights,
        finish_weight_detection=_finish_weight_detection,
        continue_weight_kind=_continue_weight_kind,
    )
)

__all__ = ["LoKrModule"]
