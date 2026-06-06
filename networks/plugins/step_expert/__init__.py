"""Step-expert LoRA plugin registration."""

from __future__ import annotations

import re
from typing import Any, Mapping

import torch

from networks.lora_modules.step_expert import StepExpertLoRAModule
from networks.registry import (
    ContinueWeightDetectionContext,
    ModuleCreationContext,
    NetworkSpec,
    WeightDetectionContext,
    register_network_spec,
)


def _int_value(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _selector(kwargs: Mapping[str, Any]) -> bool:
    return _int_value(kwargs.get("step_expert_K")) > 1


def _validate(kwargs: Mapping[str, Any]) -> None:
    if not _selector(kwargs):
        return
    if any(str(kwargs.get(key, "")).strip().lower() in {"1", "true", "yes", "on"} for key in ("use_lokr", "use_loha", "use_vera", "use_glora", "dora_wd")):
        raise ValueError(
            "step_expert_K is mutually exclusive with LoKr/LoHa/VeRA/GLoRA/DoRA."
        )
    if kwargs.get("use_moe_style") not in (None, False, "", "false", "False"):
        raise ValueError("step_expert_K is mutually exclusive with MoE LoRA.")
    if str(kwargs.get("use_chimera_hydra", "")).strip().lower() in {"1", "true"}:
        raise ValueError("step_expert_K is mutually exclusive with ChimeraHydra.")


def _module_kwargs(ctx: ModuleCreationContext) -> dict[str, Any]:
    return {"step_expert_K": int(ctx.cfg.plugin_args.get("step_expert_K", 2))}


_UP_RE = re.compile(r"\.lora_ups\.(\d+)\.weight$")


def _detect_from_weights(ctx: WeightDetectionContext) -> bool:
    key = ctx.key
    value = ctx.value
    match = _UP_RE.search(key)
    if match:
        ctx.state["has_step_expert"] = True
        ctx.state.setdefault("step_expert_module_names", set()).add(ctx.lora_name)
        ctx.state["step_expert_K"] = max(
            int(ctx.state.get("step_expert_K", 0)),
            int(match.group(1)) + 1,
        )
        if value.dim() == 2:
            ctx.modules_dim[ctx.lora_name] = int(value.size(1))
            ctx.modules_alpha.setdefault(
                ctx.lora_name, torch.tensor(float(value.size(1)))
            )
        ctx.plain_module_names.add(ctx.lora_name)
        return True
    if key.endswith(".lora_down.weight") and ctx.state.get("has_step_expert"):
        ctx.modules_dim[ctx.lora_name] = int(value.size(0))
        ctx.modules_alpha.setdefault(
            ctx.lora_name, torch.tensor(float(value.size(0)))
        )
        return True
    return False


def _finish_weight_detection(
    state: dict[str, Any],
    modules_dim: dict[str, int],
    modules_alpha: dict[str, Any],
) -> dict[str, Any]:
    del modules_dim, modules_alpha
    if not state.get("has_step_expert"):
        return {}
    return {
        "detected_spec": "step_expert",
        "plugin_args": {"step_expert_K": int(state.get("step_expert_K", 2))},
    }


def _continue_weight_kind(ctx: ContinueWeightDetectionContext) -> str | None:
    if any(".lora_ups." in key for key in ctx.lowered_keys):
        return "StepExpert LoRA"
    if str(ctx.metadata.get("ss_network_spec") or "").strip().lower() == "step_expert":
        return "StepExpert LoRA"
    return None


register_network_spec(
    NetworkSpec(
        name="step_expert",
        module_class=StepExpertLoRAModule,
        save_variant="standard",
        kwarg_flags=("step_expert_K",),
        selector=_selector,
        validate=_validate,
        module_kwargs=_module_kwargs,
        detect_from_weights=_detect_from_weights,
        finish_weight_detection=_finish_weight_detection,
        continue_weight_kind=_continue_weight_kind,
    )
)

__all__ = ["StepExpertLoRAModule"]
