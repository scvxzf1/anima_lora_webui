"""Bundled LoHa plugin registration."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from networks.plugins.loha.module import LoHaModule
from networks.plugins.loha.save import save_loha_weights
from networks.registry import (
    ContinueWeightDetectionContext,
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
    return _truthy(kwargs.get("use_loha"))


def _validate(kwargs: Mapping[str, Any]) -> None:
    if not _selector(kwargs):
        return
    if _truthy(kwargs.get("use_lokr")):
        raise ValueError("use_loha is mutually exclusive with use_lokr.")
    if (
        _truthy(kwargs.get("use_ortho"))
        or _truthy(kwargs.get("use_chimera_hydra"))
        or kwargs.get("use_moe_style") not in (None, False, "", "false", "False")
    ):
        raise ValueError(
            "use_loha is mutually exclusive with use_ortho, "
            "use_moe_style, and use_chimera_hydra."
        )


def _detect_from_weights(ctx: WeightDetectionContext) -> bool:
    key = ctx.key
    value = ctx.value

    suffixes = (".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b")
    if not key.endswith(suffixes):
        return False

    ctx.state["has_loha"] = True
    ctx.state.setdefault("loha_module_names", set()).add(ctx.lora_name)
    if value.dim() == 2:
        if key.endswith((".hada_w1_a", ".hada_w2_a")):
            rank = int(value.size(1))
        else:
            rank = int(value.size(0))
    else:
        rank = 1
    ctx.modules_dim[ctx.lora_name] = rank
    ctx.modules_alpha.setdefault(
        ctx.lora_name, torch.tensor(float(ctx.modules_dim[ctx.lora_name]))
    )
    ctx.plain_module_names.add(ctx.lora_name)
    return True


def _finish_weight_detection(
    state: dict[str, Any],
    modules_dim: dict[str, int],
    modules_alpha: dict[str, Any],
) -> dict[str, Any]:
    del modules_dim, modules_alpha
    if not state.get("has_loha"):
        return {}
    return {"detected_spec": "loha"}


def _continue_weight_kind(ctx: ContinueWeightDetectionContext) -> str | None:
    if any("hada_w1_" in key or "hada_w2_" in key for key in ctx.lowered_keys):
        return "LoHa"
    if str(ctx.metadata.get("ss_network_spec") or "").strip().lower() == "loha":
        return "LoHa"
    return None


register_save_handler("loha", save_loha_weights)
register_network_spec(
    NetworkSpec(
        name="loha",
        module_class=LoHaModule,
        save_variant="loha",
        kwarg_flags=("use_loha",),
        selector=_selector,
        validate=_validate,
        detect_from_weights=_detect_from_weights,
        finish_weight_detection=_finish_weight_detection,
        continue_weight_kind=_continue_weight_kind,
    )
)

__all__ = ["LoHaModule"]
