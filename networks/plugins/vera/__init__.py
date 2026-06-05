"""Bundled VeRA plugin registration."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from networks.plugins.vera.module import VeRAModule, make_projection_bank
from networks.plugins.vera.save import save_vera_weights
from networks.attn_fuse import iter_split_groups
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
    return _truthy(kwargs.get("use_vera"))


def _validate(kwargs: Mapping[str, Any]) -> None:
    if not _selector(kwargs):
        return
    conflicts = []
    for key in ("use_lokr", "use_loha", "use_ortho", "use_chimera_hydra"):
        if _truthy(kwargs.get(key)):
            conflicts.append(key)
    if kwargs.get("use_moe_style") not in (None, False, "", "false", "False"):
        conflicts.append("use_moe_style")
    if conflicts:
        raise ValueError(
            "use_vera is mutually exclusive with " + ", ".join(sorted(conflicts)) + "."
        )


def _as_int(value: Any, default: int) -> int:
    return int(value) if value is not None else default


def _as_float(value: Any, default: float) -> float:
    return float(value) if value is not None else default


def _module_kwargs(ctx: ModuleCreationContext) -> dict[str, Any]:
    args = ctx.cfg.plugin_args
    rank = int(ctx.cfg.modules_dim.get(ctx.lora_name, ctx.cfg.lora_dim)) if ctx.cfg.modules_dim else int(ctx.cfg.lora_dim)
    max_in = _as_int(args.get("_vera_max_in_features"), int(ctx.child_module.in_features))
    max_out = _as_int(args.get("_vera_max_out_features"), int(ctx.child_module.out_features))
    seed = _as_int(args.get("vera_projection_prng_key"), _as_int(args.get("projection_prng_key"), 0))
    save_projection = _truthy(args.get("vera_save_projection")) or _truthy(args.get("save_projection"))
    d_initial = _as_float(args.get("vera_d_initial"), _as_float(args.get("d_initial"), 0.1))

    cache_key = ("_vera_projection_bank", rank, max_in, max_out, seed, save_projection)
    bank = args.get(cache_key)
    if bank is None:
        bank = make_projection_bank(
            rank=rank,
            max_in_features=max_in,
            max_out_features=max_out,
            projection_prng_key=seed,
            save_projection=save_projection,
        )
        args[cache_key] = bank
    return {"projection_bank": bank, "d_initial": d_initial}


def _detect_from_weights(ctx: WeightDetectionContext) -> bool:
    key = ctx.key
    value = ctx.value
    if key.endswith(".vera_lambda_d"):
        ctx.state["has_vera"] = True
        ctx.state["vera_metadata"] = ctx.metadata
        ctx.state.setdefault("vera_module_names", set()).add(ctx.lora_name)
        ctx.modules_dim[ctx.lora_name] = int(value.numel())
        ctx.modules_alpha.setdefault(ctx.lora_name, torch.tensor(float(value.numel())))
        ctx.plain_module_names.add(ctx.lora_name)
        return True
    if key.endswith(".vera_lambda_b"):
        ctx.state["has_vera"] = True
        ctx.state["vera_metadata"] = ctx.metadata
        ctx.state.setdefault("vera_module_names", set()).add(ctx.lora_name)
        ctx.modules_dim.setdefault(ctx.lora_name, 1)
        ctx.modules_alpha.setdefault(ctx.lora_name, torch.tensor(float(ctx.modules_dim[ctx.lora_name])))
        ctx.plain_module_names.add(ctx.lora_name)
        return True
    return False


def _finish_weight_detection(
    state: dict[str, Any],
    modules_dim: dict[str, int],
    modules_alpha: dict[str, Any],
) -> dict[str, Any]:
    del modules_dim, modules_alpha
    if not state.get("has_vera"):
        return {}
    metadata = state.get("vera_metadata") or {}
    plugin_args: dict[str, Any] = {}
    if "ss_vera_projection_prng_key" in metadata:
        plugin_args["vera_projection_prng_key"] = metadata["ss_vera_projection_prng_key"]
    if "ss_vera_d_initial" in metadata:
        plugin_args["vera_d_initial"] = metadata["ss_vera_d_initial"]
    if "ss_vera_save_projection" in metadata:
        plugin_args["vera_save_projection"] = metadata["ss_vera_save_projection"]
    return {"detected_spec": "vera", "plugin_args": plugin_args}


def _preprocess_weights(weights_sd: dict[str, Any]) -> dict[str, Any]:
    for shared_prefix, spec in iter_split_groups(weights_sd, ".vera_lambda_b"):
        suffixes = spec.component_letters
        lambda_bs = []
        lambda_ds = []
        alphas = []
        complete = True
        for letter in suffixes:
            prefix = f"{shared_prefix}{letter}_proj"
            bk = f"{prefix}.vera_lambda_b"
            dk = f"{prefix}.vera_lambda_d"
            ak = f"{prefix}.alpha"
            if bk not in weights_sd or dk not in weights_sd:
                complete = False
                break
            lambda_bs.append(weights_sd[bk])
            lambda_ds.append(weights_sd[dk])
            alphas.append(weights_sd.get(ak))
        if not complete:
            continue
        if not all(d.shape == lambda_ds[0].shape for d in lambda_ds):
            continue
        if not all(torch.equal(lambda_ds[0], d) for d in lambda_ds[1:]):
            continue

        fused_prefix = f"{shared_prefix}{spec.fused_letters}_proj"
        weights_sd[f"{fused_prefix}.vera_lambda_b"] = torch.cat(lambda_bs, dim=0).contiguous()
        weights_sd[f"{fused_prefix}.vera_lambda_d"] = lambda_ds[0].contiguous()
        first_alpha = next((a for a in alphas if a is not None), None)
        if first_alpha is not None:
            weights_sd[f"{fused_prefix}.alpha"] = first_alpha

        for letter in suffixes:
            prefix = f"{shared_prefix}{letter}_proj"
            for leaf in ("vera_lambda_b", "vera_lambda_d", "alpha"):
                weights_sd.pop(f"{prefix}.{leaf}", None)
    return weights_sd


def _continue_weight_kind(ctx: ContinueWeightDetectionContext) -> str | None:
    if any("vera_lambda_" in key for key in ctx.lowered_keys):
        return "VeRA"
    if str(ctx.metadata.get("ss_network_spec") or "").strip().lower() == "vera":
        return "VeRA"
    return None


register_save_handler("vera", save_vera_weights)
register_network_spec(
    NetworkSpec(
        name="vera",
        module_class=VeRAModule,
        save_variant="vera",
        kwarg_flags=(
            "use_vera",
            "vera_projection_prng_key",
            "projection_prng_key",
            "vera_d_initial",
            "d_initial",
            "vera_save_projection",
            "save_projection",
        ),
        selector=_selector,
        validate=_validate,
        module_kwargs=_module_kwargs,
        preprocess_weights=_preprocess_weights,
        detect_from_weights=_detect_from_weights,
        finish_weight_detection=_finish_weight_detection,
        continue_weight_kind=_continue_weight_kind,
    )
)

__all__ = ["VeRAModule"]
