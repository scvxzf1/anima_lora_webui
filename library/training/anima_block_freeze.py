"""Experimental Anima LoRA block freezing for controlled resume A/B runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import re
from typing import Any

import torch


logger = logging.getLogger(__name__)

_BLOCK_NAME_RE = re.compile(r"^blocks\.(\d+)(?:\.|$)")


@dataclass(frozen=True)
class AnimaBlockFreezeSummary:
    blocks: tuple[int, ...]
    module_count: int
    parameter_count: int
    fused: bool
    resume_step: int
    module_names: tuple[str, ...]


def parse_anima_freeze_blocks(value: Any) -> tuple[int, ...]:
    """Parse a comma/space-separated block list into sorted unique indices."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        tokens = [str(item).strip() for item in value]
    else:
        text = str(value).strip()
        if not text or text.lower() in {"none", "off"}:
            return ()
        tokens = re.split(r"[\s,]+", text)

    try:
        blocks = {int(token) for token in tokens if token}
    except ValueError as exc:
        raise ValueError(
            f"anima_freeze_blocks must contain integer indices, got {value!r}"
        ) from exc
    if any(block < 0 for block in blocks):
        raise ValueError("anima_freeze_blocks cannot contain negative indices")
    return tuple(sorted(blocks))


def _block_index(lora: torch.nn.Module) -> int | None:
    match = _BLOCK_NAME_RE.match(str(getattr(lora, "original_name", "")))
    return int(match.group(1)) if match else None


def _validate_experiment(args: Any, network: torch.nn.Module, *, fused: bool) -> None:
    if str(getattr(args, "model_family", "") or "anima") != "anima":
        raise ValueError("anima_freeze_blocks is only supported for model_family='anima'")
    if str(getattr(args, "network_module", "") or "") != "networks.lora_anima":
        raise ValueError(
            "anima_freeze_blocks requires network_module='networks.lora_anima'"
        )
    if fused and bool(getattr(args, "torch_compile", False)):
        raise ValueError(
            "anima_freeze_fuse requires torch_compile=false because fusion occurs "
            "after resume and would invalidate captured adapter branches"
        )
    if fused and int(getattr(args, "blocks_to_swap", 0) or 0) != 0:
        raise ValueError(
            "anima_freeze_fuse requires blocks_to_swap=0 for a writable resident base"
        )
    if fused and not bool(network.is_mergeable()):
        raise ValueError("anima_freeze_fuse requires a statically mergeable LoRA network")


def _write_audit(args: Any, summary: AnimaBlockFreezeSummary) -> None:
    output_dir = Path(str(getattr(args, "output_dir", ".") or "."))
    output_name = str(getattr(args, "output_name", "training") or "training")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{output_name}.anima_block_freeze.json"
    payload = asdict(summary)
    payload["blocks"] = list(summary.blocks)
    payload["module_names"] = list(summary.module_names)
    payload["optimizer_param_groups_preserved"] = True
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def apply_anima_block_freeze(
    args: Any,
    accelerator: Any,
    network: torch.nn.Module,
    *,
    resume_step: int,
) -> AnimaBlockFreezeSummary | None:
    """Freeze selected resumed LoRA blocks and optionally fold their live delta.

    Optimizer parameter groups deliberately remain unchanged so Accelerate can
    restore the control and treatment runs from the exact same optimizer state.
    Frozen parameters have ``grad=None`` and are skipped by AdamW. When fusion
    is enabled, the base Linear still propagates gradients to its input while
    the redundant LoRA branch becomes a no-op.
    """
    blocks = parse_anima_freeze_blocks(getattr(args, "anima_freeze_blocks", None))
    if not blocks:
        return None
    if resume_step <= 0:
        raise ValueError(
            "anima_freeze_blocks is an experimental resume-only control; "
            "a checkpoint with current_step > 0 is required"
        )

    unwrapped = accelerator.unwrap_model(network)
    fused = bool(getattr(args, "anima_freeze_fuse", False))
    _validate_experiment(args, unwrapped, fused=fused)

    selected = [
        lora
        for lora in getattr(unwrapped, "unet_loras", ())
        if _block_index(lora) in blocks
    ]
    found_blocks = {_block_index(lora) for lora in selected}
    missing = sorted(set(blocks) - found_blocks)
    if missing:
        raise ValueError(f"anima_freeze_blocks did not match model blocks: {missing}")

    module_names: list[str] = []
    parameter_count = 0
    for lora in selected:
        if fused:
            fuse_weight = getattr(lora, "fuse_weight", None)
            if not callable(fuse_weight):
                raise ValueError(
                    f"LoRA module {getattr(lora, 'lora_name', type(lora).__name__)} "
                    "does not support static fusion"
                )
            fuse_weight()
        for parameter in lora.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
            parameter_count += parameter.numel()
        module_names.append(str(getattr(lora, "lora_name", lora.original_name)))

    summary = AnimaBlockFreezeSummary(
        blocks=blocks,
        module_count=len(selected),
        parameter_count=parameter_count,
        fused=fused,
        resume_step=int(resume_step),
        module_names=tuple(sorted(module_names)),
    )
    if bool(getattr(accelerator, "is_main_process", True)):
        _write_audit(args, summary)
    logger.info(
        "Anima resume freeze applied at step %d: blocks=%s modules=%d params=%d fused=%s; "
        "optimizer parameter groups preserved",
        resume_step,
        ",".join(map(str, blocks)),
        summary.module_count,
        summary.parameter_count,
        fused,
    )
    return summary
