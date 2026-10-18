"""In-process block-swap water-level auto tuner.

The existing AUTO block swap is a separate-process startup calibration and its docs
state it is deliberately not the same as blocks_to_swap = "auto". This module adds the
lighter in-process variant: after the first optimizer step the run measures its own
free VRAM, converts the remaining headroom into "how many more swapped blocks can stay
resident", and promotes them at the optimizer boundary through the existing
ModelOffloader.reconfigure() transaction.

Promotion only ever reduces the swap count, so it cannot OOM a run that started with a
conservative count; the boundary transaction rolls itself back if migration allocation
fails.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_MARGIN_MIB = 400
_ENV_MARGIN_MIB = "ANIMA_BLOCK_SWAP_AUTO_MARGIN_MIB"


def resolve_margin_bytes(margin_mib: Optional[int] = None) -> int:
    """Free-VRAM safety margin used by the promotion planning."""
    if margin_mib is None:
        raw = os.getenv(_ENV_MARGIN_MIB, "").strip()
        try:
            margin_mib = int(raw) if raw else DEFAULT_MARGIN_MIB
        except ValueError:
            margin_mib = DEFAULT_MARGIN_MIB
    return max(0, int(margin_mib)) * 1024 * 1024


def resolve_offloader_and_blocks(model: Any) -> Optional[tuple[Any, Any]]:
    """Best-effort (offloader, blocks) lookup across the model families.

    Anima and Krea-2 keep both attributes on the DiT itself; Z-Image and
    Qwen-Image 2.1 wrap the offloader in an adapter. Returns None when the model is
    not block-swapping, so callers can warn instead of failing a run.
    """
    if model is None:
        return None
    candidates = [model]
    for attr in (
        "_z_image_block_swap_adapter",
        "_qwen_image_2_1_block_swap_adapter",
        "offloader_holder",
    ):
        holder = getattr(model, attr, None)
        if holder is not None:
            candidates.append(holder)
    for holder in candidates:
        offloader = getattr(holder, "offloader", None)
        blocks = getattr(holder, "blocks", None)
        if blocks is None:
            blocks = getattr(holder, "layers", None)
        if offloader is not None and blocks is not None:
            return offloader, blocks
    return None


def per_block_bytes(offloader: Any, block_index: int) -> int:
    """Resident cost of one block: its frozen CPU masters stored bytes."""
    masters = getattr(offloader, "_cpu_weight_masters", None)
    if not masters or block_index >= len(masters) or block_index < 0:
        return 0
    total = 0
    for master in masters[block_index].values():
        stored = getattr(master, "stored_nbytes", None)
        if callable(stored):
            try:
                total += int(stored())
                continue
            except Exception:
                pass
        try:
            total += int(getattr(master, "nbytes", 0) or 0)
        except Exception:
            pass
    return total


def cuda_free_bytes(device: Any = None) -> Optional[int]:
    """Free device memory, or None when CUDA is unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info(device)
        return int(free)
    except Exception:
        return None


def plan_promotion(
    offloader: Any,
    blocks: Any,
    *,
    margin_bytes: int,
    free_bytes: Optional[int] = None,
) -> dict:
    """How many swapped blocks can become resident, and the resulting count."""
    current = int(getattr(offloader, "blocks_to_swap", 0) or 0)
    count = len(blocks)
    if free_bytes is None:
        free_bytes = cuda_free_bytes(getattr(offloader, "device", None))
    block_bytes = per_block_bytes(offloader, count - current) if current else 0
    plan = {
        "current_swap": current,
        "num_blocks": count,
        "free_bytes": free_bytes,
        "margin_bytes": int(margin_bytes),
        "block_bytes": block_bytes,
        "promotable": 0,
        "new_swap": current,
        "reason": "",
    }
    if current == 0:
        plan["reason"] = "block swap disabled"
        return plan
    if free_bytes is None:
        plan["reason"] = "CUDA unavailable"
        return plan
    if block_bytes <= 0:
        plan["reason"] = "no CPU masters measured yet"
        return plan
    headroom = int(free_bytes) - int(margin_bytes)
    if headroom <= 0:
        plan["reason"] = "no headroom above margin"
        return plan
    promotable = headroom // block_bytes
    # Keep the offloader contract: at least two blocks stay swapped.
    promotable = min(promotable, max(0, current - 2))
    plan["promotable"] = int(promotable)
    plan["new_swap"] = int(current - promotable)
    plan["reason"] = "headroom converted into resident blocks"
    return plan


def maybe_promote(
    model: Any,
    *,
    margin_mib: Optional[int] = None,
    free_bytes: Optional[int] = None,
) -> Optional[dict]:
    """Plan and apply one promotion at an optimizer boundary.

    Returns the plan with an applied flag, or None when the model is not
    block-swapping.
    """
    found = resolve_offloader_and_blocks(model)
    if found is None:
        logger.warning(
            "block_swap_auto requested but no block-swap offloader was found on the model; "
            "leaving the manual blocks_to_swap unchanged."
        )
        return None
    offloader, blocks = found
    plan = plan_promotion(
        offloader,
        blocks,
        margin_bytes=resolve_margin_bytes(margin_mib),
        free_bytes=free_bytes,
    )
    plan["applied"] = False
    if plan["promotable"] <= 0 or plan["new_swap"] == plan["current_swap"]:
        logger.info(
            "block_swap_auto: keeping blocks_to_swap=%s (free=%.2f GiB, margin=%.2f GiB, "
            "block=%.1f MiB, %s)",
            plan["current_swap"],
            (plan["free_bytes"] or 0) / 2**30,
            plan["margin_bytes"] / 2**30,
            plan["block_bytes"] / 2**20,
            plan["reason"],
        )
        return plan
    logger.info(
        "block_swap_auto: promoting %s block(s) to resident, blocks_to_swap %s -> %s "
        "(free=%.2f GiB, margin=%.2f GiB, block=%.1f MiB)",
        plan["promotable"],
        plan["current_swap"],
        plan["new_swap"],
        (plan["free_bytes"] or 0) / 2**30,
        plan["margin_bytes"] / 2**30,
        plan["block_bytes"] / 2**20,
    )
    plan["applied"] = bool(offloader.reconfigure(blocks, plan["new_swap"]))
    return plan
