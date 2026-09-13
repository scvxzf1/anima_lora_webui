"""Low-cadence advisory monitoring; never mutate the offloader or retry OOM."""

from __future__ import annotations

import logging
import random

from .resources import host_memory
from .preferences import gpu_reserve_bytes

logger = logging.getLogger(__name__)


def observe_training(args, device, step: int) -> None:
    if (
        not getattr(args, "_auto_swap_resolved", False)
        or getattr(args, "auto_block_swap_mode", "startup") == "dynamic"
        or step % 16
        or getattr(args, "_auto_swap_warned", False)
    ):
        return
    import torch

    try:
        free, total = torch.cuda.mem_get_info(device)
        headroom = (
            free
            + torch.cuda.memory_reserved(device)
            - torch.cuda.max_memory_allocated(device)
        )
        host = host_memory()
        if headroom >= gpu_reserve_bytes(args, total) and host.available > host.reserve:
            return
        args._auto_swap_warned = True
        # Rich logging consumes Python RNG; telemetry must not alter training.
        rng = random.getstate()
        try:
            logger.warning(
                "AUTO block swap: resource margin is below the calibration reserve at step %s. "
                "The swap count remains fixed. Recalibrate after stopping safely; "
                "do not retry a partially completed optimizer update. Report: %s",
                step,
                args._auto_swap_report,
            )
        finally:
            random.setstate(rng)
    except (RuntimeError, OSError):
        return
