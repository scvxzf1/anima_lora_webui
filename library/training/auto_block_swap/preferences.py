"""Shared resource preferences for calibration, runtime and advisory checks."""

import math

from .resources import GIB

PREFERENCES = ("balanced", "vram", "ram")


def gpu_reserve_bytes(args, total):
    percent = float(getattr(args, "auto_block_swap_vram_reserve_percent", 10.0))
    if not math.isfinite(percent) or not 0 <= percent <= 90:
        raise ValueError("auto_block_swap_vram_reserve_percent must be between 0 and 90")
    # The requested percentage is always based on capacity, never current free VRAM.
    return max(GIB, math.ceil(total * percent / 100))


def preference(args):
    return getattr(args, "auto_block_swap_preference", "balanced")


def swap_io_limit_bytes(args):
    """Maximum cumulative system swap IO allowed per probe/runtime window."""
    # Hand-built test/runtime namespaces from older callers retain the conservative cap.
    limit_mb = float(getattr(args, "auto_block_swap_swap_io_limit_mb", 64.0))
    if not math.isfinite(limit_mb) or limit_mb < 0:
        raise ValueError("auto_block_swap_swap_io_limit_mb must be non-negative")
    return None if limit_mb == 0 else math.ceil(limit_mb * 1024**2)


def prefer_candidate(mode, before, candidate, *, reference_seconds, candidate_seconds):
    if candidate_seconds > reference_seconds * 1.10:
        return False
    return (mode == "ram" and candidate < before) or (
        mode == "vram" and candidate > before
    )
