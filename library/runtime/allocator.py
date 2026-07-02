"""CUDA allocator defaults that must land before torch import."""

from __future__ import annotations

import os
import sys


def default_expandable_segments() -> bool:
    """Default ``PYTORCH_CUDA_ALLOC_CONF`` to ``expandable_segments:True``."""
    if sys.platform != "linux":
        return False
    if os.environ.get("ANIMA_EXPANDABLE_SEGMENTS", "1").lower() in (
        "0",
        "false",
    ):
        return False
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        return False
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return True
