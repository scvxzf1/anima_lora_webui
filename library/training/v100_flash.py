"""V100 FlashAttention configuration and provider diagnostics."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def env_flag(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }


def resolve_v100_flash_stability(args) -> str:
    value = getattr(args, "v100_flash_stability", None)
    if value is None:
        value = os.environ.get("ANIMA_V100_FLASH_STABILITY", "off")
    value = str(value).lower()
    if value not in {"off", "hybrid", "safe"}:
        logger.warning(
            "invalid ANIMA_V100_FLASH_STABILITY=%r; expected "
            "off|hybrid|safe, using off",
            value,
        )
        return "off"
    return value


def resolve_debug_finite_checks(args, v100_flash_stability: str) -> bool:
    return (
        bool(getattr(args, "debug_finite_checks", False))
        or env_flag("ANIMA_DEBUG_FINITE")
        or v100_flash_stability == "safe"
    )


def flash_attn_v100_doc(flash_attn_module) -> tuple[str, bool]:
    doc = getattr(flash_attn_module, "__doc__", None) or ""
    is_v100_fork = "Tesla V100" in doc or "Flash Attention for Tesla V100" in doc
    return doc.strip().replace("\n", " "), is_v100_fork
