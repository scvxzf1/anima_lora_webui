"""Configuration normalization helpers for runtime block swapping."""

from __future__ import annotations

import os
from typing import Optional

import torch

from library.runtime.int8_linear import INT8_LINEAR_SCOPE_MODULES

_BLOCK_SWAP_TRANSFER_DTYPES = {"bf16", "fp8_e4m3", "int8"}
_BLOCK_SWAP_RESTORE_MODES = {"foreach", "slab"}
_BLOCK_SWAP_INT8_RESTORE_MODES = {"copy", "direct_bind", "reuse_storage", "fused"}
_BLOCK_SWAP_INT8_SCOPES = {"all", *INT8_LINEAR_SCOPE_MODULES}
_DEFAULT_BLOCK_SWAP_PROFILE_POLL_INTERVAL_S = 0.05


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _block_swap_profile_poll_interval_s() -> float:
    raw = os.getenv("ANIMA_BLOCK_SWAP_PROFILE_POLL_MS")
    if raw is None or not str(raw).strip():
        return _DEFAULT_BLOCK_SWAP_PROFILE_POLL_INTERVAL_S
    try:
        value = float(str(raw).strip())
    except ValueError:
        return _DEFAULT_BLOCK_SWAP_PROFILE_POLL_INTERVAL_S
    if value <= 0:
        return _DEFAULT_BLOCK_SWAP_PROFILE_POLL_INTERVAL_S
    return max(0.001, value / 1000.0)


def _block_swap_prefetch_depth() -> int:
    """Forward prefetch lead (number of blocks ahead to start H2D restore).

    Retained only as a compatibility knob. The block-swap design keeps a fixed
    ``num_blocks - blocks_to_swap`` resident GPU slots and maps each retired
    block's storage to exactly one incoming block, so a lead greater than 1 must
    retire a block that has not run yet — parking its live weights to CPU right
    before its forward (``mat2 is on cpu``) or silently overwriting its storage.
    The lead is therefore pinned to 1 in ``submit_move_blocks`` regardless of
    this value. Kept >= 1 so existing env settings parse without error.
    """
    return max(1, _env_int("ANIMA_BLOCK_SWAP_PREFETCH_DEPTH", default=1))


def _env_int(name: str, *, default: int = 0) -> int:
    value = os.getenv(name)
    if value is None or not str(value).strip():
        return default
    try:
        parsed = int(str(value).strip())
    except ValueError:
        return default
    return max(0, parsed)


def normalize_block_swap_transfer_dtype(value: Optional[str]) -> str:
    dtype = str(value or "bf16").strip().lower()
    aliases = {
        "bfloat16": "bf16",
        "float8_e4m3": "fp8_e4m3",
        "float8_e4m3fn": "fp8_e4m3",
        "e4m3": "fp8_e4m3",
        "int8_linear": "int8",
        "i8": "int8",
    }
    dtype = aliases.get(dtype, dtype)
    if dtype not in _BLOCK_SWAP_TRANSFER_DTYPES:
        raise ValueError(
            "block_swap_transfer_dtype must be one of: "
            f"{', '.join(sorted(_BLOCK_SWAP_TRANSFER_DTYPES))}"
        )
    if dtype == "fp8_e4m3" and not hasattr(torch, "float8_e4m3fn"):
        raise ValueError("block_swap_transfer_dtype=fp8_e4m3 requires torch.float8_e4m3fn")
    return dtype


def normalize_block_swap_restore_mode(value: Optional[str]) -> str:
    mode = str(value or "foreach").strip().lower()
    aliases = {
        "default": "foreach",
        "loop": "foreach",
    }
    mode = aliases.get(mode, mode)
    if mode not in _BLOCK_SWAP_RESTORE_MODES:
        raise ValueError(
            "block_swap_restore_mode must be one of: "
            f"{', '.join(sorted(_BLOCK_SWAP_RESTORE_MODES))}"
        )
    return mode


def normalize_block_swap_int8_restore_mode(value: Optional[str]) -> str:
    mode = str(value or "fused").strip().lower()
    aliases = {
        "default": "fused",
        "reuse": "reuse_storage",
        "inplace": "reuse_storage",
        "into": "reuse_storage",
        "bind": "direct_bind",
        "direct": "direct_bind",
        "single_kernel": "fused",
        "one_kernel": "fused",
    }
    mode = aliases.get(mode, mode)
    if mode not in _BLOCK_SWAP_INT8_RESTORE_MODES:
        raise ValueError(
            "block_swap_int8_restore_mode must be one of: "
            f"{', '.join(sorted(_BLOCK_SWAP_INT8_RESTORE_MODES))}"
        )
    return mode


_DEFAULT_BLOCK_SWAP_INT8_STAGING_RING = 2
_DEFAULT_BLOCK_SWAP_INT8_RESTORE_CHUNK_ROWS = 0


def normalize_block_swap_int8_staging_ring(value: Optional[int]) -> int:
    """int8 fused staging buffer count: 0 = per slot, N >= 1 = shared ring."""
    if value is None or (isinstance(value, str) and not str(value).strip()):
        return _DEFAULT_BLOCK_SWAP_INT8_STAGING_RING
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            "block_swap_int8_staging_ring must be an integer >= 0 "
            "(0 = one buffer per swap slot, 1 = one shared buffer, "
            "N = N buffers shared by all slots)"
        )
    if parsed < 0:
        raise ValueError("block_swap_int8_staging_ring must be >= 0")
    return parsed


def normalize_block_swap_int8_restore_chunk_rows(value: Optional[int]) -> int:
    if value is None or (isinstance(value, str) and not str(value).strip()):
        return _DEFAULT_BLOCK_SWAP_INT8_RESTORE_CHUNK_ROWS
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError("block_swap_int8_restore_chunk_rows must be an integer >= 0")
    if parsed < 0:
        raise ValueError("block_swap_int8_restore_chunk_rows must be >= 0")
    return parsed


def resolve_block_swap_int8_staging_ring(value: Optional[int] = None) -> int:
    """CLI/TOML value wins; ANIMA_BLOCK_SWAP_INT8_STAGING_RING is the fallback."""
    if value is not None and not (isinstance(value, str) and not str(value).strip()):
        return normalize_block_swap_int8_staging_ring(value)
    return normalize_block_swap_int8_staging_ring(
        _env_int(
            "ANIMA_BLOCK_SWAP_INT8_STAGING_RING",
            default=_DEFAULT_BLOCK_SWAP_INT8_STAGING_RING,
        )
    )


def resolve_block_swap_int8_restore_chunk_rows(value: Optional[int] = None) -> int:
    if value is not None and not (isinstance(value, str) and not str(value).strip()):
        return normalize_block_swap_int8_restore_chunk_rows(value)
    return normalize_block_swap_int8_restore_chunk_rows(
        _env_int(
            "ANIMA_BLOCK_SWAP_INT8_RESTORE_CHUNK_ROWS",
            default=_DEFAULT_BLOCK_SWAP_INT8_RESTORE_CHUNK_ROWS,
        )
    )


_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def normalize_block_swap_int8_dual_stream(value: Optional[object]) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return False
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    raise ValueError("block_swap_int8_dual_stream must be a boolean")


def resolve_block_swap_int8_dual_stream(value: Optional[object] = None) -> bool:
    """CLI/TOML value wins; ANIMA_BLOCK_SWAP_INT8_DUAL_STREAM is the fallback."""
    if value is not None and not (isinstance(value, str) and not str(value).strip()):
        return normalize_block_swap_int8_dual_stream(value)
    return _env_flag("ANIMA_BLOCK_SWAP_INT8_DUAL_STREAM", default=False)


def normalize_block_swap_int8_scope(value: Optional[str]) -> str:
    scope = str(value or "all").strip().lower()
    if not scope:
        return "all"
    parts = [item.strip() for item in scope.split(",") if item.strip()]
    if not parts:
        return "all"
    unknown = set(parts) - _BLOCK_SWAP_INT8_SCOPES
    if unknown:
        raise ValueError(
            "block_swap_int8_scope must be a comma-separated subset of: "
            f"{', '.join(sorted(_BLOCK_SWAP_INT8_SCOPES))}"
        )
    if "all" in parts:
        return "all"
    return ",".join(dict.fromkeys(parts))
