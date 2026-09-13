"""Explicit mask-mode semantics shared by config, datasets, and WebUI."""

from __future__ import annotations

from dataclasses import dataclass
import os

MASK_MODE_AUTO = "auto"
MASK_MODE_NONE = "none"
MASK_MODE_EXTERNAL = "external"
MASK_MODE_EMBEDDED = "embedded"
MASK_MODES = frozenset(
    {MASK_MODE_AUTO, MASK_MODE_NONE, MASK_MODE_EXTERNAL, MASK_MODE_EMBEDDED}
)
LEGACY_AUTO_MASK_DIRS = (
    "post_image_dataset/masks",
    "masks/merged",
    "masks/sam",
    "masks/mit",
)


@dataclass(frozen=True)
class MaskModeConfig:
    mode: str
    alpha_mask: bool
    mask_dir: str | None


def resolve_legacy_auto_mask_dir() -> str | None:
    """Return the first legacy default mask directory present in the CWD."""

    return next((path for path in LEGACY_AUTO_MASK_DIRS if os.path.isdir(path)), None)


def normalize_mask_mode(
    value,
    *,
    alpha_mask: bool = False,
    mask_dir: str | None = None,
) -> MaskModeConfig:
    """Resolve legacy flags into one explicit mask mode.

    An omitted mode keeps the old behavior: an explicit ``mask_dir`` selects
    external masks, ``alpha_mask=true`` selects embedded alpha, and otherwise
    the caller may perform legacy default-directory discovery for ``auto``.
    """

    raw = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "off": MASK_MODE_NONE,
        "disabled": MASK_MODE_NONE,
        "alpha": MASK_MODE_EMBEDDED,
        "image_alpha": MASK_MODE_EMBEDDED,
        "mask_dir": MASK_MODE_EXTERNAL,
    }
    raw = aliases.get(raw, raw)
    if not raw:
        if str(mask_dir or "").strip():
            raw = MASK_MODE_EXTERNAL
        elif bool(alpha_mask):
            raw = MASK_MODE_EMBEDDED
        else:
            raw = MASK_MODE_AUTO
    if raw not in MASK_MODES:
        allowed = ", ".join(sorted(MASK_MODES))
        raise ValueError(f"mask_mode must be one of: {allowed}; got {value!r}")

    normalized_dir = str(mask_dir or "").strip() or None
    if raw == MASK_MODE_AUTO and normalized_dir is not None:
        raw = MASK_MODE_EXTERNAL
    elif raw == MASK_MODE_AUTO and bool(alpha_mask):
        raw = MASK_MODE_EMBEDDED
    if raw == MASK_MODE_NONE:
        return MaskModeConfig(raw, False, None)
    if raw == MASK_MODE_EMBEDDED:
        return MaskModeConfig(raw, True, None)
    if raw == MASK_MODE_EXTERNAL:
        if normalized_dir is None:
            raise ValueError("mask_mode='external' requires mask_dir")
        return MaskModeConfig(raw, True, normalized_dir)
    return MaskModeConfig(raw, bool(alpha_mask), normalized_dir)
