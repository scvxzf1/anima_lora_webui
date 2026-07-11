"""Parse cache reuse policy from training config dicts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

FingerprintMode = Literal["light", "content"]


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class CacheReusePolicy:
    reuse_dataset_cache_copy: bool = True
    reuse_vae_latents: bool = True
    reuse_text_encoder_cache: bool = True
    fingerprint_mode: FingerprintMode = "light"
    force_rebuild: bool = False


def parse_cache_reuse_policy(cfg: dict[str, Any] | None) -> CacheReusePolicy:
    cfg = cfg or {}
    mode_raw = str(cfg.get("cache_fingerprint_mode") or "light").strip().lower()
    mode: FingerprintMode = "content" if mode_raw == "content" else "light"
    return CacheReusePolicy(
        reuse_dataset_cache_copy=_as_bool(cfg.get("reuse_dataset_cache_copy"), True),
        reuse_vae_latents=_as_bool(cfg.get("reuse_vae_latents"), True),
        reuse_text_encoder_cache=_as_bool(cfg.get("reuse_text_encoder_cache"), True),
        fingerprint_mode=mode,
        force_rebuild=_as_bool(cfg.get("force_rebuild_preprocess_cache"), False),
    )
