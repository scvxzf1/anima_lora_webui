"""Local tagging provider factory with lazy imports."""

from __future__ import annotations

from typing import Any

from .base import Tagger


def get_tagger(provider: str, settings: dict[str, Any] | None = None) -> Tagger:
    provider_id = str(provider or "").strip().lower()
    if provider_id == "wd14":
        from .wd14 import WD14Tagger

        return WD14Tagger(settings)
    if provider_id == "cltagger":
        from .cltagger import CLTagger

        return CLTagger(settings)
    raise ValueError(f"不支持的本地打标 provider：{provider_id or '空值'}")


__all__ = ["get_tagger"]
