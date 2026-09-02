"""Static registry for tagging provider types.

The registry intentionally contains metadata only.  Provider implementations are
loaded by the tagging service, so adding a model type does not require exposing
an import path or executing user supplied code.
"""

from __future__ import annotations

from typing import Any


_PROVIDER_TYPES: dict[str, dict[str, Any]] = {
    "openai_compatible": {
        "id": "openai_compatible",
        "label": "OpenAI-compatible API",
        "kind": "external",
        "implemented": True,
        "requires_api_key": True,
        "requires_model": True,
        "supports_prompt": True,
        "output_kind": "caption",
    },
    "wd14": {
        "id": "wd14",
        "label": "WD14（本地 ONNX）",
        "kind": "local",
        "implemented": True,
        "requires_api_key": False,
        "requires_model": True,
        "supports_prompt": False,
        "output_kind": "tags",
    },
    "cltagger": {
        "id": "cltagger",
        "label": "CLTagger（本地 ONNX）",
        "kind": "local",
        "implemented": True,
        "requires_api_key": False,
        "requires_model": True,
        "supports_prompt": False,
        "output_kind": "tags",
    },
}


def normalize_provider(value: Any, *, allow_empty: bool = False) -> str:
    """Return a registered provider id or fail closed."""

    provider = str(value or "").strip().lower()
    if not provider and allow_empty:
        return "openai_compatible"
    if provider not in _PROVIDER_TYPES:
        raise ValueError(f"不支持的打标接入类型：{provider or '空值'}")
    return provider


def get_provider_type(provider: Any) -> dict[str, Any]:
    """Return a copy of one provider descriptor."""

    provider_id = normalize_provider(provider)
    return dict(_PROVIDER_TYPES[provider_id])


def list_provider_types() -> list[dict[str, Any]]:
    """Return descriptors safe to expose to the browser."""

    return [dict(value) for value in _PROVIDER_TYPES.values()]


def is_implemented(provider: Any) -> bool:
    return bool(get_provider_type(provider).get("implemented"))


__all__ = [
    "get_provider_type",
    "is_implemented",
    "list_provider_types",
    "normalize_provider",
]
