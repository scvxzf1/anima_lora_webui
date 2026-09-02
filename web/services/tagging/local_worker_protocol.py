"""JSONL protocol helpers shared by the local tagging worker and client."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = 1
MAX_EVENT_BYTES = 4 * 1024 * 1024
MAX_TAGS = 20_000
MAX_RAW_SCORES = 20_000


def safe_error(value: Any) -> str:
    """Return a bounded, single-line error suitable for a public job record."""

    if isinstance(value, BaseException):
        text = str(value) or value.__class__.__name__
    else:
        text = str(value or "")
    return " ".join(text.split())[:500] or "本地打标 Worker 发生未知错误"


def sanitize_result(value: Any) -> dict[str, Any]:
    """Convert a provider result to JSON-safe primitives with bounded fields."""

    if not isinstance(value, Mapping):
        raise ValueError("本地 provider 返回了无效结果")
    result: dict[str, Any] = {}
    image = value.get("image")
    if image is not None and str(image).strip():
        result["image"] = str(Path(image))
    tags = value.get("tags")
    if isinstance(tags, (list, tuple)):
        result["tags"] = [
            str(tag).strip()[:512]
            for tag in tags[:MAX_TAGS]
            if str(tag).strip()
        ]
    caption = value.get("caption")
    if caption is not None:
        result["caption"] = str(caption)[:100_000]
    error = value.get("error")
    if error:
        result["error"] = safe_error(error)
    scores = value.get("raw_scores")
    if isinstance(scores, Mapping):
        cleaned: dict[str, float] = {}
        for key, score in list(scores.items())[:MAX_RAW_SCORES]:
            try:
                number = float(score)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                cleaned[str(key)[:512]] = number
        result["raw_scores"] = cleaned
    return result


def encode_event(event: Mapping[str, Any]) -> str:
    return json.dumps(dict(event), ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def decode_event(raw: bytes | str) -> dict[str, Any]:
    data = raw.encode("utf-8") if isinstance(raw, str) else raw
    if len(data) > MAX_EVENT_BYTES:
        raise ValueError("本地打标 Worker 事件超过大小上限")
    try:
        value = json.loads(data.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("本地打标 Worker 返回了无效 JSONL") from exc
    if not isinstance(value, dict):
        raise ValueError("本地打标 Worker 事件必须是对象")
    if value.get("version") != PROTOCOL_VERSION:
        raise ValueError("本地打标 Worker 协议版本不匹配")
    return value


__all__ = [
    "MAX_EVENT_BYTES",
    "PROTOCOL_VERSION",
    "decode_event",
    "encode_event",
    "safe_error",
    "sanitize_result",
]
