"""Shared live-monitor and progress-event helpers for WebUI training services."""

from __future__ import annotations

import math
from pathlib import Path
import re
import time
from typing import Any

from web.services.training import progress_parser as _progress_parser
from web.services.training.storage import _read_json


def _live_metric_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return _progress_parser.live_metric_key(item)


def _progress_event_key(event: dict[str, Any]) -> tuple[Any, ...]:
    return _progress_parser.progress_event_key(event)


def _progress_event_wall_ts(event: dict[str, Any], task_dir: Path | None) -> float:
    started_at = None
    if task_dir is not None:
        meta = _read_json(task_dir / "meta.json")
        if isinstance(meta, dict):
            started_at = _float_or_none(meta.get("started_at"))
    return _progress_event_wall_ts_from_started_at(event, started_at)


def _progress_event_wall_ts_from_started_at(event: dict[str, Any], started_at: float | None) -> float:
    return _progress_parser.progress_event_wall_ts_from_started_at(event, started_at, now_fn=time.time)


def _metric_from_progress_jsonl_event(event: dict[str, Any], ts: float, *, rate: str = "") -> dict[str, Any] | None:
    return _progress_parser.metric_from_progress_jsonl_event(event, ts, rate=rate)


def _progress_event_loss(event: dict[str, Any]) -> float | None:
    return _progress_parser.progress_event_loss(event)


def _progress_event_lr(event: dict[str, Any]) -> float | None:
    return _progress_parser.progress_event_lr(event)


def _first_float_field(record: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    return _progress_parser.first_float_field(record, keys)


def _json_safe_training_payload(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: _json_safe_training_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_training_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_training_payload(item) for item in value]
    return value


def _first_record_separator(text: str) -> int | None:
    indexes = [idx for idx in (text.find("\n"), text.find("\r")) if idx >= 0]
    return min(indexes) if indexes else None


def _clean_output_record(text: str) -> str:
    text = text.replace("\x1b[?25l", "").replace("\x1b[?25h", "")
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    return text.strip()


def _extract_float_metric(text: str, names: tuple[str, ...]) -> float | None:
    return _progress_parser.extract_float_metric(text, names)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None
