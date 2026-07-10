"""Shared constants and scalar helpers for WebUI training services."""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any

RUN_META_FILE = "run.meta.json"
HISTORY_ARTIFACT_FILES = {
    "config-snapshot": "config.snapshot.toml",
    "logs": "logs.jsonl",
    "metrics": "metrics.jsonl",
    "system": "system.jsonl",
}
HISTORY_RUNTIME_ARTIFACT_FIELDS = {
    "runtime-config": "runtime_config_file",
    "original-config": "original_config_file",
    "dataset-config": "dataset_config_file",
}
TRAINING_PROGRESS_LOG_RE = re.compile(
    r"(?:^|\r)steps:\s*\d+%\|[^|]*\|\s*(?P<cur>\d+)/(?P<tot>\d+)\s*\[",
    re.IGNORECASE,
)


def _safe_task_id(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip()).strip(".-")
    return clean[:120] or "task"


def _format_ts(ts: float | None) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")


def _clean_history_text(value: Any, *, max_len: int) -> str:
    text = re.sub(r"[\r\n\t]+", " ", str(value or "")).strip()
    text = re.sub(r"\s{2,}", " ", text)
    return text[:max_len]


def _int_or_none(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _positive_int_or_none(value: Any) -> int | None:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None
