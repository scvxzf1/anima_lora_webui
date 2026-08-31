"""Bounded, process-local event log for the tagging workbench."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import time
from typing import Any

DEFAULT_LOG_RETENTION_LINES = 200
MIN_LOG_RETENTION_LINES = 50
MAX_LOG_RETENTION_LINES = 5000


def normalize_log_retention(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = DEFAULT_LOG_RETENTION_LINES
    return max(MIN_LOG_RETENTION_LINES, min(MAX_LOG_RETENTION_LINES, number))


class TaggingMemoryLog:
    """Keep recent structured log entries in memory only."""

    def __init__(self, retention_lines: Any = DEFAULT_LOG_RETENTION_LINES):
        self._retention = normalize_log_retention(retention_lines)
        self._lines: deque[dict[str, Any]] = deque(maxlen=self._retention)
        self._sequence = 0

    @property
    def retention_lines(self) -> int:
        return self._retention

    def set_retention(self, value: Any) -> int:
        next_retention = normalize_log_retention(value)
        if next_retention != self._retention:
            self._lines = deque(self._lines, maxlen=next_retention)
            self._retention = next_retention
        return self._retention

    def append(
        self,
        message: str,
        *,
        level: str = "info",
        event: str = "",
        job_id: str = "",
        item_id: str = "",
    ) -> dict[str, Any]:
        self._sequence += 1
        timestamp = time.time()
        entry = {
            "sequence": self._sequence,
            "timestamp": timestamp,
            "timestamp_text": _format_timestamp(timestamp),
            "level": _normalize_level(level),
            "event": str(event or "").strip()[:80],
            "job_id": str(job_id or "").strip()[:64],
            "item_id": str(item_id or "").strip()[:64],
            "message": str(message or "").replace("\r", " ").replace("\n", " ").strip()[:2000],
        }
        self._lines.append(entry)
        return dict(entry)

    def snapshot(
        self,
        *,
        after: Any = 0,
        limit: Any = None,
        job_id: str = "",
    ) -> dict[str, Any]:
        try:
            after_sequence = max(0, int(after or 0))
        except (TypeError, ValueError):
            after_sequence = 0
        requested = self._retention if limit is None else normalize_log_retention(limit)
        job_filter = str(job_id or "").strip()
        buffered = list(self._lines)
        filtered = [
            entry
            for entry in buffered
            if entry["sequence"] > after_sequence
            and (not job_filter or entry["job_id"] == job_filter)
        ]
        returned = filtered[-requested:]
        oldest = buffered[0]["sequence"] if buffered else self._sequence + 1
        return {
            "ok": True,
            "lines": [dict(entry) for entry in returned],
            "returned": len(returned),
            "buffered": len(buffered),
            "retention_lines": self._retention,
            "last_sequence": self._sequence,
            "truncated": len(filtered) > len(returned) or (after_sequence > 0 and after_sequence < oldest - 1),
        }

    def clear(self) -> dict[str, Any]:
        self._lines.clear()
        return self.snapshot()


def _normalize_level(value: Any) -> str:
    level = str(value or "info").strip().lower()
    return level if level in {"debug", "info", "success", "warning", "error"} else "info"


def _format_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
