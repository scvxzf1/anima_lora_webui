"""Optimistic guard for the exact queue snapshot confirmed by a client."""

import hashlib
import json
from typing import Any


class QueueRevisionConflict(RuntimeError):
    pass


def queue_revision(snapshot: dict[str, Any]) -> str:
    encoded = json.dumps(snapshot, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def ensure_queue_revision(service, expected_revision: str | None) -> None:
    if expected_revision is None:
        return
    if not isinstance(expected_revision, str) or not expected_revision.strip():
        raise ValueError("缺少有效的队列快照版本")
    if service.get_queue_snapshot()["revision"] != expected_revision:
        raise QueueRevisionConflict("队列已发生变化，本次操作未执行。请刷新队列并重新确认范围。")
