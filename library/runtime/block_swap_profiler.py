"""Append-only profiling helpers for runtime block swapping."""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Optional, Union


class BlockSwapProfiler:
    """Append-only JSONL writer for block-swap transfer/wait observations."""

    def __init__(self, path: str):
        self.path = str(path)
        self._lock = threading.Lock()
        self._seq = 0
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)

    def write(self, event: dict[str, Any]) -> None:
        self.write_many([event])

    def write_many(self, events: list[dict[str, Any]]) -> None:
        if not events:
            return
        try:
            with self._lock:
                with open(self.path, "a", encoding="utf-8") as f:
                    for event in events:
                        self._seq += 1
                        payload = {"seq": self._seq, **event}
                        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            return


def _resolve_profiler(profile_jsonl: Optional[Union[str, BlockSwapProfiler]]):
    if isinstance(profile_jsonl, BlockSwapProfiler):
        return profile_jsonl
    if profile_jsonl is None:
        return None
    path = str(profile_jsonl).strip()
    if not path or path.lower() in {"off", "none", "false", "0"}:
        return None
    return BlockSwapProfiler(path)
