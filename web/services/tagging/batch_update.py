"""Atomic optimistic updates of in-memory candidates, never caption files."""
from typing import Any


def update_candidates(job: dict[str, Any], updates: Any) -> int:
    if job["state"] not in {"completed", "partial", "failed", "canceled"}:
        raise RuntimeError("任务未结束，暂不能批量编辑候选")
    if not isinstance(updates, list) or not 1 <= len(updates) <= 500:
        raise ValueError("items 必须包含 1 至 500 个条目")
    by_id = {item["id"]: item for item in job["items"]}
    prepared = []
    seen = set()
    for update in updates:
        if not isinstance(update, dict):
            raise ValueError("候选条目必须是对象")
        key, text, expected = (update.get(name) for name in ("id", "proposed_caption", "expected_caption"))
        if not isinstance(key, str) or not key or key in seen or key not in by_id:
            raise ValueError("候选 ID 无效、重复或不存在")
        seen.add(key)
        if not isinstance(text, str) or not isinstance(expected, str) or len(text) > 100_000:
            raise ValueError("caption 必须是字符串且不超过 100000 个字符")
        if not text.strip():
            raise ValueError("批量操作不能清空候选 caption")
        if by_id[key]["proposed_caption"] != expected:
            raise RuntimeError("候选已改变，请核对服务器状态后再保存")
        prepared.append((by_id[key], text))
    # All checks precede mutation; no await means one event-loop transaction.
    for item, text in prepared:
        item["proposed_caption"] = text
        item["error"] = ""
        item["state"] = "ready"
    return len(prepared)
