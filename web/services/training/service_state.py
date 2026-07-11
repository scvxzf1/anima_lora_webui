"""Queue/history settings storage helpers shared by training service modules."""

from __future__ import annotations

from typing import Any

from web.services.training.common import _float_or_none, _format_ts, _safe_task_id
from web.services.training.constants import (
    QUEUE_CLEARABLE_STATE_LABELS,
    QUEUE_FAILURE_POLICIES,
    history_collections_file,
    queue_file,
)
from web.services.training.history_meta import _history_task_dir
from web.services.training.storage import _read_json_object, _write_json_atomic


def _queue_file():
    return queue_file()


def _history_collections_file():
    return history_collections_file()


def _queue_failure_policies() -> set[str]:
    return QUEUE_FAILURE_POLICIES


def _queue_clearable_state_labels() -> dict[str, str]:
    return QUEUE_CLEARABLE_STATE_LABELS


def _training_datetime():
    from datetime import datetime

    return datetime


def _load_training_queue_state() -> dict[str, Any]:
    """Load queue.json runtime state.

    Retry policy keys (auto_retry / max_attempts / retry_backoff_sec) are only
    normalized when present in the file. Missing keys stay absent so
    TrainingService can seed them from global training_policy defaults.
    """
    data = _read_training_queue_state()
    if not isinstance(data, dict):
        data = {}
    items = data.get("items")
    if not isinstance(items, list):
        data["items"] = []
    data["paused"] = bool(data.get("paused", False))
    data["failure_policy"] = _normalize_queue_failure_policy(data.get("failure_policy"))
    # Do not invent retry keys on load — absent means "use global policy seed".
    if "auto_retry" in data:
        data["auto_retry"] = _normalize_queue_auto_retry(data.get("auto_retry"))
    if "max_attempts" in data:
        data["max_attempts"] = _normalize_queue_max_attempts(data.get("max_attempts"))
    if "retry_backoff_sec" in data:
        data["retry_backoff_sec"] = _normalize_queue_retry_backoff(data.get("retry_backoff_sec"))
    return data


def _read_training_queue_state() -> dict[str, Any]:
    data = _read_json_object(_queue_file())
    if isinstance(data, dict):
        return data
    backup = _read_json_object(_queue_backup_file())
    if isinstance(backup, dict):
        try:
            _write_json_atomic(_queue_file(), backup)
        except Exception:
            pass
        return backup
    return {}


def _write_training_queue_state(payload: dict[str, Any]) -> None:
    _write_json_atomic(_queue_file(), payload)
    try:
        _write_json_atomic(_queue_backup_file(), payload)
    except Exception:
        pass


def _queue_backup_file():
    queue_file = _queue_file()
    return queue_file.with_name(queue_file.name + ".bak")


def _load_history_collection_settings() -> dict[str, Any]:
    data = _read_json_object(_history_collections_file()) or {}
    settings = _normalize_history_collection_settings(data)
    updated_at = _float_or_none(data.get("updated_at")) if isinstance(data, dict) else None
    updated_at = updated_at or 0.0
    settings["updated_at"] = updated_at
    settings["updated_at_text"] = _format_ts(updated_at) if updated_at else ""
    return {"ok": True, **settings}


def _normalize_history_collection_settings(payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}
    return {
        "collection_order": _normalize_unique_string_list(payload.get("collection_order")),
        "config_group_order": _normalize_config_group_order(payload.get("config_group_order")),
    }


def _normalize_unique_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        out.append(text)
        seen.add(text)
    return out


def _normalize_config_group_order(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, list[str]] = {}
    for raw_key, raw_order in value.items():
        key = str(raw_key or "").strip()
        if not key:
            continue
        order = _normalize_unique_string_list(raw_order)
        if order:
            out[key] = order
    return out


def _normalize_queue_failure_policy(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in _queue_failure_policies() else "pause"


def _normalize_queue_auto_retry(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _normalize_queue_max_attempts(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = 1
    return max(1, min(10, number))


def _normalize_queue_retry_backoff(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    if number < 0:
        number = 0.0
    return min(3600.0, number)



def resolve_item_retry_policy(
    item: dict[str, Any] | None,
    *,
    queue_auto_retry: bool,
    queue_max_attempts: int,
    queue_retry_backoff_sec: float,
) -> dict[str, Any]:
    """Resolve effective retry policy with item > queue precedence.

    Only keys explicitly present on the item (and not None) override the
    corresponding queue runtime value. Each key may override independently.
    """
    src = item if isinstance(item, dict) else {}
    if "auto_retry" in src and src.get("auto_retry") is not None:
        auto_retry = _normalize_queue_auto_retry(src.get("auto_retry"))
    else:
        auto_retry = _normalize_queue_auto_retry(queue_auto_retry)
    if "max_attempts" in src and src.get("max_attempts") is not None:
        max_attempts = _normalize_queue_max_attempts(src.get("max_attempts"))
    else:
        max_attempts = _normalize_queue_max_attempts(queue_max_attempts)
    if "retry_backoff_sec" in src and src.get("retry_backoff_sec") is not None:
        retry_backoff_sec = _normalize_queue_retry_backoff(src.get("retry_backoff_sec"))
    else:
        retry_backoff_sec = _normalize_queue_retry_backoff(queue_retry_backoff_sec)
    return {
        "auto_retry": bool(auto_retry),
        "max_attempts": int(max_attempts),
        "retry_backoff_sec": float(retry_backoff_sec),
    }


def _queue_clearable_state_label(states: set[str]) -> str:
    clean = {str(state or "").strip() for state in states}
    labels = _queue_clearable_state_labels()
    if clean == {"done"}:
        return labels["done"]
    if clean == {"canceled"}:
        return labels["canceled"]
    return "已结束"


def _new_queue_item_id(kind: str, methods_subdir: str, variant: str) -> str:
    raw = _training_datetime().now().strftime("%Y%m%d-%H%M%S") + f"-queue-{kind}-{methods_subdir}-{variant}"
    base = _safe_task_id(raw)
    existing = {
        str(item.get("id") or "")
        for item in _load_training_queue_state().get("items", [])
        if isinstance(item, dict)
    }
    if base not in existing:
        return base
    suffix = 2
    while f"{base}-{suffix}" in existing:
        suffix += 1
    return f"{base}-{suffix}"


def _normalize_history_task_ids(value: Any) -> list[str]:
    raw_items = value if isinstance(value, list) else [value]
    out: list[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        task_id = str(raw or "").strip()
        if not task_id or task_id in seen:
            continue
        _history_task_dir(task_id)
        out.append(task_id)
        seen.add(task_id)
    return out
