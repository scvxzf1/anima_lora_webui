"""Training subprocess management and output parsing."""

from __future__ import annotations

# Extracted training.* modules lazily bind selected facade globals so legacy
# monkeypatch paths and helper imports keep working during the split.
# ruff: noqa: F401

import asyncio
import re
from collections import deque
from pathlib import Path
from typing import Any

from aiohttp import web

from library.env import load_dotenv
from web.services.training.constants import (
    MAX_LOG_RECORDS,
    PROGRESS_RATE_SAMPLE_WINDOW,
)
from web.services.training.gpu_async import list_available_gpus as _list_available_gpus
from web.services.training.live_utils import _json_safe_training_payload

load_dotenv()

METRIC_RE = re.compile(
    r"(?:loss[:/]?\s*(?P<loss>[\d.]+))"
    r"|(?:lr[:/]?\s*(?P<lr>[\d.eE\-+]+))"
    r"|(?:norm[:/]?\s*(?P<norm>[\d.]+))"
)

def reload_runtime_storage_state(service: TrainingService | None) -> None:
    """Refresh in-memory queue view after config-root changes."""
    if service is None:
        return
    service._queue = _load_training_queue_state()
    service._queue_paused = bool(service._queue.get("paused", False))
    service._queue_failure_policy = _normalize_queue_failure_policy(service._queue.get("failure_policy"))
    service._queue_auto_retry = bool(service._queue.get("auto_retry", False))
    service._queue_max_attempts = int(service._queue.get("max_attempts", 1) or 1)
    service._queue_retry_backoff_sec = float(service._queue.get("retry_backoff_sec", 0.0) or 0.0)

class TrainingService:
    def __init__(self, app: web.Application):
            self.app = app
            self.process: asyncio.subprocess.Process | None = None
            self.status: str = "idle"
            self.current_variant: str = ""
            self.current_preset: str = ""
            self.current_methods_subdir: str = "gui-methods"
            self.current_output_dir: str = ""
            self.current_sample_dir: str = ""
            self.current_sample_config: dict[str, Any] = _default_sample_config()
            self.current_runtime_info: dict[str, str] = {}
            self.current_job: str = ""
            self.current_gpu_whitelist: list[int] = []
            self.current_task_id: str = ""
            self.current_task_dir: Path | None = None
            self.current_command: list[str] = []
            self._stop_requested = False
            self._pending_train_after_preprocess: dict[str, Any] | None = None
            self._ws_clients: set[web.WebSocketResponse] = set()
            self._anchor: tuple[float, int] | None = None
            self._stdout_rate_last: tuple[float, int] | None = None
            self._stdout_rate_samples: deque[float] = deque(maxlen=PROGRESS_RATE_SAMPLE_WINDOW)
            self._structured_rate_last: tuple[float, int] | None = None
            self._structured_rate_samples: deque[float] = deque(maxlen=PROGRESS_RATE_SAMPLE_WINDOW)
            self._metrics_history: list[dict[str, Any]] = []
            self._latest_progress: dict[str, Any] | None = None
            self._latest_system_stats: dict[str, Any] | None = None
            self._last_output_at: float | None = None
            self._last_log_line: str = ""
            self._last_lr_log_text: str = ""
            self._log_records: deque[dict[str, Any]] = deque(maxlen=MAX_LOG_RECORDS)
            self._next_log_id = 1
            self._current_history_log_count = 0
            self._metric_seen_keys: set[tuple[Any, ...]] = set()
            self._progress_jsonl_path: Path | None = None
            self._progress_jsonl_offset = 0
            self._progress_jsonl_seen: set[tuple[Any, ...]] = set()
            self._progress_jsonl_lock: asyncio.Lock | None = None
            self._progress_total_steps: int | None = None
            self._detected_error_hint: str = ""
            self._queue: dict[str, Any] = _load_training_queue_state()
            self._queue_paused: bool = bool(self._queue.get("paused", False))
            self._queue_auto_retry: bool = bool(self._queue.get("auto_retry", False))
            self._queue_max_attempts: int = int(self._queue.get("max_attempts", 1) or 1)
            self._queue_retry_backoff_sec: float = float(self._queue.get("retry_backoff_sec", 0.0) or 0.0)
            self._queue_failure_policy: str = _normalize_queue_failure_policy(self._queue.get("failure_policy"))
            self._current_queue_item_id: str = ""
            self._queue_launching_item_id: str = ""
            self._queue_dispatch_task: asyncio.Task | None = None
            self._run_generation: int = 0
            self._stopping: bool = False
            self._shutting_down: bool = False
            self._job_tasks: dict[int, set[asyncio.Task[Any]]] = {}
            self._output_task: asyncio.Task[Any] | None = None
            self._output_task_generation: int = 0
            try:
                from web.services.training.constants import apply_training_policy_to_facade
                policy = apply_training_policy_to_facade()
                # Seed in-memory defaults from global training_policy only when
                # queue.json did not specify those runtime keys (policy default
                # vs queue runtime override). Present keys always win.
                if "auto_retry" not in (self._queue or {}):
                    self._queue_auto_retry = bool(policy.get("auto_retry", False))
                if "max_attempts" not in (self._queue or {}):
                    self._queue_max_attempts = int(policy.get("max_attempts") or 1)
                if "retry_backoff_sec" not in (self._queue or {}):
                    self._queue_retry_backoff_sec = float(policy.get("retry_backoff_sec") or 0.0)
            except Exception:
                pass
            self._queue_dispatch_wake_handle = None
            self._launch_lock = asyncio.Lock()
            _mark_orphaned_running_history_tasks()
            self._repair_queue_on_startup()

    def subscribe(self, ws: web.WebSocketResponse):
            self._ws_clients.add(ws)

    def unsubscribe(self, ws: web.WebSocketResponse):
            self._ws_clients.discard(ws)

    def get_metrics_history(self) -> list[dict]:
            return [_json_safe_training_payload(item) for item in self._metrics_history[-500:]]

    def get_log_records(self, after: int = 0, limit: int = 1000) -> list[dict[str, Any]]:
            limit = max(1, min(limit, MAX_LOG_RECORDS))
            records = [record for record in self._log_records if record["id"] > after]
            return records[-limit:]

    async def list_gpus(self) -> list[dict[str, Any]]:
            return await _list_available_gpus()

    # Domain methods live in web.services.training.{queue,history,launcher,live_monitor}.
    # This facade keeps a single dynamic dispatch surface instead of dozens of 3-line wrappers.
    _ASYNC_DELEGATES = {
        '_broadcast': 'live_monitor',
        '_broadcast_progress': 'live_monitor',
        '_broadcast_queue': 'queue',
        '_dispatch_queue': 'queue',
        '_drain_output_buffer': 'live_monitor',
        '_handle_output_record': 'live_monitor',
        '_handle_progress_jsonl_event': 'live_monitor',
        '_ingest_progress_jsonl': 'live_monitor',
        '_launch_job': 'launcher',
        '_stop_unlocked': 'launcher',
        '_maybe_note_error_hint': 'live_monitor',
        '_monitor_system': 'live_monitor',
        '_read_output': 'live_monitor',
        '_record_metric': 'live_monitor',
        '_start_pending_training': 'launcher',
        '_start_preprocess_unlocked': 'launcher',
        '_start_queue_item': 'queue',
        '_start_unlocked': 'launcher',
        '_tail_progress_jsonl': 'live_monitor',
        'abort_queue_after_current': 'queue',
        'cancel_all_queue_items': 'queue',
        'cancel_queue_item': 'queue',
        'cancel_waiting_queue_items': 'queue',
        'clear_canceled_queue_items': 'queue',
        'clear_completed_queue_items': 'queue',
        'clear_finished_queue_items': 'queue',
        'clear_queue_items_by_state': 'queue',
        'enqueue_resume_from_history_task': 'queue',
        'enqueue_training': 'queue',
        'enqueue_training_batch': 'queue',
        'force_abort_queue': 'queue',
        'move_queue_item': 'queue',
        'resume_from_history_task': 'history',
        'retry_queue_item': 'queue',
        'set_queue_paused': 'queue',
        'set_queue_settings': 'queue',
        'start': 'launcher',
        'start_preprocess': 'launcher',
        'start_queue_on_startup': 'queue',
        'stop': 'launcher',
        'shutdown': 'launcher',
    }
    _SYNC_DELEGATES = {
        '_append_history_jsonl': 'history',
        '_attach_history_task_to_queue_item': 'queue',
        '_batch_delete_history_tasks': 'history',
        '_build_resume_payload': 'history',
        '_clone_queue_item_for_retry': 'queue',
        '_maybe_auto_retry': 'queue',
        '_compact_queue': 'queue',
        '_compute_rate': 'live_monitor',
        '_compute_structured_rate': 'live_monitor',
        '_ensure_launch_allowed': 'launcher',
        '_extract_metrics_from_log': 'live_monitor',
        '_extract_metrics_from_tqdm': 'live_monitor',
        '_find_queue_item': 'queue',
        '_finish_history_task': 'history',
        '_normalize_queue': 'queue',
        '_pause_queue_after_failure': 'queue',
        '_plan_history_delete': 'history',
        '_queue_history_meta': 'queue',
        '_queue_item_runtime': 'queue',
        '_queue_items': 'queue',
        '_remember_log': 'live_monitor',
        '_remember_lr_change_log': 'live_monitor',
        '_repair_queue_on_startup': 'queue',
        '_reserve_history_task_dir': 'history',
        '_reset_metric_runtime_state': 'live_monitor',
        '_reset_progress_rate_state': 'live_monitor',
        '_save_queue': 'queue',
        '_schedule_queue_dispatch': 'queue',
        '_start_history_task': 'history',
        '_update_queue_item': 'queue',
        '_write_terminal': 'launcher',
        'batch_update_history_tasks': 'history',
        'delete_history_task': 'history',
        'find_history_log_match': 'history',
        'get_config_group_timeline': 'history',
        'get_history_artifact_path': 'history',
        'get_history_collection_settings': 'history',
        'get_history_log_page': 'history',
        'get_history_log_path': 'history',
        'get_history_task': 'history',
        'get_history_task_summary': 'history',
        'get_queue_snapshot': 'queue',
        'get_resume_options': 'history',
        'get_status_snapshot': 'live_monitor',
        'list_history_tasks': 'history',
        'save_history_collection_settings': 'history',
        'update_history_task': 'history',
    }

    def __getattr__(self, name: str):
        module_name = self._ASYNC_DELEGATES.get(name)
        if module_name is not None:
            import importlib

            impl_module = importlib.import_module(f"web.services.training.{module_name}")
            impl = getattr(impl_module, name)

            async def _async_bound(*args, **kwargs):
                return await impl(self, *args, **kwargs)

            _async_bound.__name__ = name
            _async_bound.__qualname__ = f"TrainingService.{name}"
            object.__setattr__(self, name, _async_bound)
            return _async_bound

        module_name = self._SYNC_DELEGATES.get(name)
        if module_name is not None:
            import importlib

            impl_module = importlib.import_module(f"web.services.training.{module_name}")
            impl = getattr(impl_module, name)

            def _sync_bound(*args, **kwargs):
                return impl(self, *args, **kwargs)

            _sync_bound.__name__ = name
            _sync_bound.__qualname__ = f"TrainingService.{name}"
            object.__setattr__(self, name, _sync_bound)
            return _sync_bound

        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")


# Lazy compatibility surface: keep historical training_service.<name> imports without
# eagerly binding every helper into this facade module.
from web.services.training import exports as _training_exports
from web.services.training import facade_compat as _facade_compat

# Symbols referenced as bare names inside this facade must stay eagerly bound.
_default_sample_config = _training_exports._default_sample_config
_load_training_queue_state = _training_exports._load_training_queue_state
_mark_orphaned_running_history_tasks = _training_exports._mark_orphaned_running_history_tasks
_normalize_queue_failure_policy = _training_exports._normalize_queue_failure_policy


def __getattr__(name: str):
    if name in _training_exports.__all__:
        value = getattr(_training_exports, name)
        globals()[name] = value
        return value
    if name in _facade_compat.COMPAT_EXPORTS:
        value = _facade_compat.resolve_compat_export(name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_training_exports.__all__) | set(_facade_compat.COMPAT_EXPORTS))
