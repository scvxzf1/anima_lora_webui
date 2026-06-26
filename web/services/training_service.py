"""Training subprocess management and output parsing."""

from __future__ import annotations

# Extracted training.* modules lazily bind selected facade globals so legacy
# monkeypatch paths and helper imports keep working during the split.
# ruff: noqa: F401

import asyncio
from collections import deque
from datetime import datetime
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import psutil
from aiohttp import web
import toml

from library.env import load_dotenv, get_training_history_root, get_training_queue_root
from library.preprocess.captions import (
    CAPTION_SOURCE_CAPTIONS_JSON,
    CAPTIONS_JSON_FILE,
    normalize_caption_source_mode,
)
from library.runtime.launch import accelerate_training_command_prefix
from web.services.continue_lora_service import (
    inspect_continue_lora_weight as _inspect_continue_lora_weight,
)
from web.services.config_service import (
    NL_TAG_MIX_CLASSIFICATION_METHOD,
    _build_dataset_config_doc,
    _classify_nl_tag_caption_text,
    _dataset_rows_for_estimate,
    _nl_tag_mix_caption_source,
    _nl_tag_mix_image_files,
    _normalize_nl_tag_mix,
    _normalize_path_pattern,
    _normalize_trigger_clone,
    apply_auto_data_dirs,
    apply_global_model_path_defaults,
    load_merged_config,
    preflight_training_config,
    training_sample_sampler_status,
)
from web.services.settings_service import display_path as _display_settings_path
from web.services.settings_service import resolve_output_root
from web.services.training import progress_parser as _progress_parser
from web.services.training.gpu import (
    aggregate_gpu_stats_rows as _aggregate_gpu_stats_rows_impl,
    apply_gpu_whitelist as _apply_gpu_whitelist_impl,
    get_gpu_stats as _get_gpu_stats_impl,
    list_available_gpus as _list_available_gpus_impl,
    normalize_gpu_whitelist as _normalize_gpu_whitelist_impl,
    parse_gpu_stats_rows as _parse_gpu_stats_rows_impl,
)

ROOT = Path(__file__).resolve().parents[2]
HISTORY_DIR = get_training_history_root()
HISTORY_COLLECTIONS_FILE = HISTORY_DIR / "collections.json"
QUEUE_DIR = get_training_queue_root()
QUEUE_FILE = QUEUE_DIR / "queue.json"
RUN_META_FILE = "run.meta.json"
OUTPUT_READ_SIZE = 4096
MAX_LOG_RECORDS = 3000
MAX_HISTORY_ITEMS = 100
MAX_TIMELINE_LOG_RECORDS = 20000
MAX_TIMELINE_METRIC_RECORDS = 20000
MAX_HISTORY_DETAIL_LOG_RECORDS = 5000
MAX_HISTORY_DETAIL_SYSTEM_RECORDS = 1000
MAX_QUEUE_ITEMS = 200
PROGRESS_RATE_SAMPLE_WINDOW = 9
HISTORY_AVERAGE_SPEED_VERSION = 1
QUEUE_FAILURE_POLICIES = {"pause", "continue"}
QUEUE_TERMINAL_STATES = {"done", "error", "canceled"}
# 队列批量清理只移除 queue.json 里的列表记录，保留 error 方便确认后重试或手动删除。
QUEUE_CLEARABLE_STATES = {"done", "canceled"}
QUEUE_CLEARABLE_STATE_LABELS = {"done": "已完成", "canceled": "已取消"}
DATASET_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})
DATASET_CAPTION_EXTS = (".txt", ".json", ".caption")
RUNTIME_META_KEYS = (
    "run_dir",
    "runtime_config_file",
    "original_config_file",
    "dataset_config_file",
    "model_cache_dir",
    "dataset_cache_dir",
    "training_output_dir",
    "logs_dir",
    "history_source_config_file",
)
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

TQDM_RE = re.compile(
    r"^(?P<label>.*?):?\s*(?P<pct>\d+)%\|[^|]*\|\s*(?P<cur>\d+)/(?P<tot>\d+)"
    r"(?:[^\[]*\[[^\]]*?(?P<rate>[\d.]+)(?P<unit>it/s|s/it)[^\]]*\])?"
)
TRAINING_PROGRESS_LOG_RE = re.compile(
    r"(?:^|\r)steps:\s*\d+%\|[^|]*\|\s*(?P<cur>\d+)/(?P<tot>\d+)\s*\[",
    re.IGNORECASE,
)

load_dotenv()

METRIC_RE = re.compile(
    r"(?:loss[:/]?\s*(?P<loss>[\d.]+))"
    r"|(?:lr[:/]?\s*(?P<lr>[\d.eE\-+]+))"
    r"|(?:norm[:/]?\s*(?P<norm>[\d.]+))"
)

CUDA_OOM_RE = re.compile(
    r"(?:"
    r"cuda\s+out\s+of\s+memory"
    r"|torch\.outofmemoryerror"
    r"|outofmemoryerror:\s*cuda"
    r"|cublas_status_alloc_failed"
    r"|cudnn_status_alloc_failed"
    r")",
    re.IGNORECASE,
)

OOM_HINT = "大概率爆显存"


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
        self._metric_seen_keys: set[tuple[Any, ...]] = set()
        self._progress_jsonl_path: Path | None = None
        self._progress_jsonl_offset = 0
        self._progress_jsonl_seen: set[tuple[Any, ...]] = set()
        self._progress_jsonl_lock: asyncio.Lock | None = None
        self._progress_total_steps: int | None = None
        self._detected_error_hint: str = ""
        self._queue: dict[str, Any] = _load_training_queue_state()
        self._queue_paused: bool = bool(self._queue.get("paused", False))
        self._queue_failure_policy: str = _normalize_queue_failure_policy(self._queue.get("failure_policy"))
        self._current_queue_item_id: str = ""
        self._queue_launching_item_id: str = ""
        self._queue_dispatch_task: asyncio.Task | None = None
        self._launch_lock = asyncio.Lock()
        _mark_orphaned_running_history_tasks()
        self._repair_queue_on_startup()

    async def start_queue_on_startup(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return await _impl.start_queue_on_startup(self, *args, **kwargs)

    async def start(self, *args, **kwargs):
        from web.services.training import launcher as _impl
        return await _impl.start(self, *args, **kwargs)

    async def _start_unlocked(self, *args, **kwargs):
        from web.services.training import launcher as _impl
        return await _impl._start_unlocked(self, *args, **kwargs)

    async def resume_from_history_task(self, *args, **kwargs):
        from web.services.training import history as _impl
        return await _impl.resume_from_history_task(self, *args, **kwargs)

    def _build_resume_payload(self, *args, **kwargs):
        from web.services.training import history as _impl
        return _impl._build_resume_payload(self, *args, **kwargs)

    async def start_preprocess(self, *args, **kwargs):
        from web.services.training import launcher as _impl
        return await _impl.start_preprocess(self, *args, **kwargs)

    async def _start_preprocess_unlocked(self, *args, **kwargs):
        from web.services.training import launcher as _impl
        return await _impl._start_preprocess_unlocked(self, *args, **kwargs)

    async def _launch_job(self, *args, **kwargs):
        from web.services.training import launcher as _impl
        return await _impl._launch_job(self, *args, **kwargs)

    async def stop(self, *args, **kwargs):
        from web.services.training import launcher as _impl
        return await _impl.stop(self, *args, **kwargs)

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

    def get_queue_snapshot(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return _impl.get_queue_snapshot(self, *args, **kwargs)

    async def enqueue_training(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return await _impl.enqueue_training(self, *args, **kwargs)

    async def enqueue_training_batch(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return await _impl.enqueue_training_batch(self, *args, **kwargs)

    async def enqueue_resume_from_history_task(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return await _impl.enqueue_resume_from_history_task(self, *args, **kwargs)

    async def move_queue_item(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return await _impl.move_queue_item(self, *args, **kwargs)

    async def cancel_queue_item(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return await _impl.cancel_queue_item(self, *args, **kwargs)

    async def retry_queue_item(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return await _impl.retry_queue_item(self, *args, **kwargs)

    async def cancel_waiting_queue_items(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return await _impl.cancel_waiting_queue_items(self, *args, **kwargs)

    async def cancel_all_queue_items(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return await _impl.cancel_all_queue_items(self, *args, **kwargs)

    async def abort_queue_after_current(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return await _impl.abort_queue_after_current(self, *args, **kwargs)

    async def force_abort_queue(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return await _impl.force_abort_queue(self, *args, **kwargs)

    async def clear_finished_queue_items(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return await _impl.clear_finished_queue_items(self, *args, **kwargs)

    async def clear_completed_queue_items(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return await _impl.clear_completed_queue_items(self, *args, **kwargs)

    async def clear_canceled_queue_items(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return await _impl.clear_canceled_queue_items(self, *args, **kwargs)

    async def clear_queue_items_by_state(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return await _impl.clear_queue_items_by_state(self, *args, **kwargs)

    async def set_queue_settings(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return await _impl.set_queue_settings(self, *args, **kwargs)

    async def set_queue_paused(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return await _impl.set_queue_paused(self, *args, **kwargs)

    def _clone_queue_item_for_retry(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return _impl._clone_queue_item_for_retry(self, *args, **kwargs)

    def list_history_tasks(self, *args, **kwargs):
        from web.services.training import history as _impl
        return _impl.list_history_tasks(self, *args, **kwargs)

    def get_history_task(self, *args, **kwargs):
        from web.services.training import history as _impl
        return _impl.get_history_task(self, *args, **kwargs)

    def get_history_log_path(self, *args, **kwargs):
        from web.services.training import history as _impl
        return _impl.get_history_log_path(self, *args, **kwargs)

    def get_history_artifact_path(self, *args, **kwargs):
        from web.services.training import history as _impl
        return _impl.get_history_artifact_path(self, *args, **kwargs)

    def get_config_group_timeline(self, *args, **kwargs):
        from web.services.training import history as _impl
        return _impl.get_config_group_timeline(self, *args, **kwargs)

    def get_history_collection_settings(self, *args, **kwargs):
        from web.services.training import history as _impl
        return _impl.get_history_collection_settings(self, *args, **kwargs)

    def save_history_collection_settings(self, *args, **kwargs):
        from web.services.training import history as _impl
        return _impl.save_history_collection_settings(self, *args, **kwargs)

    def get_resume_options(self, *args, **kwargs):
        from web.services.training import history as _impl
        return _impl.get_resume_options(self, *args, **kwargs)

    def update_history_task(self, *args, **kwargs):
        from web.services.training import history as _impl
        return _impl.update_history_task(self, *args, **kwargs)

    def batch_update_history_tasks(self, *args, **kwargs):
        from web.services.training import history as _impl
        return _impl.batch_update_history_tasks(self, *args, **kwargs)

    def delete_history_task(self, *args, **kwargs):
        from web.services.training import history as _impl
        return _impl.delete_history_task(self, *args, **kwargs)

    def _batch_delete_history_tasks(self, *args, **kwargs):
        from web.services.training import history as _impl
        return _impl._batch_delete_history_tasks(self, *args, **kwargs)

    def _plan_history_delete(self, *args, **kwargs):
        from web.services.training import history as _impl
        return _impl._plan_history_delete(self, *args, **kwargs)

    def get_status_snapshot(self, *args, **kwargs):
        from web.services.training import live_monitor as _impl
        return _impl.get_status_snapshot(self, *args, **kwargs)

    async def _read_output(self, *args, **kwargs):
        from web.services.training import live_monitor as _impl
        return await _impl._read_output(self, *args, **kwargs)

    async def _start_pending_training(self, *args, **kwargs):
        from web.services.training import launcher as _impl
        return await _impl._start_pending_training(self, *args, **kwargs)

    def _repair_queue_on_startup(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return _impl._repair_queue_on_startup(self, *args, **kwargs)

    def _normalize_queue(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return _impl._normalize_queue(self, *args, **kwargs)

    def _queue_items(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return _impl._queue_items(self, *args, **kwargs)

    def _find_queue_item(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return _impl._find_queue_item(self, *args, **kwargs)

    def _update_queue_item(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return _impl._update_queue_item(self, *args, **kwargs)

    def _attach_history_task_to_queue_item(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return _impl._attach_history_task_to_queue_item(self, *args, **kwargs)

    def _queue_history_meta(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return _impl._queue_history_meta(self, *args, **kwargs)

    def _compact_queue(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return _impl._compact_queue(self, *args, **kwargs)

    def _save_queue(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return _impl._save_queue(self, *args, **kwargs)

    async def _broadcast_queue(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return await _impl._broadcast_queue(self, *args, **kwargs)

    def _schedule_queue_dispatch(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return _impl._schedule_queue_dispatch(self, *args, **kwargs)

    async def _dispatch_queue(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return await _impl._dispatch_queue(self, *args, **kwargs)

    def _pause_queue_after_failure(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return _impl._pause_queue_after_failure(self, *args, **kwargs)

    def _ensure_launch_allowed(self, *args, **kwargs):
        from web.services.training import launcher as _impl
        return _impl._ensure_launch_allowed(self, *args, **kwargs)

    async def _start_queue_item(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return await _impl._start_queue_item(self, *args, **kwargs)

    def _queue_item_runtime(self, *args, **kwargs):
        from web.services.training import queue as _impl
        return _impl._queue_item_runtime(self, *args, **kwargs)

    async def _drain_output_buffer(self, *args, **kwargs):
        from web.services.training import live_monitor as _impl
        return await _impl._drain_output_buffer(self, *args, **kwargs)

    async def _handle_output_record(self, *args, **kwargs):
        from web.services.training import live_monitor as _impl
        return await _impl._handle_output_record(self, *args, **kwargs)

    async def _record_metric(self, *args, **kwargs):
        from web.services.training import live_monitor as _impl
        return await _impl._record_metric(self, *args, **kwargs)

    def _reset_metric_runtime_state(self, *args, **kwargs):
        from web.services.training import live_monitor as _impl
        return _impl._reset_metric_runtime_state(self, *args, **kwargs)

    def _reset_progress_rate_state(self, *args, **kwargs):
        from web.services.training import live_monitor as _impl
        return _impl._reset_progress_rate_state(self, *args, **kwargs)

    def _remember_lr_change_log(self, *args, **kwargs):
        from web.services.training import live_monitor as _impl
        return _impl._remember_lr_change_log(self, *args, **kwargs)

    async def _tail_progress_jsonl(self, *args, **kwargs):
        from web.services.training import live_monitor as _impl
        return await _impl._tail_progress_jsonl(self, *args, **kwargs)

    async def _ingest_progress_jsonl(self, *args, **kwargs):
        from web.services.training import live_monitor as _impl
        return await _impl._ingest_progress_jsonl(self, *args, **kwargs)

    async def _handle_progress_jsonl_event(self, *args, **kwargs):
        from web.services.training import live_monitor as _impl
        return await _impl._handle_progress_jsonl_event(self, *args, **kwargs)

    async def _maybe_note_error_hint(self, *args, **kwargs):
        from web.services.training import live_monitor as _impl
        return await _impl._maybe_note_error_hint(self, *args, **kwargs)

    def _remember_log(self, *args, **kwargs):
        from web.services.training import live_monitor as _impl
        return _impl._remember_log(self, *args, **kwargs)

    def _reserve_history_task_dir(self, *args, **kwargs):
        from web.services.training import history as _impl
        return _impl._reserve_history_task_dir(self, *args, **kwargs)

    def _start_history_task(self, *args, **kwargs):
        from web.services.training import history as _impl
        return _impl._start_history_task(self, *args, **kwargs)

    def _finish_history_task(self, *args, **kwargs):
        from web.services.training import history as _impl
        return _impl._finish_history_task(self, *args, **kwargs)

    def _append_history_jsonl(self, *args, **kwargs):
        from web.services.training import history as _impl
        return _impl._append_history_jsonl(self, *args, **kwargs)

    def _write_terminal(self, *args, **kwargs):
        from web.services.training import launcher as _impl
        return _impl._write_terminal(self, *args, **kwargs)

    def _compute_rate(self, *args, **kwargs):
        from web.services.training import live_monitor as _impl
        return _impl._compute_rate(self, *args, **kwargs)

    def _compute_structured_rate(self, *args, **kwargs):
        from web.services.training import live_monitor as _impl
        return _impl._compute_structured_rate(self, *args, **kwargs)

    def _extract_metrics_from_tqdm(self, *args, **kwargs):
        from web.services.training import live_monitor as _impl
        return _impl._extract_metrics_from_tqdm(self, *args, **kwargs)

    def _extract_metrics_from_log(self, *args, **kwargs):
        from web.services.training import live_monitor as _impl
        return _impl._extract_metrics_from_log(self, *args, **kwargs)

    async def _monitor_system(self, *args, **kwargs):
        from web.services.training import live_monitor as _impl
        return await _impl._monitor_system(self, *args, **kwargs)

    async def _broadcast_progress(self, *args, **kwargs):
        from web.services.training import live_monitor as _impl
        return await _impl._broadcast_progress(self, *args, **kwargs)

    async def _broadcast(self, *args, **kwargs):
        from web.services.training import live_monitor as _impl
        return await _impl._broadcast(self, *args, **kwargs)


async def _get_gpu_stats(gpu_whitelist: list[int] | None = None) -> dict:
    return await _get_gpu_stats_impl(
        gpu_whitelist,
        create_subprocess_exec=asyncio.create_subprocess_exec,
        stdout_pipe=asyncio.subprocess.PIPE,
        stderr_devnull=asyncio.subprocess.DEVNULL,
    )


def _parse_gpu_stats_rows(text: str) -> list[dict[str, int]]:
    return _parse_gpu_stats_rows_impl(text)


def _aggregate_gpu_stats_rows(rows: list[dict[str, int]]) -> dict[str, Any]:
    return _aggregate_gpu_stats_rows_impl(rows)


async def _list_available_gpus() -> list[dict[str, Any]]:
    return await _list_available_gpus_impl(
        create_subprocess_exec=asyncio.create_subprocess_exec,
        stdout_pipe=asyncio.subprocess.PIPE,
        stderr_devnull=asyncio.subprocess.DEVNULL,
    )


def _normalize_gpu_whitelist(value: Any) -> list[int]:
    return _normalize_gpu_whitelist_impl(value)


def inspect_continue_lora_weight(
    path: str,
    *,
    variant: str = "lora",
    preset: str = "default",
    methods_subdir: str = "gui-methods",
    config_file: str | None = None,
) -> dict[str, Any]:
    cfg, config_error = _continue_lora_inspection_config(
        variant=variant,
        preset=preset,
        methods_subdir=methods_subdir,
        config_file=config_file,
    )
    return _inspect_continue_lora_weight(
        path,
        variant=variant,
        preset=preset,
        methods_subdir=methods_subdir,
        cfg=cfg,
        config_error=config_error,
        root=ROOT,
    )


def _continue_lora_inspection_config(
    *,
    variant: str,
    preset: str,
    methods_subdir: str,
    config_file: str | None,
) -> tuple[dict[str, Any] | None, Exception | None]:
    cfg = _load_config_file_config(config_file) if config_file else {}
    if cfg:
        return cfg, None
    try:
        return load_merged_config(variant, preset, methods_subdir), None
    except Exception as exc:
        return None, exc


def _normalize_continue_lora_info(
    value: Any,
    *,
    variant: str,
    preset: str,
    methods_subdir: str,
    config_file: str | None = None,
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    raw_path = str(
        value.get("continue_from_weight_abs_path")
        or value.get("abs_path")
        or value.get("path")
        or ""
    ).strip()
    if not raw_path:
        return None
    inspected = inspect_continue_lora_weight(
        raw_path,
        variant=variant,
        preset=preset,
        methods_subdir=methods_subdir,
        config_file=config_file,
    )
    if not inspected.get("compatible"):
        raise ValueError(inspected.get("message") or "当前训练配置与权重热启动来源不兼容")
    return {
        "continue_from_weight_abs_path": inspected["abs_path"],
        "continue_from_weight_name": inspected["name"],
        "continue_from_weight_kind": inspected["kind"],
    }


def _continue_lora_history_meta(continue_info: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(continue_info, dict) or not continue_info.get("continue_from_weight_abs_path"):
        return {"training_mode": "fresh"}
    return {
        "training_mode": "continue_lora",
        "continue_from_weight_abs_path": str(continue_info.get("continue_from_weight_abs_path") or ""),
        "continue_from_weight_name": str(continue_info.get("continue_from_weight_name") or ""),
        "continue_from_weight_kind": str(continue_info.get("continue_from_weight_kind") or ""),
    }


def _apply_gpu_whitelist(env: dict[str, str], whitelist: list[int]) -> None:
    _apply_gpu_whitelist_impl(env, whitelist)


def _resolve_training_runtime_info(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._resolve_training_runtime_info(*args, **kwargs)


def _ensure_training_data_dirs(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._ensure_training_data_dirs(*args, **kwargs)


def _write_config_snapshot(
    path: Path,
    variant: str,
    preset: str,
    methods_subdir: str,
    *,
    config_file: str | None = None,
    continue_info: dict[str, Any] | None = None,
) -> None:
    try:
        if config_file:
            source = _resolve_display_path(config_file)
            if source is None or not _path_exists(source):
                raise FileNotFoundError("续训配置快照不存在")
            text = source.read_text(encoding="utf-8", errors="replace")
            path.write_text(_append_continue_lora_snapshot_note(text, continue_info), encoding="utf-8")
            return
        cfg = apply_auto_data_dirs(load_merged_config(variant, preset, methods_subdir))
        path.write_text(_append_continue_lora_snapshot_note(toml_dumps_sorted(cfg), continue_info), encoding="utf-8")
    except Exception as e:
        path.write_text(f"# 无法生成配置快照: {e}\n", encoding="utf-8")


def _append_continue_lora_snapshot_note(text: str, continue_info: dict[str, Any] | None) -> str:
    if not isinstance(continue_info, dict) or not continue_info.get("continue_from_weight_abs_path"):
        return text
    base = text.rstrip()
    lines = [
        "",
        "",
        "# WebUI 权重热启动来源",
        '# training_mode = "continue_lora"',
        f'# continue_from_weight_kind = "{_toml_comment_string(continue_info.get("continue_from_weight_kind"))}"',
        f'# continue_from_weight_name = "{_toml_comment_string(continue_info.get("continue_from_weight_name"))}"',
        f'# continue_from_weight_abs_path = "{_toml_comment_string(continue_info.get("continue_from_weight_abs_path"))}"',
        "",
    ]
    return base + "\n".join(lines)


def _toml_comment_string(value: Any) -> str:
    return str(value or "").replace("\\", "\\\\").replace('"', '\\"')


def _load_config_file_config(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._load_config_file_config(*args, **kwargs)


def toml_dumps_sorted(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl.toml_dumps_sorted(*args, **kwargs)


def _prepare_web_runtime_config(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._prepare_web_runtime_config(*args, **kwargs)


def _apply_runtime_env(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._apply_runtime_env(*args, **kwargs)


def _runtime_meta(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._runtime_meta(*args, **kwargs)


def _delete_queue_item_runtime_dir(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._delete_queue_item_runtime_dir(*args, **kwargs)


def _queue_item_runtime_dir_label(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._queue_item_runtime_dir_label(*args, **kwargs)


def _queue_item_runtime_delete_dir(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._queue_item_runtime_delete_dir(*args, **kwargs)


def _validate_queue_runtime_dir_match(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._validate_queue_runtime_dir_match(*args, **kwargs)


def _path_is_relative_to(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._path_is_relative_to(*args, **kwargs)


def _write_runtime_run_meta(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._write_runtime_run_meta(*args, **kwargs)


def _read_runtime_run_meta(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._read_runtime_run_meta(*args, **kwargs)


def _runtime_from_config_file(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._runtime_from_config_file(*args, **kwargs)


def _clone_frozen_runtime_config(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._clone_frozen_runtime_config(*args, **kwargs)


def _clone_runtime_dataset_rows(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._clone_runtime_dataset_rows(*args, **kwargs)


def _runtime_dataset_child_name(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._runtime_dataset_child_name(*args, **kwargs)


def _bool_value_for_row(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._bool_value_for_row(*args, **kwargs)


def _prepare_runtime_nl_tag_mix_source(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._prepare_runtime_nl_tag_mix_source(*args, **kwargs)


def _prepare_runtime_trigger_clone_source(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._prepare_runtime_trigger_clone_source(*args, **kwargs)


def _nl_tag_mix_caption_settings(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._nl_tag_mix_caption_settings(*args, **kwargs)


def _build_nl_tag_mix_source(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._build_nl_tag_mix_source(*args, **kwargs)


def _classify_nl_tag_mix_samples(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._classify_nl_tag_mix_samples(*args, **kwargs)


def _nl_tag_mix_caption_entries(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._nl_tag_mix_caption_entries(*args, **kwargs)


def _nl_tag_mix_source_counts(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._nl_tag_mix_source_counts(*args, **kwargs)


def _nl_tag_mix_dominant_source(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._nl_tag_mix_dominant_source(*args, **kwargs)


def _cycle_nl_tag_entries(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._cycle_nl_tag_entries(*args, **kwargs)


def _select_nl_tag_caption_entries(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._select_nl_tag_caption_entries(*args, **kwargs)


def _nl_tag_mix_relative_image_path(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._nl_tag_mix_relative_image_path(*args, **kwargs)


def _select_nl_tag_mix_samples(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._select_nl_tag_mix_samples(*args, **kwargs)


def _copy_nl_tag_caption_sidecars(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._copy_nl_tag_caption_sidecars(*args, **kwargs)


def _copy_runtime_dataset_dir(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._copy_runtime_dataset_dir(*args, **kwargs)


def _is_materialized_runtime_source_dir(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._is_materialized_runtime_source_dir(*args, **kwargs)


def _unique_runtime_dir(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._unique_runtime_dir(*args, **kwargs)


def _history_group_meta(
    methods_subdir: str,
    variant: str,
    preset: str,
    *,
    output_dir: str = "",
    runtime_info: dict[str, Any] | None = None,
    resume_info: dict[str, Any] | None = None,
    task_id: str = "",
) -> dict[str, str]:
    runtime = runtime_info if isinstance(runtime_info, dict) else {}
    resume = resume_info if isinstance(resume_info, dict) else {}

    inherited_key = str(resume.get("history_group_key") or "").strip()
    inherited_label = str(resume.get("history_group_label") or "").strip()
    inherited_source = str(resume.get("history_source_config_file") or "").strip()
    if inherited_key:
        return {
            "history_group_key": inherited_key,
            "history_group_label": inherited_label or inherited_source or inherited_key,
            "history_source_config_file": inherited_source,
            "history_run_label": _history_run_label_from_runtime(output_dir, runtime, task_id),
        }

    source_config_file = str(runtime.get("history_source_config_file") or "").strip()
    if source_config_file:
        source_display = _display_project_path(source_config_file)
        key = "source:" + source_display
        return {
            "history_group_key": key,
            "history_group_label": source_display,
            "history_source_config_file": source_display,
            "history_run_label": _history_run_label_from_runtime(output_dir, runtime, task_id),
        }

    group = _history_config_group(methods_subdir, variant, preset)
    return {
        "history_group_key": _legacy_history_group_key(group),
        "history_group_label": _legacy_history_group_label(group),
        "history_source_config_file": "",
        "history_run_label": _history_run_label_from_runtime(output_dir, runtime, task_id),
    }


def _fill_history_group_meta(task: dict[str, Any]) -> None:
    existing_key = str(task.get("history_group_key") or "").strip()
    if existing_key:
        task["history_group_key"] = existing_key
        task["history_group_label"] = str(
            task.get("history_group_label")
            or task.get("history_source_config_file")
            or existing_key
        )
        task["history_source_config_file"] = str(task.get("history_source_config_file") or "")
        if not str(task.get("history_run_label") or "").strip():
            task["history_run_label"] = _history_run_label_from_runtime(
                str(task.get("training_output_dir") or task.get("output_dir") or ""),
                task,
                str(task.get("id") or ""),
            )
        return
    task.update(_history_group_meta(
        str(task.get("methods_subdir") or ""),
        str(task.get("variant") or ""),
        str(task.get("preset") or "default"),
        output_dir=str(task.get("training_output_dir") or task.get("output_dir") or ""),
        runtime_info=task,
        resume_info=task.get("resume_from") if isinstance(task.get("resume_from"), dict) else None,
        task_id=str(task.get("id") or ""),
    ))


def _history_run_label_from_runtime(
    output_dir: str,
    runtime_info: dict[str, Any] | None,
    task_id: str = "",
) -> str:
    runtime = runtime_info if isinstance(runtime_info, dict) else {}
    for key in ("run_dir", "training_output_dir", "output_dir"):
        raw = str(runtime.get(key) or "").strip()
        label = _history_run_label_from_path(raw)
        if label:
            return label
    return _history_run_label_from_path(output_dir) or str(task_id or "").strip()


def _history_run_label_from_path(value: str) -> str:
    path = _resolve_display_path(str(value or ""))
    if path is None:
        return ""
    if path.name == "training_output":
        return path.parent.name
    return path.name


def _legacy_history_group_key(group: dict[str, str]) -> str:
    return "legacy:" + "\u0001".join([
        group.get("methods_subdir") or "",
        group.get("variant") or "",
        group.get("preset") or "default",
    ])


def _legacy_history_group_label(group: dict[str, str]) -> str:
    return f"{group.get('methods_subdir') or '-'} / {group.get('variant') or '-'} / {group.get('preset') or 'default'}"


def _safe_run_stem(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._safe_run_stem(*args, **kwargs)


def _load_training_queue_state() -> dict[str, Any]:
    data = _read_training_queue_state()
    if not isinstance(data, dict):
        data = {}
    items = data.get("items")
    if not isinstance(items, list):
        data["items"] = []
    data["paused"] = bool(data.get("paused", False))
    data["failure_policy"] = _normalize_queue_failure_policy(data.get("failure_policy"))
    return data


def _read_training_queue_state() -> dict[str, Any]:
    data = _read_json_object(QUEUE_FILE)
    if isinstance(data, dict):
        return data
    backup = _read_json_object(_queue_backup_file())
    if isinstance(backup, dict):
        try:
            _write_json_atomic(QUEUE_FILE, backup)
        except Exception:
            pass
        return backup
    return {}


def _write_training_queue_state(payload: dict[str, Any]) -> None:
    _write_json_atomic(QUEUE_FILE, payload)
    try:
        _write_json_atomic(_queue_backup_file(), payload)
    except Exception:
        pass


def _queue_backup_file() -> Path:
    return QUEUE_FILE.with_name(QUEUE_FILE.name + ".bak")


def _load_history_collection_settings() -> dict[str, Any]:
    data = _read_json_object(HISTORY_COLLECTIONS_FILE) or {}
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
    return text if text in QUEUE_FAILURE_POLICIES else "pause"


def _queue_clearable_state_label(states: set[str]) -> str:
    clean = {str(state or "").strip() for state in states}
    if clean == {"done"}:
        return QUEUE_CLEARABLE_STATE_LABELS["done"]
    if clean == {"canceled"}:
        return QUEUE_CLEARABLE_STATE_LABELS["canceled"]
    return "已结束"


def _new_queue_item_id(kind: str, methods_subdir: str, variant: str) -> str:
    raw = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-queue-{kind}-{methods_subdir}-{variant}"
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


def _list_history_tasks(*, include_archived: bool = False, limit: int | None = None) -> list[dict[str, Any]]:
    meta_paths = _history_meta_paths()
    records = _history_meta_records(meta_paths, repair=True)
    _sync_bound_history_collection_groups(records=records)

    tasks = []
    for record in records:
        meta_path = record["path"]
        task = _safe_history_summary(record["meta"], meta_path.parent)
        if task is None:
            continue
        if include_archived or not task.get("archived"):
            tasks.append(task)
    tasks.sort(key=lambda item: item.get("started_at") or 0, reverse=True)
    if limit is None:
        limit = MAX_HISTORY_ITEMS
    if limit and limit > 0:
        return tasks[:limit]
    return tasks


def _history_meta_paths() -> list[Path]:
    if not _path_exists(HISTORY_DIR):
        return []
    try:
        candidates = list(HISTORY_DIR.iterdir())
    except OSError:
        return []
    out: list[Path] = []
    for task_dir in candidates:
        if _is_deleting_history_dir(task_dir):
            continue
        meta_path = task_dir / "meta.json"
        if _path_exists(meta_path):
            out.append(meta_path)
    return out


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


def _batch_archive_history_tasks(task_ids: list[str], *, archived: bool) -> dict[str, Any]:
    tasks = []
    for task_id in task_ids:
        tasks.append(_update_history_task(task_id, {"archived": archived})["task"])
    return {
        "ok": True,
        "message": "已归档所选历史任务" if archived else "已取消归档所选历史任务",
        "updated": len(tasks),
        "tasks": tasks,
    }


def _batch_set_history_group(task_ids: list[str], group: Any) -> dict[str, Any]:
    expanded_task_ids = _bound_history_task_ids(task_ids)
    tasks = []
    for task_id in expanded_task_ids:
        tasks.append(_update_history_task(task_id, {"group": group}, bind_group=False)["task"])
    return {
        "ok": True,
        "message": "已更新同配置文件自动分组内的历史任务集合",
        "updated": len(tasks),
        "requested": len(task_ids),
        "tasks": tasks,
    }


def _history_meta_records(
    meta_paths: list[Path] | None = None,
    *,
    repair: bool = False,
) -> list[dict[str, Any]]:
    paths = meta_paths if meta_paths is not None else _history_meta_paths()
    records: list[dict[str, Any]] = []
    for meta_path in paths:
        if _is_deleting_history_dir(meta_path.parent):
            continue
        meta = _read_json(meta_path)
        if not meta:
            continue
        if repair:
            _repair_history_meta(meta_path, meta)
        records.append(_history_meta_record(meta_path, meta))
    return records


def _history_meta_record(meta_path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    task_id = str(meta.get("id") or meta_path.parent.name).strip()
    work = dict(meta)
    _fill_history_runtime_meta(work)
    _fill_history_group_meta(work)
    return {
        "id": task_id,
        "path": meta_path,
        "meta": meta,
        "group_key": str(work.get("history_group_key") or "").strip(),
        "job": str(meta.get("job") or "").strip(),
        "group": _clean_history_text(meta.get("group"), max_len=48),
        "updated_at": float(meta.get("updated_at") or 0),
        "started_at": float(meta.get("started_at") or 0),
    }


def _bound_history_task_ids(task_ids: list[str]) -> list[str]:
    requested = [str(task_id or "").strip() for task_id in task_ids if str(task_id or "").strip()]
    if not requested:
        return []
    requested_set = set(requested)
    records = _history_meta_records()
    selected_keys = {
        str(record["group_key"])
        for record in records
        if record["id"] in requested_set and str(record["group_key"])
    }
    if not selected_keys:
        return requested
    expanded = [
        str(record["id"])
        for record in records
        if record["id"] in requested_set or str(record["group_key"]) in selected_keys
    ]
    ordered: list[str] = []
    for task_id in requested + expanded:
        if task_id not in ordered:
            ordered.append(task_id)
    return ordered


def _sync_bound_history_collection_groups(
    meta_paths: list[Path] | None = None,
    *,
    records: list[dict[str, Any]] | None = None,
) -> int:
    records = records if records is not None else _history_meta_records(meta_paths)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = str(record.get("group_key") or "")
        if not key:
            continue
        grouped.setdefault(key, []).append(record)

    changed = 0
    for records_in_group in grouped.values():
        target_group = _preferred_bound_history_collection_group(records_in_group)
        if target_group is None:
            continue
        for record in records_in_group:
            if str(record.get("group") or "") == target_group:
                continue
            meta = dict(record["meta"])
            meta["group"] = target_group
            now = time.time()
            meta["updated_at"] = now
            meta["updated_at_text"] = _format_ts(now)
            try:
                _write_json_atomic(record["path"], meta)
                record["meta"] = meta
                record["group"] = target_group
                record["updated_at"] = now
                changed += 1
            except OSError:
                continue
    return changed


def _preferred_bound_history_collection_group(records: list[dict[str, Any]]) -> str | None:
    candidates = [record for record in records if str(record.get("group") or "").strip()]
    if not candidates:
        return None
    candidates.sort(
        key=lambda record: (
            1 if record.get("job") == "training" else 0,
            float(record.get("updated_at") or 0),
            float(record.get("started_at") or 0),
        ),
        reverse=True,
    )
    return str(candidates[0].get("group") or "").strip()


def _mark_orphaned_running_history_tasks() -> int:
    count = 0
    for meta_path in _history_meta_paths():
        if _is_deleting_history_dir(meta_path.parent):
            continue
        meta = _read_json(meta_path)
        if not meta or meta.get("state") != "running":
            continue
        task_dir = meta_path.parent
        finished_at = _last_history_event_ts(task_dir, meta)
        meta.update({
            "state": "interrupted",
            "finished_at": finished_at,
            "finished_at_text": _format_ts(finished_at),
            "message": "WebUI 上次退出时任务仍标记为运行中，已自动标记为中断。",
            "returncode": meta.get("returncode"),
            "log_count": _count_jsonl(task_dir / "logs.jsonl"),
            "metric_count": _count_jsonl(task_dir / "metrics.jsonl"),
            "interrupted_at": time.time(),
            "interrupted_at_text": _format_ts(time.time()),
        })
        try:
            _write_json_atomic(meta_path, meta)
        except OSError:
            continue
        count += 1
    return count


def _last_history_event_ts(task_dir: Path, meta: dict[str, Any]) -> float:
    candidates = [
        _float_or_none(meta.get("finished_at")),
        _float_or_none(meta.get("updated_at")),
    ]
    for filename in ("logs.jsonl", "metrics.jsonl", "system.jsonl"):
        records = _read_jsonl(task_dir / filename)
        for record in reversed(records):
            ts = _float_or_none(record.get("ts"))
            if ts is not None:
                candidates.append(ts)
                break
    candidates.append(_float_or_none(meta.get("started_at")))
    candidates = [value for value in candidates if value is not None]
    return max(candidates) if candidates else time.time()


def _ensure_history_average_speed_meta(meta_path: Path, task_dir: Path, meta: dict[str, Any]) -> None:
    if str(meta.get("job") or "").strip() != "training":
        return
    if str(meta.get("state") or "").strip() in {"running", "compiling", "queued"}:
        return
    current_version = _int_or_none(meta.get("average_step_speed_version"))
    existing_seconds = _float_or_none(meta.get("average_step_seconds"))
    if current_version == HISTORY_AVERAGE_SPEED_VERSION and existing_seconds is not None and existing_seconds > 0:
        return
    stats = _history_average_speed_from_logs(task_dir / "logs.jsonl")
    if not stats:
        return
    now = time.time()
    meta.update(stats)
    meta["average_step_speed_version"] = HISTORY_AVERAGE_SPEED_VERSION
    meta["average_step_computed_at"] = now
    meta["average_step_computed_at_text"] = _format_ts(now)
    try:
        _write_json_atomic(meta_path, meta)
    except OSError:
        pass


def _history_average_speed_from_logs(logs_path: Path) -> dict[str, Any] | None:
    first: tuple[int, float] | None = None
    last: tuple[int, float] | None = None
    sample_count = 0
    try:
        with logs_path.open("r", encoding="utf-8") as f:
            for raw in f:
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if record.get("kind") != "progress":
                    continue
                sample = _training_progress_log_sample(record)
                if sample is None:
                    continue
                step, ts = sample
                sample_count += 1
                if first is None:
                    first = sample
                    continue
                first_step, first_ts = first
                if step <= first_step or ts <= first_ts:
                    continue
                if last is None or step > last[0] or (step == last[0] and ts > last[1]):
                    last = sample
    except OSError:
        return None
    if first is None or last is None:
        return None
    first_step, first_ts = first
    last_step, last_ts = last
    step_delta = last_step - first_step
    seconds_delta = last_ts - first_ts
    if step_delta <= 0 or seconds_delta <= 0:
        return None
    seconds_per_step = seconds_delta / step_delta
    if not _is_finite_number(seconds_per_step) or seconds_per_step <= 0:
        return None
    return {
        "average_step_seconds": round(seconds_per_step, 4),
        "average_step_rate": _format_step_rate(seconds_per_step),
        "average_step_source": "logs.jsonl",
        "average_step_sample_count": sample_count,
        "average_step_start_step": first_step,
        "average_step_end_step": last_step,
        "average_step_started_at": first_ts,
        "average_step_finished_at": last_ts,
    }


def _training_progress_log_sample(record: dict[str, Any]) -> tuple[int, float] | None:
    ts = _float_or_none(record.get("ts"))
    if ts is None:
        return None
    line = str(record.get("line") or "")
    match = TRAINING_PROGRESS_LOG_RE.search(line)
    if not match:
        return None
    step = _int_or_none(match.group("cur"))
    total = _int_or_none(match.group("tot"))
    if step is None or total is None or step < 0 or total <= 0:
        return None
    return step, ts


def _history_task_dir(task_id: str) -> Path:
    safe_id = _safe_task_id(task_id)
    if safe_id != task_id:
        raise ValueError("任务 ID 不合法")
    task_dir = (HISTORY_DIR / safe_id).resolve()
    try:
        task_dir.relative_to(HISTORY_DIR.resolve())
    except ValueError as exc:
        raise ValueError("任务 ID 不合法") from exc
    return task_dir


def _load_history_task(task_id: str) -> dict[str, Any]:
    task_dir = _history_task_dir(task_id)
    if not _path_exists(task_dir):
        raise FileNotFoundError("任务不存在")
    meta = _read_json(task_dir / "meta.json")
    if not meta:
        raise FileNotFoundError("任务元信息不存在")
    _repair_history_meta(task_dir / "meta.json", meta)
    _ensure_history_average_speed_meta(task_dir / "meta.json", task_dir, meta)
    snapshot_path = task_dir / "config.snapshot.toml"
    logs, logs_total, logs_truncated = _read_jsonl_limited(
        task_dir / "logs.jsonl",
        limit=MAX_HISTORY_DETAIL_LOG_RECORDS,
    )
    system, system_total, system_truncated = _read_jsonl_limited(
        task_dir / "system.jsonl",
        limit=MAX_HISTORY_DETAIL_SYSTEM_RECORDS,
    )
    metrics = _history_metrics_for_task(task_dir, logs=logs)
    task = _history_summary(meta, task_dir)
    if str(task.get("job") or "").strip() == "training":
        task["linked_preprocess_task"] = _linked_preprocess_task_for_training(task)
    return {
        "ok": True,
        "task": task,
        "logs": logs,
        "metrics": metrics,
        "system": system,
        "limits": {
            "logs_total": logs_total,
            "logs_returned": len(logs),
            "logs_truncated": logs_truncated,
            "system_total": system_total,
            "system_returned": len(system),
            "system_truncated": system_truncated,
            "metrics_total": len(metrics),
            "metrics_returned": len(metrics),
            "metrics_truncated": False,
        },
        "config_toml": _read_text_file(snapshot_path),
    }


def _history_log_path(task_id: str) -> Path:
    task_dir = _history_task_dir(task_id)
    if not _path_exists(task_dir):
        raise FileNotFoundError("任务不存在")
    path = (task_dir / "logs.jsonl").resolve()
    try:
        path.relative_to(task_dir.resolve())
    except ValueError as exc:
        raise ValueError("任务日志路径不合法") from exc
    if not _path_exists(path) or not path.is_file():
        raise FileNotFoundError("任务日志不存在")
    return path


def _history_artifact_path(task_id: str, artifact_key: str) -> Path:
    key = str(artifact_key or "").strip()
    if key in HISTORY_ARTIFACT_FILES:
        return _history_task_file_artifact_path(task_id, HISTORY_ARTIFACT_FILES[key])
    if key in HISTORY_RUNTIME_ARTIFACT_FIELDS:
        return _history_runtime_artifact_path(task_id, HISTORY_RUNTIME_ARTIFACT_FIELDS[key])
    raise ValueError("历史文件类型不支持")


def _history_task_file_artifact_path(task_id: str, filename: str) -> Path:
    task_dir = _history_task_dir(task_id)
    if not _path_exists(task_dir):
        raise FileNotFoundError("任务不存在")
    path = (task_dir / filename).resolve()
    try:
        path.relative_to(task_dir.resolve())
    except ValueError as exc:
        raise ValueError("历史文件路径不合法") from exc
    if not _path_exists(path) or not path.is_file():
        raise FileNotFoundError("历史文件不存在")
    return path


def _history_runtime_artifact_path(task_id: str, field: str) -> Path:
    task_dir = _history_task_dir(task_id)
    if not _path_exists(task_dir):
        raise FileNotFoundError("任务不存在")
    meta = _read_json(task_dir / "meta.json")
    if not meta:
        raise FileNotFoundError("任务元信息不存在")
    task = _history_summary(meta, task_dir)
    run_dir = _resolve_display_path(str(task.get("run_dir") or ""))
    path = _resolve_display_path(str(task.get(field) or ""))
    if run_dir is None or path is None:
        raise FileNotFoundError("运行文件不存在")
    run_dir = run_dir.resolve()
    path = path.resolve()
    try:
        path.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError("运行文件路径不合法") from exc
    output_root = resolve_output_root().resolve()
    if not _path_is_relative_to(run_dir, output_root) and not _is_web_runtime_dir(run_dir):
        raise ValueError("运行文件路径不合法")
    if not _path_exists(path) or not path.is_file():
        raise FileNotFoundError("运行文件不存在")
    return path


def _build_config_group_timeline(
    methods_subdir: str,
    variant: str,
    preset: str,
    *,
    group_key: str = "",
    include_archived: bool = False,
    task_ids: list[str] | None = None,
) -> dict[str, Any]:
    group = _history_config_group(methods_subdir, variant, preset)
    group_key = str(group_key or "").strip()
    selected_ids = _normalize_timeline_task_ids(task_ids)
    all_tasks = _list_history_tasks(include_archived=True)
    if selected_ids:
        tasks = _select_timeline_tasks_by_id(
            all_tasks,
            selected_ids,
            include_archived=include_archived,
        )
        groups = _timeline_groups_for_tasks(tasks)
        if len(groups) == 1:
            group = groups[0]
        else:
            group = {
                "methods_subdir": "手动选择",
                "variant": f"{len(groups)} 个配置分组",
                "preset": "selected",
            }
    else:
        tasks = [
            task for task in all_tasks
            if task.get("job") == "training"
            and (
                _task_history_group_matches(task, group_key)
                if group_key
                else _task_config_group_matches(task, group)
            )
            and (include_archived or not task.get("archived"))
        ]
        if group_key and tasks:
            group = _history_group_from_task(tasks[0])
    tasks.sort(key=lambda item: (float(item.get("started_at") or 0), str(item.get("id") or "")))
    if not tasks:
        if selected_ids:
            raise FileNotFoundError("没有找到可合并的已选训练任务")
        raise FileNotFoundError("这个配置文件分组没有可合并的训练任务")

    logs: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    next_visual_step = 1

    for index, task in enumerate(tasks, start=1):
        task_id = str(task.get("id") or "")
        task_dir = _history_task_dir(task_id)
        task_logs = _read_jsonl(task_dir / "logs.jsonl")
        visible_logs = [record for record in task_logs if record.get("kind") != "progress"]
        task_metrics = _timeline_training_metrics(_history_metrics_for_task(task_dir, logs=task_logs))
        if task_metrics:
            start_visual_step = next_visual_step
            next_visual_step = _assign_visual_steps(task_metrics, next_visual_step)
            end_visual_step = next_visual_step - 1
            display_step_offset = _timeline_resume_step_offset(task)
            start_display_step, end_display_step = _assign_display_steps(task_metrics, display_step_offset)
            raw_steps = [_int_or_none(item.get("step")) for item in task_metrics]
            raw_steps = [step for step in raw_steps if step is not None]
            start_raw_step = raw_steps[0] if raw_steps else None
            end_raw_step = raw_steps[-1] if raw_steps else None
        else:
            start_visual_step = None
            end_visual_step = None
            display_step_offset = _timeline_resume_step_offset(task)
            start_display_step = None
            end_display_step = None
            start_raw_step = None
            end_raw_step = None

        source_label = _timeline_task_label(task)
        for record in visible_logs:
            item = dict(record)
            item["source_task_id"] = task_id
            item["source_task_index"] = index
            item["source_task_label"] = source_label
            logs.append(item)

        for metric_offset, metric in enumerate(task_metrics):
            item = dict(metric)
            item["source_task_id"] = task_id
            item["source_task_index"] = index
            item["source_task_label"] = source_label
            item["stage_break_before"] = index > 1 and metric_offset == 0
            metrics.append(item)

        segments.append({
            "task": _timeline_task_brief(task),
            "index": index,
            "log_count": len(visible_logs),
            "raw_log_count": len(task_logs),
            "progress_count": max(0, len(task_logs) - len(visible_logs)),
            "metric_count": len(task_metrics),
            "loss_count": sum(1 for item in task_metrics if item.get("loss") is not None),
            "start_visual_step": start_visual_step,
            "end_visual_step": end_visual_step,
            "start_display_step": start_display_step,
            "end_display_step": end_display_step,
            "display_step_offset": display_step_offset,
            "start_raw_step": start_raw_step,
            "end_raw_step": end_raw_step,
        })

    logs.sort(key=lambda item: (
        float(item.get("ts") or 0),
        int(item.get("source_task_index") or 0),
        int(item.get("id") or 0),
    ))
    metrics.sort(key=lambda item: (
        float(item.get("ts") or 0),
        int(item.get("source_task_index") or 0),
        int(item.get("visual_step") or 0),
    ))

    if len(logs) > MAX_TIMELINE_LOG_RECORDS:
        logs = logs[-MAX_TIMELINE_LOG_RECORDS:]
    if len(metrics) > MAX_TIMELINE_METRIC_RECORDS:
        metrics = metrics[-MAX_TIMELINE_METRIC_RECORDS:]

    return {
        "ok": True,
        "mode": "config_group",
        "group": group,
        "tasks": [_timeline_task_brief(task) for task in tasks],
        "segments": segments,
        "logs": logs,
        "metrics": metrics,
        "summary": {
            "task_count": len(tasks),
            "log_count": len(logs),
            "raw_log_count": sum(segment["raw_log_count"] for segment in segments),
            "progress_count": sum(segment["progress_count"] for segment in segments),
            "metric_count": len(metrics),
            "loss_count": sum(1 for item in metrics if item.get("loss") is not None),
            "started_at": tasks[0].get("started_at") if tasks else None,
            "started_at_text": tasks[0].get("started_at_text") if tasks else "",
            "finished_at": tasks[-1].get("finished_at") if tasks and tasks[-1].get("finished_at") else None,
            "finished_at_text": tasks[-1].get("finished_at_text") if tasks and tasks[-1].get("finished_at") else "",
            "start_display_step": next((segment["start_display_step"] for segment in segments if segment["start_display_step"] is not None), None),
            "end_display_step": next((segment["end_display_step"] for segment in reversed(segments) if segment["end_display_step"] is not None), None),
            "include_archived": include_archived,
            "selection_mode": "manual" if selected_ids else "config_group",
            "selected_task_ids": [str(task.get("id") or "") for task in tasks],
            "group_count": len(_timeline_groups_for_tasks(tasks)),
        },
    }


def _normalize_timeline_task_ids(task_ids: list[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in task_ids or []:
        task_id = str(raw or "").strip()
        if not task_id or task_id in seen:
            continue
        out.append(task_id)
        seen.add(task_id)
    return out


def _select_timeline_tasks_by_id(
    tasks: list[dict[str, Any]],
    task_ids: list[str],
    *,
    include_archived: bool,
) -> list[dict[str, Any]]:
    by_id = {str(task.get("id") or ""): task for task in tasks}
    selected: list[dict[str, Any]] = []
    invalid: list[str] = []
    for task_id in task_ids:
        task = by_id.get(task_id)
        if (
            not task
            or task.get("job") != "training"
            or (task.get("archived") and not include_archived)
        ):
            invalid.append(task_id)
            continue
        selected.append(task)
    if invalid:
        raise ValueError("所选训练任务不存在、已隐藏或不能参与合并: " + ", ".join(invalid))
    return selected


def _timeline_groups_for_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, str]]:
    groups: list[dict[str, str]] = []
    seen: set[str] = set()
    for task in tasks:
        group = _history_group_from_task(task)
        key = str(group.get("history_group_key") or "")
        if key in seen:
            continue
        seen.add(key)
        groups.append(group)
    return groups


def _history_config_group(methods_subdir: str, variant: str, preset: str) -> dict[str, str]:
    return {
        "methods_subdir": str(methods_subdir or "").strip(),
        "variant": str(variant or "").strip(),
        "preset": str(preset or "default").strip() or "default",
    }


def _task_config_group_matches(task: dict[str, Any], group: dict[str, str]) -> bool:
    task_group = _history_config_group(
        str(task.get("methods_subdir") or ""),
        str(task.get("variant") or ""),
        str(task.get("preset") or "default"),
    )
    return task_group == group


def _task_history_group_matches(task: dict[str, Any], group_key: str) -> bool:
    return str(task.get("history_group_key") or "").strip() == str(group_key or "").strip()


def _history_group_from_task(task: dict[str, Any]) -> dict[str, str]:
    group = _history_config_group(
        str(task.get("methods_subdir") or ""),
        str(task.get("variant") or ""),
        str(task.get("preset") or "default"),
    )
    history_key = str(task.get("history_group_key") or "").strip() or _legacy_history_group_key(group)
    history_label = str(task.get("history_group_label") or "").strip() or _legacy_history_group_label(group)
    source_config = str(task.get("history_source_config_file") or "").strip()
    run_label = str(task.get("history_run_label") or "").strip()
    return {
        **group,
        "key": history_key,
        "history_group_key": history_key,
        "history_group_label": history_label,
        "history_source_config_file": source_config,
        "history_run_label": run_label,
        "label": history_label,
    }


def _history_metrics_for_task(
    task_dir: Path,
    *,
    logs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    task_logs = logs if logs is not None else _read_jsonl(task_dir / "logs.jsonl")
    metrics = _read_jsonl(task_dir / "metrics.jsonl")
    progress_metrics = _metrics_from_progress_jsonl(task_dir / "progress.jsonl", task_dir)
    return _metrics_from_history(task_logs, metrics, progress_metrics)


def _metrics_from_progress_jsonl(progress_path: Path, task_dir: Path) -> list[dict[str, Any]]:
    events = _read_jsonl(progress_path)
    if not events:
        return []
    meta = _read_json(task_dir / "meta.json")
    started_at = _float_or_none(meta.get("started_at")) if isinstance(meta, dict) else None
    out: list[dict[str, Any]] = []
    rate_last: tuple[float, int] | None = None
    rate_samples: deque[float] = deque(maxlen=PROGRESS_RATE_SAMPLE_WINDOW)
    for event in events:
        ev = str(event.get("ev") or "")
        if ev not in {"step", "val"}:
            continue
        ts = _progress_event_wall_ts_from_started_at(event, started_at)
        step = _int_or_none(event.get("global_step"))
        rate = ""
        if ev == "step" and step is not None:
            rate, rate_last = _step_rate_text_from_sample(rate_last, rate_samples, step, ts)
        metric = _metric_from_progress_jsonl_event(
            event,
            ts,
            rate=rate,
        )
        if metric:
            out.append(metric)
    return out


def _step_rate_text_from_sample(
    last: tuple[float, int] | None,
    samples: deque[float],
    step: int,
    timestamp: float,
) -> tuple[str, tuple[float, int] | None]:
    return _progress_parser.step_rate_text_from_sample(last, samples, step, timestamp)


def _median_or_none(values: deque[float] | list[float]) -> float | None:
    return _progress_parser.median_or_none(values)


def _format_step_rate(seconds_per_step: float | None) -> str:
    return _progress_parser.format_step_rate(seconds_per_step)


def _is_finite_number(value: Any) -> bool:
    return _progress_parser.is_finite_number(value)


def _metrics_from_history(
    logs: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    progress_metrics: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[int | None, float | None, float | None, float | None, str]] = set()

    def add_metric(item: dict[str, Any]) -> None:
        normalized = _normalize_metric_record(item)
        if normalized is None:
            return
        key = _metric_seen_key(normalized)
        if key in seen:
            return
        seen.add(key)
        out.append(normalized)

    normalized_progress = [
        item
        for item in (_normalize_metric_record(record) for record in (progress_metrics or []))
        if item is not None
    ]
    has_structured_loss = any(
        str(item.get("kind") or "") != "val" and _float_or_none(item.get("loss")) is not None
        for item in normalized_progress
    )

    if has_structured_loss:
        for item in normalized_progress:
            add_metric(item)
        for item in metrics:
            normalized = _normalize_metric_record(item)
            if normalized is not None and _is_validation_metric(normalized):
                add_metric(normalized)
    else:
        for item in metrics:
            add_metric(item)
        for item in normalized_progress:
            add_metric(item)

    for record in logs:
        if record.get("kind") != "progress":
            continue
        parsed = _metric_from_progress_line(str(record.get("line") or ""))
        if parsed is None:
            continue
        if record.get("ts") is not None:
            parsed["ts"] = record.get("ts")
        if has_structured_loss and not _is_validation_metric(parsed):
            continue
        add_metric(parsed)

    out.sort(key=lambda item: (float(item.get("ts") or 0), int(item.get("step") or 0)))
    return out


def _is_validation_metric(item: dict[str, Any]) -> bool:
    return str(item.get("kind") or "") == "val" or _float_or_none(item.get("cmmd")) is not None


def _timeline_training_metrics(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _progress_parser.timeline_training_metrics(metrics)


def _normalize_metric_record(item: dict[str, Any]) -> dict[str, Any] | None:
    return _progress_parser.normalize_metric_record(item)


def _metric_from_progress_line(line: str) -> dict[str, Any] | None:
    return _progress_parser.metric_from_progress_line(line)


def _metric_seen_key(item: dict[str, Any]) -> tuple[int | None, float | None, float | None, float | None, str]:
    return _progress_parser.metric_seen_key(item)


def _assign_visual_steps(metrics: list[dict[str, Any]], next_step: int) -> int:
    for item in metrics:
        item["visual_step"] = next_step
        next_step += 1
    return next_step


def _timeline_resume_step_offset(task: dict[str, Any]) -> int:
    resume_from = task.get("resume_from")
    if not isinstance(resume_from, dict):
        return 0
    checkpoint_step = _int_or_none(resume_from.get("checkpoint_step"))
    return checkpoint_step if checkpoint_step is not None and checkpoint_step > 0 else 0


def _assign_display_steps(metrics: list[dict[str, Any]], offset: int) -> tuple[int | None, int | None]:
    return _progress_parser.assign_display_steps(metrics, offset)


def _timeline_task_brief(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": task.get("id", ""),
        "name": task.get("name", ""),
        "label": _timeline_task_label(task),
        "training_mode": task.get("training_mode", ""),
        "continue_from_weight_abs_path": task.get("continue_from_weight_abs_path", ""),
        "continue_from_weight_name": task.get("continue_from_weight_name", ""),
        "continue_from_weight_kind": task.get("continue_from_weight_kind", ""),
        "state": task.get("state", ""),
        "variant": task.get("variant", ""),
        "preset": task.get("preset", ""),
        "methods_subdir": task.get("methods_subdir", ""),
        "output_dir": task.get("output_dir", ""),
        "run_dir": task.get("run_dir", ""),
        "history_dir": task.get("history_dir", ""),
        "history_group_key": task.get("history_group_key", ""),
        "history_group_label": task.get("history_group_label", ""),
        "history_source_config_file": task.get("history_source_config_file", ""),
        "history_run_label": task.get("history_run_label", ""),
        "resume_from": task.get("resume_from") if isinstance(task.get("resume_from"), dict) else {},
        "started_at": task.get("started_at"),
        "started_at_text": task.get("started_at_text", ""),
        "finished_at": task.get("finished_at"),
        "finished_at_text": task.get("finished_at_text", ""),
        "log_count": int(task.get("log_count") or 0),
        "metric_count": int(task.get("metric_count") or 0),
        "archived": bool(task.get("archived", False)),
    }


def _timeline_task_label(task: dict[str, Any]) -> str:
    return str(
        task.get("name")
        or task.get("history_run_label")
        or f"{task.get('methods_subdir') or '-'} / {task.get('variant') or task.get('id') or '-'}"
    )


def _update_history_task(task_id: str, patch: dict[str, Any], *, bind_group: bool = True) -> dict[str, Any]:
    if bind_group and set(patch.keys()) == {"group"}:
        expanded_task_ids = _bound_history_task_ids([task_id])
        tasks = [
            _update_history_task(bound_id, patch, bind_group=False)["task"]
            for bound_id in expanded_task_ids
        ]
        primary = next((task for task in tasks if task.get("id") == task_id), tasks[0] if tasks else {})
        return {
            "ok": True,
            "task": primary,
            "tasks": tasks,
            "updated": len(tasks),
        }

    task_dir = _history_task_dir(task_id)
    if not task_dir.exists():
        raise FileNotFoundError("任务不存在")
    meta_path = task_dir / "meta.json"
    meta = _read_json(meta_path)
    if not meta:
        raise FileNotFoundError("任务元信息不存在")

    if "name" in patch:
        meta["name"] = _clean_history_text(patch.get("name"), max_len=80)
    if "group" in patch:
        meta["group"] = _clean_history_text(patch.get("group"), max_len=48)
    if "archived" in patch:
        meta["archived"] = bool(patch.get("archived"))

    meta["updated_at"] = time.time()
    meta["updated_at_text"] = _format_ts(meta["updated_at"])
    _write_json_atomic(meta_path, meta)
    return {"ok": True, "task": _history_summary(meta, task_dir)}


def _history_task_ids_for_delete(task_id: str) -> list[str]:
    task_dir = _history_task_dir(task_id)
    if not _path_exists(task_dir):
        raise FileNotFoundError("任务不存在")
    meta = _read_json(task_dir / "meta.json")
    if not meta:
        return [task_id]
    _repair_history_meta(task_dir / "meta.json", meta)
    task = _history_summary(meta, task_dir)
    task_ids = [task_id]
    if str(task.get("job") or "").strip() != "training":
        return task_ids

    seen = {task_id}
    for candidate in _linked_preprocess_tasks_for_training(task):
        candidate_id = str(candidate.get("id") or "").strip()
        if not candidate_id or candidate_id in seen:
            continue
        task_ids.append(candidate_id)
        seen.add(candidate_id)
    return task_ids


def _linked_preprocess_task_for_training(task: dict[str, Any]) -> dict[str, Any] | None:
    tasks = _linked_preprocess_tasks_for_training(task)
    if not tasks:
        return None
    return _history_linked_task_brief(tasks[0])


def _linked_preprocess_tasks_for_training(task: dict[str, Any]) -> list[dict[str, Any]]:
    if str(task.get("job") or "").strip() != "training":
        return []
    run_key = _history_delete_run_key(task)
    if not run_key:
        return []

    current_id = str(task.get("id") or "").strip()
    out: list[dict[str, Any]] = []
    seen = {current_id} if current_id else set()
    for candidate in _list_history_tasks(include_archived=True, limit=0):
        candidate_id = str(candidate.get("id") or "").strip()
        if not candidate_id or candidate_id in seen:
            continue
        if str(candidate.get("job") or "").strip() != "preprocess":
            continue
        if _history_delete_run_key(candidate) != run_key:
            continue
        out.append(candidate)
        seen.add(candidate_id)
    return out


def _history_linked_task_brief(task: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "id",
        "name",
        "job",
        "state",
        "archived",
        "started_at",
        "started_at_text",
        "finished_at",
        "finished_at_text",
        "history_run_label",
        "history_source_config_file",
        "history_group_key",
        "run_dir",
        "output_dir",
        "training_output_dir",
        "log_count",
        "metric_count",
    )
    return {key: task.get(key) for key in keys if key in task}


def _history_delete_run_key(task: dict[str, Any]) -> str:
    for key in ("run_dir", "training_output_dir", "output_dir"):
        path = _resolve_display_path(str(task.get(key) or ""))
        if path is None:
            continue
        if path.name == "training_output":
            path = path.parent
        return str(path)
    return ""


def _history_delete_task_preview(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(task.get("id") or ""),
        "name": str(task.get("name") or task.get("history_run_label") or ""),
        "job": str(task.get("job") or ""),
        "state": str(task.get("state") or ""),
        "started_at_text": str(task.get("started_at_text") or ""),
        "run_dir": str(task.get("run_dir") or ""),
        "output_dir": str(task.get("training_output_dir") or task.get("output_dir") or ""),
    }


def _history_runtime_delete_dirs_for_tasks(tasks: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    out: list[dict[str, str]] = []
    blocked: list[dict[str, str]] = []
    seen: set[str] = set()
    for task in tasks:
        raw = _history_delete_run_key(task)
        if not raw or raw in seen:
            continue
        seen.add(raw)
        path = _resolve_display_path(raw)
        label = _display_settings_path(path) if path is not None else raw
        if path is None:
            blocked.append({"path": raw, "reason": "运行目录路径无效"})
            continue
        if not _path_exists(path):
            out.append({"path": label, "status": "missing"})
            continue
        if not path.is_dir():
            blocked.append({"path": label, "reason": "运行目录不是文件夹"})
            continue
        try:
            output_root = resolve_output_root()
        except Exception as exc:
            blocked.append({"path": label, "reason": f"无法解析输出根目录: {exc}"})
            continue
        if path == output_root or not _path_is_relative_to(path, output_root):
            blocked.append({"path": label, "reason": "运行目录不在 WebUI 输出根目录内"})
            continue
        if not _is_web_runtime_dir(path):
            blocked.append({"path": label, "reason": "缺少 WebUI runtime 标记"})
            continue
        out.append({"path": label, "status": "ready"})
    return out, blocked


def _is_web_runtime_dir(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._is_web_runtime_dir(*args, **kwargs)


def _queue_runtime_delete_blockers(
    queue_items: list[dict[str, Any]],
    runtime_dirs: list[dict[str, str]],
) -> list[dict[str, str]]:
    protected = {
        str(item.get("path") or "")
        for item in runtime_dirs
        if str(item.get("status") or "") == "ready"
    }
    if not protected:
        return []
    blocked: list[dict[str, str]] = []
    for item in queue_items:
        if item.get("state") not in {"queued", "running"}:
            continue
        run_dir = _queue_item_runtime_delete_dir(item)
        if run_dir is None:
            continue
        label = _display_settings_path(run_dir)
        if label in protected:
            blocked.append({
                "id": str(item.get("id") or ""),
                "path": label,
                "reason": "运行目录仍被等待或运行中的队列项引用",
            })
    return blocked


def _delete_history_tasks(task_ids: list[str]) -> dict[str, Any]:
    cleanup_errors: dict[str, str] = {}
    deleted_task_ids: list[str] = []
    for task_id in task_ids:
        result = _delete_history_task(task_id)
        deleted_task_ids.append(task_id)
        if result.get("cleanup_error"):
            cleanup_errors[task_id] = str(result.get("cleanup_error"))

    linked_count = max(0, len(deleted_task_ids) - 1)
    message = "任务已删除"
    if linked_count:
        message = f"任务已删除，并一并删除 {linked_count} 个对应预处理任务"
    if cleanup_errors:
        message = "任务已从列表移除，部分磁盘残留稍后可手动清理"
    payload: dict[str, Any] = {
        "ok": True,
        "message": message,
        "deleted_task_ids": deleted_task_ids,
        "linked_preprocess_deleted": linked_count,
    }
    if cleanup_errors:
        payload["cleanup_errors"] = cleanup_errors
        payload["cleanup_error"] = "; ".join(
            f"{key}: {value}" for key, value in cleanup_errors.items()
        )
    return payload


def _delete_history_task(task_id: str) -> dict[str, Any]:
    task_dir = _history_task_dir(task_id)
    if not _path_exists(task_dir):
        raise FileNotFoundError("任务不存在")
    deleting_dir = _reserve_deleting_history_dir(task_dir)
    try:
        task_dir.rename(deleting_dir)
    except OSError as exc:
        raise ValueError(f"删除任务失败: {exc}") from exc

    try:
        shutil.rmtree(deleting_dir)
    except OSError as exc:
        # 先改名再清理，避免异常文件导致前端列表一直卡着删不掉。
        return {
            "ok": True,
            "message": "任务已从列表移除，部分磁盘残留稍后可手动清理",
            "cleanup_error": str(exc),
        }
    return {"ok": True, "message": "任务已删除"}


def _history_summary(meta: dict[str, Any], task_dir: Path) -> dict[str, Any]:
    out = dict(meta)
    out["id"] = task_dir.name
    out["name"] = str(out.get("name") or "")
    out["group"] = str(out.get("group") or "")
    if not str(out.get("training_mode") or "").strip():
        out["training_mode"] = "continue_lora" if out.get("continue_from_weight_abs_path") else "fresh"
    for key in (
        "continue_from_weight_abs_path",
        "continue_from_weight_name",
        "continue_from_weight_kind",
    ):
        out[key] = str(out.get(key) or "")
    out["archived"] = _history_task_archived(out)
    out["project_root_abs"] = str(ROOT.resolve())
    out["history_dir"] = _display_project_path(str(task_dir))
    out["history_dir_abs"] = str(task_dir)
    out["config_snapshot"] = _display_project_path(str(task_dir / "config.snapshot.toml"))
    out["logs_path"] = _display_project_path(str(task_dir / "logs.jsonl"))
    out["metrics_path"] = _display_project_path(str(task_dir / "metrics.jsonl"))
    out["system_path"] = _display_project_path(str(task_dir / "system.jsonl"))
    data_dirs = out.get("data_dirs") if isinstance(out.get("data_dirs"), dict) else {}
    for key in ("source_image_dir", "resized_image_dir", "lora_cache_dir"):
        out[key] = str(out.get(key) or data_dirs.get(key) or "")
    _fill_history_runtime_meta(out)
    out["run_dir_abs"] = _absolute_display_path(out.get("run_dir"))
    _fill_history_group_meta(out)
    if not out["name"]:
        out["name"] = _default_preprocess_history_name(out)
    out["log_count"] = _history_jsonl_count(out, "log_count", task_dir / "logs.jsonl")
    out["metric_count"] = _history_jsonl_count(out, "metric_count", task_dir / "metrics.jsonl")
    return out


def _history_jsonl_count(meta: dict[str, Any], key: str, path: Path) -> int:
    if key in meta:
        count = _int_or_none(meta.get(key))
        if count is not None and count >= 0:
            return count
    return _count_jsonl(path)


def _safe_history_summary(meta: dict[str, Any], task_dir: Path) -> dict[str, Any] | None:
    try:
        return _history_summary(meta, task_dir)
    except (OSError, TypeError, ValueError):
        return None


def _repair_history_meta(meta_path: Path, meta: dict[str, Any]) -> None:
    before = dict(meta)
    _fill_history_runtime_meta(meta)
    _fill_history_group_meta(meta)
    if str(meta.get("job") or "").strip() == "preprocess":
        # 旧版本写入 archived=false；没有 updated_at 表示用户没有手动取消归档。
        if "updated_at" not in meta and meta.get("archived") is not True:
            meta["archived"] = True
        name = _default_preprocess_history_name(meta)
        if name and _is_legacy_auto_preprocess_name(meta.get("name"), name):
            meta["name"] = name
    if meta != before:
        try:
            _write_json_atomic(meta_path, meta)
        except OSError:
            pass


def _is_deleting_history_dir(task_dir: Path) -> bool:
    return ".deleting-" in task_dir.name


def _reserve_deleting_history_dir(task_dir: Path) -> Path:
    base = f".{task_dir.name}.deleting-{int(time.time() * 1000)}"
    candidate = task_dir.with_name(base)
    suffix = 1
    while _path_exists(candidate):
        suffix += 1
        candidate = task_dir.with_name(f"{base}-{suffix}")
    return candidate


def _default_history_archived(job: str) -> bool:
    return str(job or "").strip() == "preprocess"


def _history_task_archived(task: dict[str, Any]) -> bool:
    archived = bool(task.get("archived", False))
    if archived:
        return True
    if str(task.get("job") or "").strip() != "preprocess":
        return False
    # 旧版本预处理占位默认写成 archived=false。没有用户更新痕迹时，
    # 读取时按新的默认规则隐藏；用户手动取消归档后会带 updated_at。
    return "updated_at" not in task


def _default_preprocess_history_name(task: dict[str, Any]) -> str:
    if str(task.get("job") or "").strip() == "training" and str(task.get("training_mode") or "") == "continue_lora":
        kind = str(task.get("continue_from_weight_kind") or "LoRA").strip() or "LoRA"
        name = str(task.get("continue_from_weight_name") or "").strip()
        suffix = f" · {name}" if name else ""
        return f"权重热启动 {kind}{suffix}"
    if str(task.get("job") or "").strip() != "preprocess":
        return ""
    label = str(task.get("history_run_label") or "").strip()
    if not label:
        label = _history_run_label_from_runtime(
            str(task.get("output_dir") or ""),
            _runtime_meta(task),
            str(task.get("id") or ""),
        )
    label = label or str(task.get("id") or "").strip()
    if not label:
        return "预处理"
    return label


def _is_legacy_auto_preprocess_name(value: Any, default_name: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    return bool(default_name) and text == f"预处理 {default_name}"


def _fill_history_runtime_meta(task: dict[str, Any]) -> None:
    run_dir_raw = str(task.get("run_dir") or "").strip()
    if not run_dir_raw:
        output_dir = _resolve_display_path(str(task.get("training_output_dir") or task.get("output_dir") or ""))
        if output_dir and output_dir.name == "training_output":
            run_dir_raw = _display_project_path(str(output_dir.parent))
            task["run_dir"] = run_dir_raw
    run_dir = _resolve_display_path(run_dir_raw)
    if not run_dir:
        return

    defaults = {
        "runtime_config_file": run_dir / "config.runtime.toml",
        "original_config_file": run_dir / "config.original.toml",
        "dataset_config_file": run_dir / "dataset.runtime.toml",
        "model_cache_dir": run_dir / "model_cache",
        "dataset_cache_dir": run_dir / "dataset_cache",
        "training_output_dir": run_dir / "training_output",
        "logs_dir": run_dir / "model_cache" / "logs",
    }
    for key, path in defaults.items():
        if not str(task.get(key) or "").strip():
            task[key] = _display_project_path(str(path))


def _history_snapshot_path(task_id: str) -> Path | None:
    task_dir = _history_task_dir(task_id)
    snapshot = task_dir / "config.snapshot.toml"
    if _path_exists(snapshot):
        return snapshot
    return None


def _list_resume_checkpoints(task: dict[str, Any]) -> list[dict[str, Any]]:
    output_dir = _resolve_display_path(str(task.get("output_dir") or ""))
    if output_dir is None or not _path_exists(output_dir) or not output_dir.is_dir():
        return []

    started_at = _float_or_none(task.get("started_at"))
    finished_at = _float_or_none(task.get("finished_at"))
    lower = started_at - 180 if started_at is not None else None
    upper = (finished_at + 180) if finished_at is not None else (datetime.now().timestamp() + 180)

    items: list[dict[str, Any]] = []
    for child in sorted(output_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if _is_transient_resume_state_dir(child.name):
            continue
        state_file = child / "train_state.json"
        if not _path_exists(state_file):
            continue
        state = _read_json(state_file)
        step = _int_or_none(state.get("current_step"))
        if step is None:
            continue
        epoch = _int_or_none(state.get("current_epoch"))
        mtime = _state_mtime(child, state_file)
        scope = "task" if lower is not None and lower <= mtime <= upper else "other"
        if scope != "task":
            continue
        kind = _resume_state_kind(child.name)
        paired_weight = _paired_resume_weight(child, output_dir)
        integrity = _resume_state_integrity(child)
        items.append({
            "id": _display_project_path(str(child)),
            "path": _display_project_path(str(child)),
            "name": child.name,
            "kind": kind,
            "kind_label": _resume_state_kind_label(kind),
            "scope": scope,
            "scope_label": "本任务" if scope == "task" else "同目录其他训练",
            "epoch": epoch,
            "step": step,
            "current_epoch": epoch,
            "current_step": step,
            "mtime": mtime,
            "mtime_text": _format_ts(mtime),
            "train_state_file": _display_project_path(str(state_file)),
            "paired_weight": paired_weight,
            "state_integrity": integrity,
            "state_complete": bool(integrity.get("ok")),
            "missing_state_files": list(integrity.get("missing") or []),
        })

    items.sort(key=_resume_state_sort_key)
    return items


def _resume_checkpoint_diagnostic(task: dict[str, Any], checkpoints: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    raw_output_dir = str(task.get("output_dir") or "")
    output_dir = _resolve_display_path(raw_output_dir)
    diagnostic: dict[str, Any] = {
        "output_dir": raw_output_dir,
        "output_dir_resolved": _display_project_path(str(output_dir)) if output_dir is not None else "",
        "output_dir_valid": output_dir is not None,
        "output_dir_exists": bool(output_dir is not None and _path_exists(output_dir)),
        "output_dir_is_dir": bool(output_dir is not None and _path_exists(output_dir) and output_dir.is_dir()),
        "all_subdir_count": 0,
        "state_dir_count": 0,
        "train_state_count": 0,
        "complete_state_count": 0,
        "incomplete_state_count": 0,
        "missing_state_files": [],
        "checkpoint_count": len(checkpoints or []),
        "reason": "",
        "recommendation": "如需权重热启动，可回到配置页选择这个任务导出的 LoRA/LoHa/LoKr/GLoRA 权重；热启动不会恢复 optimizer、scheduler 和已完成步数。",
    }
    if output_dir is None:
        diagnostic["reason"] = "这个历史任务记录的输出目录不合法，无法扫描完整续训状态。"
        return diagnostic
    if not _path_exists(output_dir):
        diagnostic["reason"] = "这个历史任务记录的输出目录不存在，完整续训所需的 train_state.json 状态目录无法读取。"
        return diagnostic
    if not output_dir.is_dir():
        diagnostic["reason"] = "这个历史任务记录的输出路径不是目录，无法扫描完整续训状态。"
        return diagnostic

    all_subdirs = [
        child
        for child in output_dir.iterdir()
        if child.is_dir() and not _is_transient_resume_state_dir(child.name)
    ]
    state_dirs = [child for child in all_subdirs if child.name.endswith("-state")]
    diagnostic["all_subdir_count"] = len(all_subdirs)
    diagnostic["state_dir_count"] = len(state_dirs)
    diagnostic["train_state_count"] = sum(1 for child in state_dirs if _path_exists(child / "train_state.json"))
    integrity_items = [
        item.get("state_integrity")
        for item in (checkpoints or [])
        if isinstance(item.get("state_integrity"), dict)
    ]
    diagnostic["complete_state_count"] = sum(1 for item in integrity_items if item.get("ok"))
    diagnostic["incomplete_state_count"] = sum(1 for item in integrity_items if not item.get("ok"))
    missing_files: list[str] = []
    for item in integrity_items:
        missing_files.extend(str(name) for name in (item.get("missing") or []))
    diagnostic["missing_state_files"] = sorted(set(missing_files))
    resume_from = task.get("resume_from") if isinstance(task.get("resume_from"), dict) else {}
    resume_checkpoint = str(resume_from.get("checkpoint") or "").strip()
    resume_checkpoint_path = _resolve_display_path(resume_checkpoint) if resume_checkpoint else None
    resume_train_state_exists = bool(
        resume_checkpoint_path is not None and _path_exists(resume_checkpoint_path / "train_state.json")
    )
    diagnostic["resume_source_checkpoint"] = resume_checkpoint
    diagnostic["resume_source_train_state_exists"] = resume_train_state_exists
    if diagnostic["complete_state_count"]:
        diagnostic["reason"] = "已找到可完整续训的状态目录。"
    elif checkpoints and diagnostic["incomplete_state_count"]:
        diagnostic["reason"] = (
            "找到包含 train_state.json 的状态目录，但缺少完整续训必需的 "
            f"{'、'.join(diagnostic['missing_state_files'])}，无法恢复 optimizer/scheduler 状态。"
        )
    elif resume_checkpoint and not resume_train_state_exists and int(task.get("metric_count") or 0) == 0:
        diagnostic["reason"] = "这次续训没有产生训练步，且完整续训状态目录已不存在；可用缓存/权重仍可能存在，但 optimizer/scheduler 状态无法恢复。"
    elif diagnostic["train_state_count"]:
        diagnostic["reason"] = "输出目录里存在 train_state.json 状态目录，但不属于当前历史任务时间范围。"
    elif diagnostic["state_dir_count"]:
        diagnostic["reason"] = "输出目录里有子目录，但没有包含 train_state.json 的完整续训状态目录。"
    else:
        diagnostic["reason"] = "输出目录里没有完整续训状态目录；旧版本训练完成时 checkpoint-state 可能已被清理，或该配置未写出训练状态。"
    return diagnostic


def _is_transient_resume_state_dir(name: str) -> bool:
    return name.endswith((".tmp", ".backup"))


def _resume_state_integrity(state_dir: Path) -> dict[str, Any]:
    """Check the minimum Accelerate files needed for a real full resume."""
    checks = {
        "train_state": _path_exists(state_dir / "train_state.json"),
        "model": _state_has_any_file(state_dir, ("model.safetensors", "model_*.safetensors", "pytorch_model*.bin")),
        "optimizer": _state_has_any_file(state_dir, ("optimizer.bin", "optimizer_*.bin", "optimizer*.bin")),
        "scheduler": _state_has_any_file(state_dir, ("scheduler.bin", "scheduler_*.bin", "scheduler*.bin")),
        "random_state": _state_has_any_file(state_dir, ("random_states_*.pkl", "random_state*.pkl")),
    }
    labels = {
        "train_state": "train_state.json",
        "model": "model.safetensors",
        "optimizer": "optimizer.bin",
        "scheduler": "scheduler.bin",
    }
    missing = [label for key, label in labels.items() if not checks.get(key)]
    return {
        "ok": not missing,
        "missing": missing,
        **checks,
    }


def _state_has_any_file(state_dir: Path, patterns: tuple[str, ...]) -> bool:
    for pattern in patterns:
        if any(state_dir.glob(pattern)):
            return True
    return False


def _resume_state_integrity_unavailable_reason(integrity: dict[str, Any] | None) -> str:
    if not isinstance(integrity, dict) or integrity.get("ok"):
        return ""
    missing = [str(name) for name in (integrity.get("missing") or []) if str(name)]
    if not missing:
        return ""
    return (
        "这个状态目录不完整，缺少 "
        f"{'、'.join(missing)}，无法完整恢复 optimizer、scheduler 和步数。"
    )


def _select_resume_checkpoint(
    checkpoints: list[dict[str, Any]],
    checkpoint: str | None,
) -> dict[str, Any] | None:
    if not checkpoints:
        return None
    if not checkpoint:
        return checkpoints[0]

    target = _resolve_display_path(checkpoint)
    if target is None:
        return None
    target_text = _display_project_path(str(target))
    for item in checkpoints:
        if _display_project_path(str(item.get("path") or "")) == target_text:
            return item
    return None


def _resume_state_kind(name: str) -> str:
    if re.search(r"-checkpoint-\d{6}-state$", name):
        return "checkpoint"
    if name.endswith("-checkpoint-state"):
        return "checkpoint"
    if re.search(r"-step\d+-state$", name):
        return "step"
    if re.search(r"-\d{6}-state$", name):
        return "epoch"
    if name.endswith("-state"):
        return "last"
    return "state"


def _resume_state_kind_label(kind: str) -> str:
    return {
        "checkpoint": "自动续训检查点",
        "step": "按步保存状态",
        "epoch": "按轮保存状态",
        "last": "训练结束状态",
        "state": "训练状态",
    }.get(kind, "训练状态")


def _resume_state_sort_key(item: dict[str, Any]) -> tuple[int, int, int, float, str]:
    scope_rank = {"task": 0, "other": 1}
    kind_rank = {"checkpoint": 0, "last": 1, "epoch": 2, "step": 3, "state": 4}
    step = int(item.get("step") or -1)
    return (
        int(scope_rank.get(str(item.get("scope")), 9)),
        int(kind_rank.get(str(item.get("kind")), 9)),
        -step,
        -float(item.get("mtime") or 0),
        str(item.get("name") or ""),
    )


def _state_mtime(state_dir: Path, state_file: Path) -> float:
    for path in (state_file, state_dir):
        try:
            return float(path.stat().st_mtime)
        except OSError:
            continue
    return datetime.now().timestamp()


def _paired_resume_weight(state_dir: Path, output_dir: Path) -> str:
    name = state_dir.name
    if not name.endswith("-state"):
        return ""
    base_name = name[:-6]
    weight = output_dir / f"{base_name}.safetensors"
    if _path_exists(weight):
        return _display_project_path(str(weight))
    return ""


def _read_json(path: Path) -> dict[str, Any]:
    data = _read_json_object(path)
    return data if isinstance(data, dict) else {}


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not _path_exists(path):
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, indent=2))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        _fsync_parent_dir(path)
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def _fsync_parent_dir(path: Path) -> None:
    try:
        fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return _read_jsonl_limited(path)[0]


def _read_jsonl_limited(path: Path, *, limit: int | None = None) -> tuple[list[dict[str, Any]], int, bool]:
    if not _path_exists(path):
        return [], 0, False
    out: list[dict[str, Any]] = []
    try:
        lines = [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    except OSError:
        return [], 0, False
    total = len(lines)
    safe_limit = _positive_int_or_none(limit)
    truncated = bool(safe_limit and total > safe_limit)
    if safe_limit:
        lines = lines[-safe_limit:]
    for line in lines:
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                out.append(value)
        except Exception:
            continue
    return out, total, truncated


def _count_jsonl(path: Path) -> int:
    if not _path_exists(path):
        return 0
    try:
        return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())
    except Exception:
        return 0


def _read_text_file(path: Path) -> str:
    if not _path_exists(path):
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _path_exists(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._path_exists(*args, **kwargs)


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


def _default_sample_config() -> dict[str, Any]:
    return {
        "enabled": False,
        "sample_prompts": "",
        "sample_prompts_exists": False,
        "sample_every_n_epochs": None,
        "sample_every_n_steps": None,
        "sample_at_first": False,
        "sample_sampler": "euler",
        "message": "未启用训练中采样",
    }


def _sample_config_from_cfg(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._sample_config_from_cfg(*args, **kwargs)


def _cli_arg_overrides(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._cli_arg_overrides(*args, **kwargs)


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


def _resolve_display_path(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._resolve_display_path(*args, **kwargs)


def _display_project_path(*args, **kwargs):
    from web.services.training import runtime_config as _impl
    return _impl._display_project_path(*args, **kwargs)


def _absolute_display_path(value: Any) -> str:
    path = _resolve_display_path(str(value or ""))
    return str(path) if path is not None else ""


def _command_has_option(args: list[str], option: str) -> bool:
    prefix = f"{option}="
    return any(arg == option or str(arg).startswith(prefix) for arg in args)


def _command_option_value(args: list[str], option: str) -> str | None:
    prefix = f"{option}="
    for idx, arg in enumerate(args):
        if arg == option and idx + 1 < len(args):
            return str(args[idx + 1])
        if str(arg).startswith(prefix):
            return str(arg).split("=", 1)[1]
    return None


def _resolve_block_swap_profile_auto_arg(args: list[str], path: Path) -> list[str]:
    return _resolve_auto_path_arg(args, "--block_swap_profile_jsonl", path)


def _resolve_block_swap_profile_auto_config(config_file: str | None, path: Path) -> bool:
    return _resolve_auto_path_config(
        config_file,
        config_key="block_swap_profile_jsonl",
        path=path,
        is_history_path_fn=_is_history_block_swap_profile_path,
    )


def _resolve_memory_probe_auto_arg(args: list[str], path: Path) -> list[str]:
    return _resolve_auto_path_arg(args, "--memory_probe_jsonl", path)


def _resolve_memory_probe_auto_config(config_file: str | None, path: Path) -> bool:
    return _resolve_auto_path_config(
        config_file,
        config_key="memory_probe_jsonl",
        path=path,
        is_history_path_fn=_is_history_memory_probe_path,
    )


def _resolve_peak_probe_auto_arg(args: list[str], path: Path) -> list[str]:
    return _resolve_auto_path_arg(args, "--peak_probe_jsonl", path)


def _resolve_peak_probe_auto_config(config_file: str | None, path: Path) -> bool:
    return _resolve_auto_path_config(
        config_file,
        config_key="peak_probe_jsonl",
        path=path,
        is_history_path_fn=_is_history_peak_probe_path,
    )


def _resolve_auto_path_arg(args: list[str], option: str, path: Path) -> list[str]:
    out = list(args)
    prefix = f"{option}="
    replacement = str(path)
    idx = 0
    while idx < len(out):
        arg = str(out[idx])
        if arg == option and idx + 1 < len(out):
            if str(out[idx + 1]).strip().lower() == "auto":
                out[idx + 1] = replacement
            idx += 2
            continue
        if arg.startswith(prefix) and arg.split("=", 1)[1].strip().lower() == "auto":
            out[idx] = f"{option}={replacement}"
        idx += 1
    return out


def _resolve_auto_path_config(
    config_file: str | None,
    *,
    config_key: str,
    path: Path,
    is_history_path_fn,
) -> bool:
    config_path = _resolve_display_path(str(config_file or ""))
    if config_path is None or not _path_exists(config_path) or not config_path.is_file():
        return False
    try:
        cfg = toml.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    value = str(cfg.get(config_key) or "").strip()
    if value.lower() != "auto" and not is_history_path_fn(value):
        return False
    if config_path.name == "config.runtime.toml":
        cfg[config_key] = str(path)
        config_path.write_text(toml_dumps_sorted(cfg), encoding="utf-8")
    return True


def _is_history_block_swap_profile_path(value: str) -> bool:
    return _is_history_artifact_path(value, "block_swap_profile.jsonl")


def _is_history_memory_probe_path(value: str) -> bool:
    return _is_history_artifact_path(value, "memory_probe.jsonl")


def _is_history_peak_probe_path(value: str) -> bool:
    return _is_history_artifact_path(value, "peak_probe.jsonl")


def _is_history_artifact_path(value: str, filename: str) -> bool:
    artifact_path = _resolve_display_path(value)
    if artifact_path is None or artifact_path.name != filename:
        return False
    try:
        artifact_path.resolve().relative_to(HISTORY_DIR.resolve())
    except ValueError:
        return False
    return True


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


def classify_training_error(text: str) -> str:
    """Return a short user-facing hint for known high-signal training failures."""
    if text and CUDA_OOM_RE.search(text):
        return OOM_HINT
    return ""


def format_training_anomaly(status_data: dict[str, Any]) -> str | None:
    """检测训练异常状态并生成可读的错误提示。

    Args:
        status_data: get_status_snapshot() 返回的完整状态字典

    Returns:
        格式化的错误提示文本，如果没有检测到异常则返回 None
    """
    latest_metric = status_data.get("latest_metric", {}) if isinstance(status_data, dict) else {}
    if not isinstance(latest_metric, dict) or not latest_metric:
        return None

    loss = latest_metric.get("loss")
    lr = latest_metric.get("lr")
    step = latest_metric.get("step")
    rate = str(latest_metric.get("rate") or "").strip() or "未知"

    anomaly_kind = _loss_anomaly_kind(loss)

    if anomaly_kind is None:
        return None

    title = {
        "nan": "损失值变为 NaN",
        "inf": "损失值变为无穷大",
    }.get(anomaly_kind, "损失值异常")

    lines = [
        f"⚠️ 训练异常：{title}",
        f"  • 发生步数：第 {step} 步" if step is not None else "  • 发生步数：未知",
        f"  • 当前学习率：{_format_anomaly_value(lr)}",
        f"  • 训练速度：{rate}",
    ]

    latest_system = status_data.get("latest_system", {})
    if isinstance(latest_system, dict) and latest_system:
        vram_used = _float_or_none(latest_system.get("vram_used_gb"))
        vram_total = _float_or_none(latest_system.get("vram_total_gb"))
        if (
            vram_used is not None
            and vram_total is not None
            and math.isfinite(vram_used)
            and math.isfinite(vram_total)
        ):
            vram_pct = (vram_used / vram_total * 100) if vram_total > 0 else 0
            lines.append(f"  • 显存占用：{vram_used:.2f}GB / {vram_total:.2f}GB ({vram_pct:.1f}%)")

    lines.extend([
        "",
        "常见原因（按可能性排序）：",
        "  1. 学习率过高",
        "     → 建议降至 5e-5 或更低，并添加 warmup_steps = 50",
        "  2. 混合精度数值溢出",
        "     → 尝试改用 bf16 或临时关闭混合精度 (mixed_precision = \"no\")",
        "  3. 缓存文件损坏",
        "     → 删除 *_anima*.npz 和 *_anima_te.safetensors 后重新运行预处理",
        "  4. 图片或 caption 异常",
        "     → 检查是否有全黑/全白图片或空 caption 文件",
    ])

    config_file = str(status_data.get("history_source_config_file") or "").strip()
    if config_file:
        config_name = Path(config_file).name
        lines.extend([
            "",
            f"配置文件：{config_name}",
            f"完整路径：{config_file}",
        ])

    lines.extend([
        "",
        "详细排查步骤请参考项目文档或查看训练日志。",
    ])

    return "\n".join(lines)


def _loss_anomaly_kind(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"nan", "+nan", "-nan"}:
            return "nan"
        if text in {"inf", "+inf", "infinity", "+infinity"}:
            return "inf"
        if text in {"-inf", "-infinity"}:
            return "inf"
    number = _float_or_none(value)
    if number is None:
        return None
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf"
    return None


def _format_anomaly_value(value: Any) -> str:
    if value is None or value == "":
        return "未知"
    number = _float_or_none(value)
    if number is None:
        return str(value)
    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "Infinity" if number > 0 else "-Infinity"
    return str(value)


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


def _message_with_error_hint(message: str, hint: str) -> str:
    if not hint or not message:
        return message
    if hint in message:
        return message
    return f"{message}：{hint}"


def _first_record_separator(text: str) -> int | None:
    indexes = [idx for idx in (text.find("\n"), text.find("\r")) if idx >= 0]
    return min(indexes) if indexes else None


def _clean_output_record(text: str) -> str:
    text = text.replace("\x1b[?25l", "").replace("\x1b[?25h", "")
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    return text.strip()


def _extract_float_metric(text: str, names: tuple[str, ...]) -> float | None:
    return _progress_parser.extract_float_metric(text, names)
