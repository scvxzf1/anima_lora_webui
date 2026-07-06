"""Delegated training service methods.

This module is a mechanical extraction from ``web.services.training_service``.
The public ``TrainingService`` class keeps the same method names and delegates
here so HTTP routes and WebSocket payloads remain unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncio
    import time
    from typing import Any

    from web.services.training_service import (
        MAX_QUEUE_ITEMS,
        QUEUE_CLEARABLE_STATES,
        QUEUE_DIR,
        QUEUE_TERMINAL_STATES,
        _clone_frozen_runtime_config,
        _delete_queue_item_runtime_dir,
        _display_project_path,
        _format_ts,
        _int_or_none,
        _load_config_file_config,
        _new_queue_item_id,
        _normalize_continue_lora_info,
        _normalize_gpu_whitelist,
        _normalize_queue_failure_policy,
        _positive_int_or_none,
        _prepare_web_runtime_config,
        _path_exists,
        _queue_clearable_state_label,
        _queue_item_runtime_dir_label,
        _resolve_display_path,
        _resume_state_integrity,
        _resume_state_integrity_unavailable_reason,
        _runtime_from_config_file,
        _runtime_meta,
        _sample_config_from_cfg,
        _write_training_queue_state,
    )


_LOCAL_IMPL_NAMES = {
    "_bind_legacy",
    "start_queue_on_startup",
    "get_queue_snapshot",
    "enqueue_training",
    "enqueue_training_batch",
    "enqueue_resume_from_history_task",
    "_ensure_queue_resume_checkpoint_exists",
    "_queue_resume_runtime_config_file",
    "move_queue_item",
    "cancel_queue_item",
    "retry_queue_item",
    "cancel_waiting_queue_items",
    "cancel_all_queue_items",
    "abort_queue_after_current",
    "force_abort_queue",
    "clear_finished_queue_items",
    "clear_completed_queue_items",
    "clear_canceled_queue_items",
    "clear_queue_items_by_state",
    "set_queue_settings",
    "set_queue_paused",
    "_clone_queue_item_for_retry",
    "_repair_queue_on_startup",
    "_normalize_queue",
    "_queue_items",
    "_find_queue_item",
    "_update_queue_item",
    "_attach_history_task_to_queue_item",
    "_queue_history_meta",
    "_compact_queue",
    "_save_queue",
    "_broadcast_queue",
    "_schedule_queue_dispatch",
    "_dispatch_queue",
    "_pause_queue_after_failure",
    "_start_queue_item",
    "_queue_item_runtime",
}


def _bind_legacy() -> None:
    """Bind legacy module globals lazily after training_service has loaded."""
    from web.services import training_service as legacy

    for name, value in vars(legacy).items():
        if name.startswith("__") or name in _LOCAL_IMPL_NAMES:
            continue
        globals()[name] = value


async def start_queue_on_startup(self) -> None:
    _bind_legacy()
    self._schedule_queue_dispatch()

def get_queue_snapshot(self) -> dict[str, Any]:
    _bind_legacy()
    self._normalize_queue()
    summary = {
        "total": 0,
        "queued": 0,
        "running": 0,
        "done": 0,
        "error": 0,
        "canceled": 0,
    }
    for item in self._queue_items():
        summary["total"] += 1
        state = str(item.get("state") or "")
        if state in summary:
            summary[state] += 1
    return {
        "ok": True,
        "paused": self._queue_paused,
        "failure_policy": self._queue_failure_policy,
        "status": self.status,
        "current_item_id": self._current_queue_item_id,
        "summary": summary,
        "items": [dict(item) for item in self._queue_items()],
    }

async def enqueue_training(
    self,
    variant: str,
    preset: str,
    methods_subdir: str = "gui-methods",
    *,
    extra_args: list[str] | None = None,
    config_file: str | None = None,
    gpu_whitelist: list[Any] | None = None,
    continue_info: dict[str, Any] | None = None,
    requires_preprocess: bool = True,
    start_paused: bool = False,
) -> dict[str, Any]:
    _bind_legacy()
    extra = list(extra_args or [])
    gpu_selection = _normalize_gpu_whitelist(gpu_whitelist)
    runtime = None
    runtime_config_file = str(config_file or "").strip()
    source_config_file = str(config_file or "").strip()
    if requires_preprocess:
        runtime = _prepare_web_runtime_config(
            variant,
            preset,
            methods_subdir,
            source_config_file=config_file,
        )
        runtime_config_file = runtime["runtime_config_file"]
        source_config_file = runtime.get("history_source_config_file") or source_config_file

    continue_payload = _normalize_continue_lora_info(
        continue_info,
        variant=variant,
        preset=preset,
        methods_subdir=methods_subdir,
        config_file=runtime_config_file or config_file,
    )
    now = time.time()
    item = {
        "id": _new_queue_item_id("training", methods_subdir, variant),
        "state": "queued",
        "kind": "training",
        "requires_preprocess": bool(requires_preprocess),
        "variant": variant,
        "preset": preset,
        "methods_subdir": methods_subdir,
        "runtime_config_file": runtime_config_file,
        "source_config_file": source_config_file,
        "extra_args": extra,
        "gpu_whitelist": gpu_selection,
        "continue_info": continue_payload or {},
        "resume_info": {},
        "retry_of": "",
        "attempt": 1,
        "history_task_ids": [],
        "message": "等待队列调度",
        "created_at": now,
        "created_at_text": _format_ts(now),
        "started_at": None,
        "started_at_text": "",
        "finished_at": None,
        "finished_at_text": "",
        "runtime_info": _runtime_meta(runtime) if runtime else {},
    }
    if start_paused:
        self._queue_paused = True
        self._queue["paused"] = True
    self._queue_items().append(item)
    self._compact_queue()
    self._save_queue()
    await self._broadcast_queue()
    if not self._queue_paused:
        self._schedule_queue_dispatch()
    message = "已加入训练队列，队列已暂停" if start_paused else "已加入训练队列"
    return {"ok": True, "message": message, "item": dict(item), **self.get_queue_snapshot()}

async def enqueue_training_batch(
    self,
    entries: list[dict[str, Any]],
    *,
    default_preset: str = "default",
    gpu_whitelist: list[Any] | None = None,
    start_paused: bool = True,
) -> dict[str, Any]:
    _bind_legacy()
    if not isinstance(entries, list) or not entries:
        raise ValueError("items 必须是非空数组")

    queued: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    gpu_selection = _normalize_gpu_whitelist(gpu_whitelist)
    if start_paused:
        self._queue_paused = True
        self._queue["paused"] = True

    for index, raw in enumerate(entries):
        if not isinstance(raw, dict):
            failures.append({
                "index": index,
                "error": "队列项格式不合法",
            })
            break
        variant = str(raw.get("variant") or "").strip()
        preset = str(raw.get("preset") or default_preset or "default").strip() or "default"
        methods_subdir = str(raw.get("methods_subdir") or "gui-methods").strip() or "gui-methods"
        config_file = str(raw.get("config_file") or "").strip() or None
        label = str(raw.get("label") or raw.get("filename") or raw.get("display_name") or "").strip()
        requires_preprocess = bool(raw.get("requires_preprocess", True))
        if not variant:
            failure = {
                "index": index,
                "config_file": config_file or "",
                "error": "缺少 variant",
            }
            if label:
                failure["label"] = label
            failures.append(failure)
            break
        try:
            payload = await self.enqueue_training(
                variant,
                preset,
                methods_subdir,
                extra_args=list(raw.get("extra_args") or []),
                config_file=config_file,
                gpu_whitelist=gpu_selection,
                continue_info=raw.get("continue_info") if isinstance(raw.get("continue_info"), dict) else None,
                requires_preprocess=requires_preprocess,
                start_paused=start_paused,
            )
        except Exception as exc:
            failure = {
                "index": index,
                "variant": variant,
                "preset": preset,
                "methods_subdir": methods_subdir,
                "config_file": config_file or "",
                "error": str(exc),
            }
            if label:
                failure["label"] = label
            failures.append(failure)
            break
        queued.append(dict(payload.get("item") or {}))

    snapshot = self.get_queue_snapshot()
    queued_count = len(queued)
    total = len(entries)
    ok = not failures
    if ok:
        message = f"已将 {queued_count} 个配置加入训练队列"
    elif queued_count:
        message = f"已加入 {queued_count} 个配置，批量加入在第 {failures[0]['index'] + 1} 项停止"
    else:
        message = "批量加入队列失败"
    return {
        **snapshot,
        "ok": ok,
        "message": message,
        "queued_count": queued_count,
        "requested_count": total,
        "queued_items": queued,
        "failures": failures,
    }

async def enqueue_resume_from_history_task(
    self,
    task_id: str,
    checkpoint: str | None = None,
    *,
    duration_overrides: dict[str, Any] | None = None,
    gpu_whitelist: list[Any] | None = None,
) -> dict[str, Any]:
    _bind_legacy()
    task, selected, snapshot_path, resume_info = self._build_resume_payload(
        task_id,
        checkpoint,
        duration_overrides=duration_overrides,
    )
    runtime = _clone_frozen_runtime_config(
        _display_project_path(str(snapshot_path)),
        source_config_file=str(task.get("history_source_config_file") or ""),
        reset_data_dirs=False,
        resume_step=_int_or_none(selected.get("step")),
        duration_overrides=duration_overrides,
    )
    if isinstance(runtime.get("resume_duration"), dict) and runtime["resume_duration"]:
        resume_info["duration_overrides"] = runtime["resume_duration"]
        resume_info["target_total_steps"] = runtime["resume_duration"].get("target_total_steps")
        resume_info["remaining_steps"] = runtime["resume_duration"].get("append_steps")
    now = time.time()
    item = {
        "id": _new_queue_item_id("resume", str(task.get("methods_subdir") or "gui-methods"), str(task.get("variant") or "training")),
        "state": "queued",
        "kind": "resume",
        "requires_preprocess": False,
        "variant": str(task.get("variant") or ""),
        "preset": str(task.get("preset") or "default"),
        "methods_subdir": str(task.get("methods_subdir") or "gui-methods"),
        "runtime_config_file": str(runtime.get("runtime_config_file") or _display_project_path(str(snapshot_path))),
        "source_config_file": str(runtime.get("history_source_config_file") or task.get("history_source_config_file") or ""),
        "extra_args": ["--resume", selected["path"], "--skip_until_initial_step"],
        "gpu_whitelist": _normalize_gpu_whitelist(gpu_whitelist),
        "continue_info": {},
        "resume_info": resume_info,
        "retry_of": "",
        "attempt": 1,
        "history_task_ids": [],
        "message": "等待续训队列调度",
        "created_at": now,
        "created_at_text": _format_ts(now),
        "started_at": None,
        "started_at_text": "",
        "finished_at": None,
        "finished_at_text": "",
        "runtime_info": _runtime_meta(runtime),
    }
    self._queue_items().append(item)
    self._compact_queue()
    self._save_queue()
    await self._broadcast_queue()
    self._schedule_queue_dispatch()
    return {
        "ok": True,
        "message": "续训任务已加入队列",
        "item": dict(item),
        "checkpoint": selected,
        **self.get_queue_snapshot(),
    }

async def move_queue_item(self, item_id: str, direction: str) -> dict[str, Any]:
    _bind_legacy()
    direction = str(direction or "").strip()
    items = self._queue_items()
    queued_indices = [i for i, item in enumerate(items) if item.get("state") == "queued"]
    index = next((i for i in queued_indices if items[i].get("id") == item_id), None)
    if index is None:
        raise ValueError("只能移动等待中的队列任务")
    position = queued_indices.index(index)
    if direction == "up" and position > 0:
        other = queued_indices[position - 1]
        items[index], items[other] = items[other], items[index]
    elif direction == "down" and position < len(queued_indices) - 1:
        other = queued_indices[position + 1]
        items[index], items[other] = items[other], items[index]
    elif direction == "top" and position > 0:
        item = items.pop(index)
        items.insert(queued_indices[0], item)
    elif direction == "bottom" and position < len(queued_indices) - 1:
        item = items.pop(index)
        insert_at = queued_indices[-1]
        if index < insert_at:
            insert_at -= 1
        items.insert(insert_at + 1, item)
    else:
        return self.get_queue_snapshot()
    self._save_queue()
    await self._broadcast_queue()
    return self.get_queue_snapshot()

async def cancel_queue_item(self, item_id: str, *, delete_runtime: bool = False) -> dict[str, Any]:
    _bind_legacy()
    item = self._find_queue_item(item_id)
    if item is None:
        raise FileNotFoundError("队列任务不存在")
    if item.get("state") == "running" and item_id == self._current_queue_item_id:
        await self.stop()
        return self.get_queue_snapshot()
    if item.get("state") in QUEUE_TERMINAL_STATES:
        deleted_runtime = False
        runtime_dir = ""
        if delete_runtime:
            runtime_dir = _queue_item_runtime_dir_label(item)
            now = time.time()
            item.update({
                "cleanup_state": "deleting_runtime",
                "cleanup_runtime_dir": runtime_dir,
                "cleanup_error": "",
                "cleanup_started_at": now,
                "cleanup_started_at_text": _format_ts(now),
            })
            self._save_queue()
            try:
                delete_result = _delete_queue_item_runtime_dir(item)
                deleted_runtime = bool(delete_result.get("deleted"))
                runtime_dir = str(delete_result.get("runtime_dir") or runtime_dir)
            except Exception as e:
                item.update({
                    "cleanup_state": "error",
                    "cleanup_error": str(e),
                })
                self._save_queue()
                await self._broadcast_queue()
                raise
        before = len(self._queue_items())
        self._queue["items"] = [
            entry for entry in self._queue_items()
            if str(entry.get("id") or "") != str(item_id or "")
        ]
        removed = before - len(self._queue_items())
        if removed:
            self._save_queue()
            await self._broadcast_queue()
        message = "已删除队列记录和运行缓存" if delete_runtime else "已从队列列表移除"
        return {
            "ok": True,
            "message": message,
            "deleted": removed,
            "deleted_runtime": deleted_runtime,
            "runtime_dir": runtime_dir,
            **self.get_queue_snapshot(),
        }
    if item.get("state") != "queued":
        raise ValueError("只能取消等待中的队列任务或删除已结束记录")
    now = time.time()
    item.update({
        "state": "canceled",
        "message": "已取消",
        "finished_at": now,
        "finished_at_text": _format_ts(now),
    })
    self._save_queue()
    await self._broadcast_queue()
    return self.get_queue_snapshot()

async def retry_queue_item(self, item_id: str) -> dict[str, Any]:
    _bind_legacy()
    item = self._find_queue_item(item_id)
    if item is None:
        raise FileNotFoundError("队列任务不存在")
    if item.get("state") == "running":
        raise ValueError("运行中的队列任务不能重新入队")
    retry = self._clone_queue_item_for_retry(item)
    self._queue_items().append(retry)
    self._compact_queue()
    self._save_queue()
    await self._broadcast_queue()
    self._schedule_queue_dispatch()
    return {"ok": True, "message": "已重新加入队列", "item": dict(retry), **self.get_queue_snapshot()}

async def cancel_waiting_queue_items(self) -> dict[str, Any]:
    _bind_legacy()
    now = time.time()
    count = 0
    for item in self._queue_items():
        if item.get("state") != "queued":
            continue
        item.update({
            "state": "canceled",
            "message": "已批量取消",
            "finished_at": now,
            "finished_at_text": _format_ts(now),
        })
        count += 1
    if count:
        self._save_queue()
        await self._broadcast_queue()
    return {"ok": True, "message": f"已取消 {count} 个等待任务", "canceled": count, **self.get_queue_snapshot()}

async def cancel_all_queue_items(self) -> dict[str, Any]:
    _bind_legacy()
    now = time.time()
    waiting_count = 0
    stale_running_count = 0
    stop_running = False
    running_item_id = str(self._current_queue_item_id or "")
    for item in self._queue_items():
        state = item.get("state")
        if state == "queued":
            item.update({
                "state": "canceled",
                "message": "已一键取消队列",
                "finished_at": now,
                "finished_at_text": _format_ts(now),
            })
            waiting_count += 1
        elif state == "running":
            item_id = str(item.get("id") or "")
            if item_id == running_item_id and self.process and self.process.returncode is None:
                stop_running = True
            else:
                item.update({
                    "state": "canceled",
                    "message": "已一键取消队列",
                    "finished_at": now,
                    "finished_at_text": _format_ts(now),
                })
                stale_running_count += 1
    if waiting_count or stale_running_count:
        self._save_queue()
        await self._broadcast_queue()
    if stop_running:
        await self.stop()
    canceled = waiting_count + stale_running_count + (1 if stop_running else 0)
    return {
        "ok": True,
        "message": f"已取消 {canceled} 个队列任务",
        "canceled": canceled,
        "canceled_waiting": waiting_count,
        "stopped_running": 1 if stop_running else 0,
        **self.get_queue_snapshot(),
    }

async def abort_queue_after_current(self) -> dict[str, Any]:
    _bind_legacy()
    async with self._launch_lock:
        now = time.time()
        canceled_waiting = 0
        self._queue_paused = True
        self._queue["paused"] = True
        for item in self._queue_items():
            if item.get("state") != "queued":
                continue
            item.update({
                "state": "canceled",
                "message": "已中止后续队列，当前运行任务完成后不会继续下一项",
                "finished_at": now,
                "finished_at_text": _format_ts(now),
            })
            canceled_waiting += 1
        self._save_queue()
        await self._broadcast_queue()
        running_kept = sum(1 for item in self._queue_items() if item.get("state") == "running")
    return {
        "ok": True,
        "message": (
            f"已中止后续队列，取消 {canceled_waiting} 个等待任务"
            if canceled_waiting
            else "队列已暂停，没有等待任务需要中止"
        ),
        "canceled_waiting": canceled_waiting,
        "running_kept": running_kept,
        **self.get_queue_snapshot(),
    }

async def force_abort_queue(self) -> dict[str, Any]:
    _bind_legacy()
    async with self._launch_lock:
        now = time.time()
        canceled_waiting = 0
        canceled_stale_running = 0
        canceled_launching = 0
        running_item_id = str(self._current_queue_item_id or "")
        launching_item_id = str(self._queue_launching_item_id or "")
        active_process = bool(self.process and self.process.returncode is None)

        self._queue_paused = True
        self._queue["paused"] = True
        for item in self._queue_items():
            state = item.get("state")
            item_id = str(item.get("id") or "")
            if state == "queued":
                item.update({
                    "state": "canceled",
                    "message": "已强制中止队列后续任务",
                    "finished_at": now,
                    "finished_at_text": _format_ts(now),
                })
                canceled_waiting += 1
                continue
            if state != "running":
                continue
            if active_process and item_id == running_item_id:
                continue
            item.update({
                "state": "canceled",
                "message": "已强制中止队列任务",
                "finished_at": now,
                "finished_at_text": _format_ts(now),
            })
            if item_id == launching_item_id:
                canceled_launching += 1
            else:
                canceled_stale_running += 1

        self._save_queue()
        await self._broadcast_queue()
        if active_process:
            await self.stop()
            await self._broadcast_queue()
        stopped_running = 1 if active_process else 0
        canceled = canceled_waiting + canceled_stale_running + canceled_launching + stopped_running
        snapshot = self.get_queue_snapshot()
    return {
        "ok": True,
        "message": f"已强制中止队列，处理 {canceled} 个任务",
        "canceled": canceled,
        "canceled_waiting": canceled_waiting,
        "stopped_running": stopped_running,
        "canceled_launching": canceled_launching,
        "canceled_stale_running": canceled_stale_running,
        **snapshot,
    }

async def clear_finished_queue_items(self) -> dict[str, Any]:
    _bind_legacy()
    return await self.clear_queue_items_by_state(QUEUE_CLEARABLE_STATES, label="已结束")

async def clear_completed_queue_items(self) -> dict[str, Any]:
    _bind_legacy()
    return await self.clear_queue_items_by_state({"done"}, label="已完成")

async def clear_canceled_queue_items(self) -> dict[str, Any]:
    _bind_legacy()
    return await self.clear_queue_items_by_state({"canceled"}, label="已取消")

async def clear_queue_items_by_state(self, states: set[str], *, label: str = "") -> dict[str, Any]:
    _bind_legacy()
    clear_states = {str(state or "").strip() for state in states}
    if not clear_states or clear_states - QUEUE_CLEARABLE_STATES:
        raise ValueError("只能清理已完成或已取消的队列记录")
    before = len(self._queue_items())
    removed_by_state = {state: 0 for state in sorted(clear_states)}
    remaining: list[dict[str, Any]] = []
    for item in self._queue_items():
        state = str(item.get("state") or "")
        if state in clear_states:
            removed_by_state[state] = removed_by_state.get(state, 0) + 1
            continue
        remaining.append(item)
    self._queue["items"] = remaining
    removed = before - len(self._queue["items"])
    if removed:
        self._save_queue()
        await self._broadcast_queue()
    clean_label = label or _queue_clearable_state_label(clear_states)
    return {
        "ok": True,
        "message": f"已清理 {removed} 条{clean_label}记录",
        "removed": removed,
        "removed_by_state": removed_by_state,
        **self.get_queue_snapshot(),
    }

async def set_queue_settings(
    self,
    *,
    paused: bool | None = None,
    failure_policy: str | None = None,
) -> dict[str, Any]:
    _bind_legacy()
    if paused is not None:
        self._queue_paused = bool(paused)
        self._queue["paused"] = self._queue_paused
    if failure_policy is not None:
        self._queue_failure_policy = _normalize_queue_failure_policy(failure_policy)
        self._queue["failure_policy"] = self._queue_failure_policy
    self._save_queue()
    await self._broadcast_queue()
    if not self._queue_paused:
        self._schedule_queue_dispatch()
    return self.get_queue_snapshot()

async def set_queue_paused(self, paused: bool) -> dict[str, Any]:
    _bind_legacy()
    return await self.set_queue_settings(paused=paused)

def _clone_queue_item_for_retry(self, item: dict[str, Any]) -> dict[str, Any]:
    _bind_legacy()
    runtime = _clone_frozen_runtime_config(
        str(item.get("runtime_config_file") or ""),
        source_config_file=str(item.get("source_config_file") or ""),
        reset_data_dirs=bool(item.get("requires_preprocess")),
    )
    now = time.time()
    retry_of = str(item.get("retry_of") or item.get("id") or "")
    attempt = int(item.get("attempt") or 1) + 1
    retry = {
        key: value for key, value in item.items()
        if key not in {
            "id", "state", "message", "created_at", "created_at_text",
            "started_at", "started_at_text", "finished_at", "finished_at_text",
            "history_task_ids", "runtime_config_file", "runtime_info",
        }
    }
    retry.update({
        "id": _new_queue_item_id(str(item.get("kind") or "retry"), str(item.get("methods_subdir") or "gui-methods"), str(item.get("variant") or "training")),
        "state": "queued",
        "runtime_config_file": runtime["runtime_config_file"],
        "source_config_file": str(item.get("source_config_file") or runtime.get("history_source_config_file") or ""),
        "retry_of": retry_of,
        "attempt": attempt,
        "history_task_ids": [],
        "message": f"第 {attempt} 次尝试，等待队列调度",
        "created_at": now,
        "created_at_text": _format_ts(now),
        "started_at": None,
        "started_at_text": "",
        "finished_at": None,
        "finished_at_text": "",
        "runtime_info": _runtime_meta(runtime),
    })
    return retry

def _repair_queue_on_startup(self) -> None:
    _bind_legacy()
    changed = False
    now = time.time()
    self._normalize_queue()
    for item in self._queue_items():
        if item.get("state") == "running":
            item.update({
                "state": "error",
                "message": "WebUI 重启时发现旧运行中队列项，已标记为异常",
                "finished_at": now,
                "finished_at_text": _format_ts(now),
            })
            changed = True
    if changed:
        self._queue_paused = True
        self._queue["paused"] = True
        self._save_queue()

def _normalize_queue(self) -> None:
    _bind_legacy()
    if not isinstance(self._queue, dict):
        self._queue = {}
    items = self._queue.get("items")
    if not isinstance(items, list):
        self._queue["items"] = []
    self._queue_paused = bool(self._queue.get("paused", self._queue_paused))
    self._queue["paused"] = self._queue_paused
    self._queue_failure_policy = _normalize_queue_failure_policy(
        self._queue.get("failure_policy", self._queue_failure_policy)
    )
    self._queue["failure_policy"] = self._queue_failure_policy
    for item in self._queue["items"]:
        if not isinstance(item, dict):
            continue
        item.setdefault("retry_of", "")
        item["attempt"] = max(1, _positive_int_or_none(item.get("attempt")) or 1)

def _queue_items(self) -> list[dict[str, Any]]:
    _bind_legacy()
    self._normalize_queue()
    items = self._queue["items"]
    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            out.append(item)
    self._queue["items"] = out
    return out

def _find_queue_item(self, item_id: str) -> dict[str, Any] | None:
    _bind_legacy()
    needle = str(item_id or "").strip()
    for item in self._queue_items():
        if str(item.get("id") or "") == needle:
            return item
    return None

def _update_queue_item(self, item_id: str, patch: dict[str, Any]) -> None:
    _bind_legacy()
    item = self._find_queue_item(item_id)
    if item is not None:
        item.update(patch)

def _attach_history_task_to_queue_item(self, item_id: str, task_id: str) -> None:
    _bind_legacy()
    item = self._find_queue_item(item_id)
    if item is None or not task_id:
        return
    history_ids = item.get("history_task_ids")
    if not isinstance(history_ids, list):
        history_ids = []
        item["history_task_ids"] = history_ids
    if task_id not in history_ids:
        history_ids.append(task_id)
    item["message"] = "正在运行"
    self._save_queue()

def _queue_history_meta(self, item_id: str) -> dict[str, Any]:
    _bind_legacy()
    item = self._find_queue_item(item_id)
    if item is None:
        return {}
    return {
        "from_queue": True,
        "queue_item_id": str(item.get("id") or item_id or ""),
        "queue_kind": str(item.get("kind") or ""),
        "queue_retry_of": str(item.get("retry_of") or ""),
        "queue_attempt": max(1, _positive_int_or_none(item.get("attempt")) or 1),
        "queue_created_at": item.get("created_at"),
        "queue_created_at_text": str(item.get("created_at_text") or ""),
    }

def _compact_queue(self) -> None:
    _bind_legacy()
    items = self._queue_items()
    if len(items) <= MAX_QUEUE_ITEMS:
        return
    protected = [item for item in items if item.get("state") in {"queued", "running"}]
    finished = [item for item in items if item.get("state") not in {"queued", "running"}]
    keep_finished = max(0, MAX_QUEUE_ITEMS - len(protected))
    self._queue["items"] = [*protected, *(finished[-keep_finished:] if keep_finished else [])]

def _save_queue(self) -> None:
    _bind_legacy()
    self._normalize_queue()
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    self._queue["paused"] = self._queue_paused
    self._queue["failure_policy"] = self._queue_failure_policy
    self._queue["updated_at"] = time.time()
    self._queue["updated_at_text"] = _format_ts(self._queue["updated_at"])
    _write_training_queue_state(self._queue)

async def _broadcast_queue(self) -> None:
    _bind_legacy()
    await self._broadcast({"type": "queue", **self.get_queue_snapshot()})

def _schedule_queue_dispatch(self) -> None:
    _bind_legacy()
    if self._queue_paused or self.status == "running" or self._queue_launching_item_id:
        return
    if self._queue_dispatch_task and not self._queue_dispatch_task.done():
        return
    if not any(item.get("state") == "queued" for item in self._queue_items()):
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    self._queue_dispatch_task = loop.create_task(self._dispatch_queue())

async def _dispatch_queue(self) -> None:
    _bind_legacy()
    failed = False
    queue_item_id = ""
    item: dict[str, Any] | None = None
    async with self._launch_lock:
        if self._queue_paused or self.status == "running" or self._queue_launching_item_id:
            return
        item = next((entry for entry in self._queue_items() if entry.get("state") == "queued"), None)
        if item is None:
            return
        queue_item_id = str(item.get("id") or "")
        if not queue_item_id:
            return
        self._queue_launching_item_id = queue_item_id
        now = time.time()
        try:
            item.update({
                "state": "running",
                "message": "正在启动",
                "started_at": now,
                "started_at_text": _format_ts(now),
                "finished_at": None,
                "finished_at_text": "",
            })
            self._save_queue()
        except Exception as e:
            failed = True
            now = time.time()
            self._pause_queue_after_failure()
            item.update({
                "state": "error",
                "message": f"队列任务启动失败: {e}",
                "finished_at": now,
                "finished_at_text": _format_ts(now),
            })
            self._current_queue_item_id = ""
            self.status = "idle"
            self._save_queue()
        finally:
            if failed and self._queue_launching_item_id == queue_item_id:
                self._queue_launching_item_id = ""
    await self._broadcast_queue()
    if not failed and item is not None:
        try:
            async with self._launch_lock:
                if item.get("state") != "running":
                    return
                await self._start_queue_item(item)
        except Exception as e:
            failed = True
            now = time.time()
            self._pause_queue_after_failure()
            item.update({
                "state": "error",
                "message": f"队列任务启动失败: {e}",
                "finished_at": now,
                "finished_at_text": _format_ts(now),
            })
            self._current_queue_item_id = ""
            self.status = "idle"
            self._save_queue()
        finally:
            if self._queue_launching_item_id == queue_item_id:
                self._queue_launching_item_id = ""
        if failed:
            await self._broadcast_queue()
    if failed:
        self._schedule_queue_dispatch()

def _pause_queue_after_failure(self) -> bool:
    _bind_legacy()
    if self._queue_failure_policy != "pause":
        return False
    self._queue_paused = True
    self._queue["paused"] = True
    return True

async def _start_queue_item(self, item: dict[str, Any]) -> None:
    _bind_legacy()
    variant = str(item.get("variant") or "")
    preset = str(item.get("preset") or "default")
    methods_subdir = str(item.get("methods_subdir") or "gui-methods")
    extra_args = list(item.get("extra_args") or [])
    queue_item_id = str(item.get("id") or "")
    _ensure_queue_resume_checkpoint_exists(item)
    if item.get("requires_preprocess"):
        runtime = self._queue_item_runtime(item)
        await self._start_preprocess_unlocked(
            variant,
            preset,
            methods_subdir,
            extra_args,
            True,
            gpu_whitelist=item.get("gpu_whitelist"),
            config_file=str(item.get("source_config_file") or item.get("runtime_config_file") or ""),
            continue_info=item.get("continue_info") if isinstance(item.get("continue_info"), dict) else None,
            runtime=runtime,
            queue_item_id=queue_item_id,
        )
        return

    runtime_config_file = _queue_resume_runtime_config_file(self, item)
    await self._start_unlocked(
        variant,
        preset,
        extra_args,
        methods_subdir,
        config_file=runtime_config_file,
        start_message=(
            f"从队列启动续训: {item.get('resume_info', {}).get('checkpoint_name')}"
            if item.get("kind") == "resume"
            else f"从队列启动训练: {methods_subdir}/{variant} / {preset}"
        ),
        command_label="队列训练命令",
        resume_info=item.get("resume_info") if isinstance(item.get("resume_info"), dict) else None,
        continue_info=item.get("continue_info") if isinstance(item.get("continue_info"), dict) else None,
        gpu_whitelist=item.get("gpu_whitelist"),
        source_config_file=str(item.get("source_config_file") or ""),
        use_runtime_dir=False,
        queue_item_id=queue_item_id,
    )

def _ensure_queue_resume_checkpoint_exists(item: dict[str, Any]) -> None:
    if item.get("kind") != "resume":
        return
    resume_info = item.get("resume_info") if isinstance(item.get("resume_info"), dict) else {}
    checkpoint = str(resume_info.get("checkpoint") or "").strip()
    path = _resolve_display_path(checkpoint)
    if path is None or not _path_exists(path / "train_state.json"):
        raise FileNotFoundError("续训检查点状态已不存在，请重新选择包含 train_state.json 的状态目录")
    reason = _resume_state_integrity_unavailable_reason(_resume_state_integrity(path))
    if reason:
        raise FileNotFoundError(reason)

def _queue_resume_runtime_config_file(self, item: dict[str, Any]) -> str:
    runtime_config_file = str(item.get("runtime_config_file") or "")
    if item.get("kind") != "resume":
        return runtime_config_file
    runtime = _runtime_from_config_file(
        runtime_config_file,
        source_config_file=str(item.get("source_config_file") or "") or None,
    )
    if runtime is None:
        runtime = _clone_frozen_runtime_config(
            runtime_config_file,
            source_config_file=str(item.get("source_config_file") or ""),
            reset_data_dirs=False,
        )
        item["runtime_config_file"] = str(runtime.get("runtime_config_file") or runtime_config_file)
        item["source_config_file"] = str(runtime.get("history_source_config_file") or item.get("source_config_file") or "")
        item["runtime_info"] = _runtime_meta(runtime)
        self._save_queue()
    return str(item.get("runtime_config_file") or runtime_config_file)

def _queue_item_runtime(self, item: dict[str, Any]) -> dict[str, Any]:
    _bind_legacy()
    runtime_config_file = str(item.get("runtime_config_file") or "")
    runtime = _runtime_from_config_file(
        runtime_config_file,
        source_config_file=str(item.get("source_config_file") or "") or None,
    )
    if runtime is None:
        raise FileNotFoundError(f"队列运行配置不可用: {runtime_config_file}")
    runtime["sample_config"] = _sample_config_from_cfg(
        _load_config_file_config(runtime["runtime_config_file"]),
        list(item.get("extra_args") or []),
    )
    return runtime
