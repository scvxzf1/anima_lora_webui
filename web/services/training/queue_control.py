"""Queue control helpers delegated from ``TrainingService``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from web.services.training.common import _format_ts
from web.services.training.service_state import (
    _normalize_queue_failure_policy,
    _queue_clearable_state_label,
)

import time
from typing import Any

from web.services.training.constants import QUEUE_CLEARABLE_STATES, QUEUE_TERMINAL_STATES
from web.services.training.runtime_state import (
    _delete_queue_item_runtime_dir,
    _queue_item_runtime_dir_label,
)


async def move_queue_item(self, item_id: str, direction: str) -> dict[str, Any]:
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
    return await self.clear_queue_items_by_state(QUEUE_CLEARABLE_STATES, label="已结束")


async def clear_completed_queue_items(self) -> dict[str, Any]:
    return await self.clear_queue_items_by_state({"done"}, label="已完成")


async def clear_canceled_queue_items(self) -> dict[str, Any]:
    return await self.clear_queue_items_by_state({"canceled"}, label="已取消")


async def clear_queue_items_by_state(self, states: set[str], *, label: str = "") -> dict[str, Any]:
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
    return await self.set_queue_settings(paused=paused)