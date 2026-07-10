"""Queue dispatch helpers delegated from ``TrainingService``."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from web.services.training.common import _format_ts
from web.services.training.resume import (
    _resume_state_integrity,
    _resume_state_integrity_unavailable_reason,
)
from web.services.training.runtime_common import (
    _load_config_file_config,
    _sample_config_from_cfg,
)
from web.services.training.runtime_paths import _path_exists, _resolve_display_path
from web.services.training.runtime_resume import _clone_frozen_runtime_config
from web.services.training.runtime_state import (
    _runtime_from_config_file,
    _runtime_meta,
)

def _schedule_queue_dispatch(self) -> None:
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
    failed = False
    queue_item_id = ""
    item: dict[str, Any] | None = None
    async with self._launch_lock:
        if self._queue_paused or self.status == "running" or self._queue_launching_item_id:
            return
        now = time.time()
        item = next(
            (
                entry for entry in self._queue_items()
                if entry.get("state") == "queued"
                and (
                    entry.get("next_run_at") in (None, "")
                    or float(entry.get("next_run_at") or 0) <= now
                )
            ),
            None,
        )
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
    if self._queue_failure_policy != "pause":
        return False
    self._queue_paused = True
    self._queue["paused"] = True
    return True


async def _start_queue_item(self, item: dict[str, Any]) -> None:
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