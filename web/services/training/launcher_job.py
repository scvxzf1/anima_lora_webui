"""Training job lifecycle helpers for WebUI launcher."""

from __future__ import annotations

import asyncio
import sys
import time
from typing import Any

import psutil

from web.services.training.common import _format_ts
from web.services.training.launch_support import (
    _command_has_option,
    _command_option_value,
    _resolve_block_swap_profile_auto_arg,
    _resolve_block_swap_profile_auto_config,
    _resolve_memory_probe_auto_arg,
    _resolve_memory_probe_auto_config,
    _resolve_peak_probe_auto_arg,
    _resolve_peak_probe_auto_config,
)
from web.services.training.launcher_runtime import (
    _resolve_display_path,
    _root,
    _runtime_meta,
)


async def _launch_job(
    self,
    cmd: list[str],
    env: dict[str, str],
    *,
    variant: str,
    preset: str,
    methods_subdir: str,
    output_dir: str,
    sample_dir: str,
    data_dirs: dict[str, str],
    sample_config: dict[str, Any],
    job: str,
    start_message: str,
    command_label: str,
    reset_logs: bool = True,
    config_file: str | None = None,
    resume_info: dict[str, Any] | None = None,
    continue_info: dict[str, Any] | None = None,
    gpu_whitelist: list[int] | None = None,
    runtime_info: dict[str, str] | None = None,
    queue_item_id: str = "",
):
    self.status = "running"
    self._current_queue_item_id = str(queue_item_id or "")
    self.current_job = job
    self.current_variant = variant
    self.current_preset = preset
    self.current_methods_subdir = methods_subdir
    self.current_output_dir = output_dir
    self.current_sample_dir = sample_dir
    self.current_sample_config = sample_config
    self.current_runtime_info = _runtime_meta(runtime_info)
    self.current_gpu_whitelist = list(gpu_whitelist or [])
    self._anchor = None
    self._reset_progress_rate_state()
    self._reset_metric_runtime_state()
    self._latest_progress = None
    self._latest_system_stats = None
    self._progress_jsonl_path = None
    self._progress_jsonl_offset = 0
    self._progress_jsonl_seen = set()
    self._progress_jsonl_lock = asyncio.Lock()
    self._progress_total_steps = None
    self._detected_error_hint = ""
    self._stop_requested = False
    self.current_task_id = ""
    self.current_task_dir = None
    if job != "preprocess":
        self._pending_train_after_preprocess = None
    self._last_output_at = time.time()
    self._last_log_line = ""
    if reset_logs:
        self._log_records.clear()
        self._next_log_id = 1

    task_dir = self._reserve_history_task_dir(job, methods_subdir, variant)
    if job == "training" and not _command_has_option(cmd, "--progress_jsonl"):
        cmd = [*cmd, "--progress_jsonl", str(task_dir / "progress.jsonl")]
    if job == "training":
        block_swap_profile_path = task_dir / "block_swap_profile.jsonl"
        config_wants_block_swap_profile = _resolve_block_swap_profile_auto_config(
            config_file, block_swap_profile_path
        )
        cmd = _resolve_block_swap_profile_auto_arg(cmd, block_swap_profile_path)
        if (
            config_wants_block_swap_profile
            and not _command_has_option(cmd, "--block_swap_profile_jsonl")
        ):
            cmd = [*cmd, "--block_swap_profile_jsonl", str(block_swap_profile_path)]
        memory_probe_path = task_dir / "memory_probe.jsonl"
        config_wants_memory_probe = _resolve_memory_probe_auto_config(
            config_file, memory_probe_path
        )
        cmd = _resolve_memory_probe_auto_arg(cmd, memory_probe_path)
        if (
            config_wants_memory_probe
            and not _command_has_option(cmd, "--memory_probe_jsonl")
        ):
            cmd = [*cmd, "--memory_probe_jsonl", str(memory_probe_path)]
        peak_probe_path = task_dir / "peak_probe.jsonl"
        config_wants_peak_probe = _resolve_peak_probe_auto_config(
            config_file, peak_probe_path
        )
        cmd = _resolve_peak_probe_auto_arg(cmd, peak_probe_path)
        if (
            config_wants_peak_probe
            and not _command_has_option(cmd, "--peak_probe_jsonl")
        ):
            cmd = [*cmd, "--peak_probe_jsonl", str(peak_probe_path)]
    if job == "training":
        progress_jsonl = _command_option_value(cmd, "--progress_jsonl")
        self._progress_jsonl_path = _resolve_display_path(progress_jsonl or str(task_dir / "progress.jsonl"))
    self.current_command = cmd
    self._start_history_task(
        job=job,
        variant=variant,
        preset=preset,
        methods_subdir=methods_subdir,
        output_dir=output_dir,
        sample_dir=sample_dir,
        data_dirs=data_dirs,
        sample_config=sample_config,
        command=cmd,
        config_file=config_file,
        resume_info=resume_info,
        continue_info=continue_info,
        gpu_whitelist=gpu_whitelist,
        runtime_info=self.current_runtime_info,
        queue_info=self._queue_history_meta(self._current_queue_item_id),
    )
    try:
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=str(_root()),
            start_new_session=True,
        )
    except Exception as e:
        self.status = "idle"
        self._finish_history_task(state="error", message=f"任务启动失败: {e}", returncode=-1)
        raise
    self._remember_log("status", f"{command_label}: {' '.join(cmd)}")
    if gpu_whitelist:
        self._remember_log("status", f"GPU 白名单: {','.join(str(item) for item in gpu_whitelist)}")

    await self._broadcast({
        "type": "status",
        "state": "running",
        "job": job,
        "message": start_message,
        "variant": variant,
        "preset": preset,
        "methods_subdir": methods_subdir,
        "output_dir": self.current_output_dir,
        "sample_dir": self.current_sample_dir,
        "sample_config": self.current_sample_config,
        **self.current_runtime_info,
        "task_id": self.current_task_id,
        "queue_item_id": self._current_queue_item_id,
    })
    if self._current_queue_item_id:
        self._attach_history_task_to_queue_item(self._current_queue_item_id, self.current_task_id)
        await self._broadcast_queue()
    asyncio.create_task(self._read_output())
    asyncio.create_task(self._monitor_system())
    if self._progress_jsonl_path:
        asyncio.create_task(self._tail_progress_jsonl())

async def stop(self):
    if not self.process or self.process.returncode is not None:
        self.status = "idle"
        return
    queue_item_id = self._current_queue_item_id
    if queue_item_id:
        self._queue_paused = True
        self._queue["paused"] = True
        self._update_queue_item(queue_item_id, {
            "state": "canceled",
            "message": "用户停止了队列任务，队列已自动暂停",
            "finished_at": time.time(),
            "finished_at_text": _format_ts(time.time()),
        })
        self._save_queue()
    try:
        pid = self.process.pid
        parent = psutil.Process(pid)
        family = [parent] + parent.children(recursive=True)
        for p in family:
            try:
                p.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        _, alive = psutil.wait_procs(family, timeout=3.0)
        for p in alive:
            try:
                p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except psutil.NoSuchProcess:
        pass
    job = self.current_job
    self._stop_requested = True
    self._pending_train_after_preprocess = None
    self.status = "idle"
    message = "预处理已停止" if job == "preprocess" else "训练已停止"
    await self._broadcast({
        "type": "status",
        "state": "idle",
        "job": job,
        "message": message,
        "output_dir": self.current_output_dir,
        "sample_dir": self.current_sample_dir,
        "sample_config": self.current_sample_config,
        **self.current_runtime_info,
        "task_id": self.current_task_id,
        "queue_item_id": queue_item_id,
    })
    if queue_item_id:
        await self._broadcast_queue()

def _ensure_launch_allowed(self, queue_item_id: str = "") -> None:
    launching = self._queue_launching_item_id
    same_queue_item = launching and str(queue_item_id or "") == launching
    if self.status == "running" or (launching and not same_queue_item):
        raise RuntimeError("已有任务在运行中")

def _write_terminal(self, text: str) -> None:
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except Exception:
        pass
