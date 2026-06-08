"""Delegated training service methods.

This module is a mechanical extraction from ``web.services.training_service``.
The public ``TrainingService`` class keeps the same method names and delegates
here so HTTP routes and WebSocket payloads remain unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncio
    import os
    import sys
    import time
    from pathlib import Path
    from typing import Any

    import psutil

    from library.runtime.launch import (
        ACCELERATE_MIXED_PRECISION_ENV,
        accelerate_training_command_prefix,
        resolve_accelerate_mixed_precision,
    )
    from web.services.config_service import preflight_training_config

    from web.services.training_service import (
        ROOT,
        _apply_gpu_whitelist,
        _apply_runtime_env,
        _command_has_option,
        _command_option_value,
        _ensure_training_data_dirs,
        _format_ts,
        _load_config_file_config,
        _normalize_continue_lora_info,
        _normalize_gpu_whitelist,
        _prepare_web_runtime_config,
        _resolve_block_swap_profile_auto_arg,
        _resolve_block_swap_profile_auto_config,
        _resolve_display_path,
        _resolve_memory_probe_auto_arg,
        _resolve_memory_probe_auto_config,
        _resolve_peak_probe_auto_arg,
        _resolve_peak_probe_auto_config,
        _resolve_training_runtime_info,
        _runtime_from_config_file,
        _runtime_meta,
        _sample_config_from_cfg,
    )

from library.runtime.launch import (
    ACCELERATE_MIXED_PRECISION_ENV,
    resolve_accelerate_mixed_precision,
)


_LOCAL_IMPL_NAMES = {
    "_bind_legacy",
    "_accelerate_mixed_precision_for_training",
    "start",
    "_start_unlocked",
    "start_preprocess",
    "_start_preprocess_unlocked",
    "_launch_job",
    "stop",
    "_start_pending_training",
    "_ensure_launch_allowed",
    "_write_terminal",
}


def _bind_legacy() -> None:
    """Bind legacy module globals lazily after training_service has loaded."""
    from web.services import training_service as legacy

    for name, value in vars(legacy).items():
        if name.startswith("__") or name in _LOCAL_IMPL_NAMES:
            continue
        globals()[name] = value


def _accelerate_mixed_precision_for_training(
    config_file: str | None,
    extra_args: list[str] | None = None,
) -> str | None:
    _bind_legacy()
    value = _command_option_value(list(extra_args or []), "--mixed_precision")
    if value is None and config_file:
        raw = _load_config_file_config(config_file).get("mixed_precision")
        value = str(raw).strip() if raw is not None else None
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return resolve_accelerate_mixed_precision({
        ACCELERATE_MIXED_PRECISION_ENV: normalized,
    })


async def start(
    self,
    variant: str,
    preset: str,
    extra_args: list[str] | None = None,
    methods_subdir: str = "gui-methods",
    *,
    reset_logs: bool = True,
    config_file: str | None = None,
    start_message: str | None = None,
    command_label: str | None = None,
    resume_info: dict[str, Any] | None = None,
    continue_info: dict[str, Any] | None = None,
    gpu_whitelist: list[Any] | None = None,
    source_config_file: str | None = None,
    use_runtime_dir: bool = True,
    queue_item_id: str = "",
):
    _bind_legacy()
    async with self._launch_lock:
        await self._start_unlocked(
            variant,
            preset,
            extra_args,
            methods_subdir,
            reset_logs=reset_logs,
            config_file=config_file,
            start_message=start_message,
            command_label=command_label,
            resume_info=resume_info,
            continue_info=continue_info,
            gpu_whitelist=gpu_whitelist,
            source_config_file=source_config_file,
            use_runtime_dir=use_runtime_dir,
            queue_item_id=queue_item_id,
        )

async def _start_unlocked(
    self,
    variant: str,
    preset: str,
    extra_args: list[str] | None = None,
    methods_subdir: str = "gui-methods",
    *,
    reset_logs: bool = True,
    config_file: str | None = None,
    start_message: str | None = None,
    command_label: str | None = None,
    resume_info: dict[str, Any] | None = None,
    continue_info: dict[str, Any] | None = None,
    gpu_whitelist: list[Any] | None = None,
    source_config_file: str | None = None,
    use_runtime_dir: bool = True,
    queue_item_id: str = "",
):
    _bind_legacy()
    self._ensure_launch_allowed(queue_item_id)

    runtime = None
    if source_config_file and use_runtime_dir:
        runtime = _prepare_web_runtime_config(
            variant,
            preset,
            methods_subdir,
            source_config_file=source_config_file,
        )
        config_file = runtime["runtime_config_file"]
    elif source_config_file and not config_file:
        config_file = source_config_file

    venv_python = str(ROOT / ".venv" / "bin" / "python")
    if not Path(venv_python).exists():
        venv_python = sys.executable

    env = os.environ.copy()
    gpu_selection = _normalize_gpu_whitelist(gpu_whitelist)
    _apply_gpu_whitelist(env, gpu_selection)
    mixed_precision = _accelerate_mixed_precision_for_training(config_file, extra_args)
    if mixed_precision:
        env[ACCELERATE_MIXED_PRECISION_ENV] = mixed_precision
    cmd = [
        *accelerate_training_command_prefix(venv_python, ROOT / "train.py", env),
        "--method", variant,
        "--preset", preset,
        "--methods_subdir", methods_subdir,
    ]
    if config_file:
        cmd.extend(["--config_file", config_file])
    if extra_args:
        cmd.extend(extra_args)
    continue_payload = _normalize_continue_lora_info(
        continue_info,
        variant=variant,
        preset=preset,
        methods_subdir=methods_subdir,
        config_file=config_file,
    )
    if continue_payload:
        cmd.extend([
            "--network_weights",
            continue_payload["continue_from_weight_abs_path"],
            "--dim_from_weights",
        ])

    env["PYTHONUNBUFFERED"] = "1"
    env["PATH"] = str(ROOT / ".venv" / "bin") + ":" + env.get("PATH", "")
    active_runtime = runtime or _runtime_from_config_file(
        config_file,
        source_config_file=source_config_file,
    )
    _apply_runtime_env(env, active_runtime)
    runtime_info = _runtime_meta(active_runtime)

    if active_runtime:
        output_dir = active_runtime["output_dir"]
        sample_dir = active_runtime["sample_dir"]
        sample_config = _sample_config_from_cfg(
            _load_config_file_config(active_runtime["runtime_config_file"]),
            extra_args or [],
        )
        data_dirs = active_runtime["data_dirs"]
    else:
        output_dir, sample_dir, sample_config = _resolve_training_runtime_info(
            variant,
            preset,
            methods_subdir,
            extra_args or [],
            config_file=config_file,
        )
        data_dirs = _ensure_training_data_dirs(
            variant,
            preset,
            methods_subdir,
            config_file=config_file,
        )
    await self._launch_job(
        cmd,
        env,
        variant=variant,
        preset=preset,
        methods_subdir=methods_subdir,
        output_dir=output_dir,
        sample_dir=sample_dir,
        data_dirs=data_dirs,
        sample_config=sample_config,
        job="training",
        start_message=start_message or f"训练启动: {methods_subdir}/{variant} / {preset}",
        command_label=command_label or "训练命令",
        reset_logs=reset_logs,
        config_file=config_file,
        resume_info=resume_info,
        continue_info=continue_payload,
        gpu_whitelist=gpu_selection,
        runtime_info=runtime_info,
        queue_item_id=queue_item_id,
    )

async def start_preprocess(
    self,
    variant: str,
    preset: str,
    methods_subdir: str = "gui-methods",
    extra_args: list[str] | None = None,
    train_after: bool = False,
    gpu_whitelist: list[Any] | None = None,
    config_file: str | None = None,
    continue_info: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
    queue_item_id: str = "",
):
    _bind_legacy()
    async with self._launch_lock:
        await self._start_preprocess_unlocked(
            variant,
            preset,
            methods_subdir,
            extra_args,
            train_after,
            gpu_whitelist,
            config_file,
            continue_info,
            runtime,
            queue_item_id,
        )

async def _start_preprocess_unlocked(
    self,
    variant: str,
    preset: str,
    methods_subdir: str = "gui-methods",
    extra_args: list[str] | None = None,
    train_after: bool = False,
    gpu_whitelist: list[Any] | None = None,
    config_file: str | None = None,
    continue_info: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
    queue_item_id: str = "",
):
    _bind_legacy()
    self._ensure_launch_allowed(queue_item_id)

    continue_payload = _normalize_continue_lora_info(
        continue_info,
        variant=variant,
        preset=preset,
        methods_subdir=methods_subdir,
        config_file=config_file,
    )
    venv_python = str(ROOT / ".venv" / "bin" / "python")
    if not Path(venv_python).exists():
        venv_python = sys.executable

    runtime = runtime or _prepare_web_runtime_config(
        variant,
        preset,
        methods_subdir,
        source_config_file=config_file,
    )

    cmd = [venv_python, "tasks.py", "preprocess"]
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    gpu_selection = _normalize_gpu_whitelist(gpu_whitelist)
    _apply_gpu_whitelist(env, gpu_selection)
    env["PYTHONUNBUFFERED"] = "1"
    env["PATH"] = str(ROOT / ".venv" / "bin") + ":" + env.get("PATH", "")
    env["METHOD"] = variant
    env["METHODS_SUBDIR"] = methods_subdir
    _apply_runtime_env(env, runtime)
    env["PRESET"] = preset

    output_dir = runtime["output_dir"]
    sample_dir = runtime["sample_dir"]
    sample_config = runtime["sample_config"]
    data_dirs = runtime["data_dirs"]
    runtime_info = _runtime_meta(runtime)
    self._pending_train_after_preprocess = {
        "variant": variant,
        "preset": preset,
        "methods_subdir": methods_subdir,
        "extra_args": list(extra_args or []),
        "config_file": runtime["runtime_config_file"],
        "source_config_file": runtime.get("history_source_config_file") or config_file,
        "gpu_whitelist": gpu_selection,
        "continue_info": continue_payload,
        "queue_item_id": queue_item_id,
    } if train_after else None
    await self._launch_job(
        cmd,
        env,
        variant=variant,
        preset=preset,
        methods_subdir=methods_subdir,
        output_dir=output_dir,
        sample_dir=sample_dir,
        data_dirs=data_dirs,
        sample_config=sample_config,
        job="preprocess",
        start_message=f"预处理启动: {methods_subdir}/{variant} / {preset}",
        command_label="预处理命令",
        gpu_whitelist=gpu_selection,
        config_file=runtime["runtime_config_file"],
        runtime_info=runtime_info,
        queue_item_id=queue_item_id,
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
    _bind_legacy()
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
            cwd=str(ROOT),
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
    _bind_legacy()
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

async def _start_pending_training(self, pending: dict[str, Any]) -> None:
    _bind_legacy()
    self._remember_log("status", "预处理完成，自动开始训练")
    await self._broadcast({
        "type": "log",
        **self._remember_log("log", "[状态] 预处理完成，自动开始训练"),
    })
    try:
        preflight = preflight_training_config(
            pending["variant"],
            pending["preset"],
            pending["methods_subdir"],
            config_file=pending.get("config_file"),
        )
        if not preflight.get("ok", False):
            errors = preflight.get("summary", {}).get("errors", 0)
            raise RuntimeError(f"预处理后仍有 {errors} 个预检测错误")
        await self.start(
            pending["variant"],
            pending["preset"],
            pending.get("extra_args") or [],
            pending["methods_subdir"],
            reset_logs=False,
            config_file=pending.get("config_file"),
            source_config_file=pending.get("source_config_file"),
            gpu_whitelist=pending.get("gpu_whitelist"),
            continue_info=pending.get("continue_info"),
            use_runtime_dir=False,
            queue_item_id=pending.get("queue_item_id") or "",
        )
    except Exception as e:
        msg = f"自动开始训练失败: {e}"
        queue_item_id = str(pending.get("queue_item_id") or "")
        if queue_item_id:
            self._pause_queue_after_failure()
            self._update_queue_item(queue_item_id, {
                "state": "error",
                "message": msg,
                "finished_at": time.time(),
                "finished_at_text": _format_ts(time.time()),
            })
            self._save_queue()
            await self._broadcast_queue()
        self._remember_log("status", msg)
        await self._broadcast({
            "type": "status",
            "state": "error",
            "job": "training",
            "message": msg,
            "output_dir": self.current_output_dir,
            "sample_dir": self.current_sample_dir,
            "sample_config": self.current_sample_config,
            **self.current_runtime_info,
            "task_id": self.current_task_id,
            "queue_item_id": queue_item_id,
        })
        self._schedule_queue_dispatch()

def _ensure_launch_allowed(self, queue_item_id: str = "") -> None:
    _bind_legacy()
    launching = self._queue_launching_item_id
    same_queue_item = launching and str(queue_item_id or "") == launching
    if self.status == "running" or (launching and not same_queue_item):
        raise RuntimeError("已有任务在运行中")

def _write_terminal(self, text: str) -> None:
    _bind_legacy()
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except Exception:
        pass
