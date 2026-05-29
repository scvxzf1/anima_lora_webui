"""Training subprocess management and output parsing."""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime
import json
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

from library.env import load_dotenv
from library.preprocess.captions import (
    CAPTION_SOURCE_CAPTIONS_JSON,
    CAPTIONS_JSON_FILE,
    normalize_caption_source_mode,
)
from library.runtime.launch import accelerate_training_command_prefix
from web.services.config_service import (
    NL_TAG_MIX_CLASSIFICATION_METHOD,
    _build_dataset_config_doc,
    _classify_nl_tag_caption_text,
    _dataset_rows_for_estimate,
    _nl_tag_mix_caption_source,
    _nl_tag_mix_image_files,
    _normalize_nl_tag_mix,
    _normalize_trigger_clone,
    apply_auto_data_dirs,
    load_merged_config,
    preflight_training_config,
)
from web.services.settings_service import display_path as _display_settings_path
from web.services.settings_service import resolve_output_root

ROOT = Path(__file__).resolve().parents[2]
HISTORY_DIR = ROOT / "configs" / "web-training-history"
HISTORY_COLLECTIONS_FILE = HISTORY_DIR / "collections.json"
QUEUE_DIR = ROOT / "configs" / "web-training-queue"
QUEUE_FILE = QUEUE_DIR / "queue.json"
RUN_META_FILE = "run.meta.json"
OUTPUT_READ_SIZE = 4096
MAX_LOG_RECORDS = 3000
MAX_HISTORY_ITEMS = 100
MAX_RESUME_CHECKPOINTS = 100
MAX_TIMELINE_LOG_RECORDS = 20000
MAX_TIMELINE_METRIC_RECORDS = 20000
MAX_HISTORY_DETAIL_LOG_RECORDS = 5000
MAX_HISTORY_DETAIL_SYSTEM_RECORDS = 1000
MAX_QUEUE_ITEMS = 200
QUEUE_FAILURE_POLICIES = {"pause", "continue"}
QUEUE_TERMINAL_STATES = {"done", "error", "canceled"}
# “清理已结束”保留 error，方便用户确认后重试或手动删除异常记录。
QUEUE_CLEARABLE_STATES = {"done", "canceled"}
DATASET_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})
DATASET_CAPTION_EXTS = (".txt", ".json", ".caption")
CONTINUE_LORA_KINDS = {"LoRA", "LoKr"}
CONTINUE_LORA_ACCEPTED_LORA_SPECS = {"", "lora", "standard", "ortho", "ortholora", "tlora", "t_lora"}
CONTINUE_LORA_UNSUPPORTED_SPEC_TOKENS = (
    "hydra",
    "chimera",
    "stacked",
    "fera",
    "moe",
    "reft",
    "postfix",
    "ip_adapter",
    "easycontrol",
    "soft_tokens",
)
CONTINUE_LORA_UNSUPPORTED_KEY_FRAGMENTS = (
    ".lora_ups.",
    ".lora_downs.",
    ".lora_up_weight",
    ".lora_down_weight",
    ".lora_up_c_weight",
    ".lora_up_f_weight",
    ".lora_down_c.",
    ".lora_down_f.",
    ".router.",
    "freq_router.",
    "content_router.",
)
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

TQDM_RE = re.compile(
    r"^(?P<label>.*?):?\s*(?P<pct>\d+)%\|[^|]*\|\s*(?P<cur>\d+)/(?P<tot>\d+)"
    r"(?:[^\[]*\[[^\]]*?(?P<rate>[\d.]+)(?P<unit>it/s|s/it)[^\]]*\])?"
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
        self.current_task_id: str = ""
        self.current_task_dir: Path | None = None
        self.current_command: list[str] = []
        self._stop_requested = False
        self._pending_train_after_preprocess: dict[str, Any] | None = None
        self._ws_clients: set[web.WebSocketResponse] = set()
        self._anchor: tuple[float, int] | None = None
        self._metrics_history: list[dict[str, Any]] = []
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

    async def start_queue_on_startup(self) -> None:
        self._schedule_queue_dispatch()

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

    async def resume_from_history_task(
        self,
        task_id: str,
        checkpoint: str | None = None,
        *,
        gpu_whitelist: list[Any] | None = None,
    ) -> dict[str, Any]:
        task, selected, snapshot_path, resume_info = self._build_resume_payload(task_id, checkpoint)
        config_file = _display_project_path(str(snapshot_path))

        await self.start(
            str(task.get("variant") or ""),
            str(task.get("preset") or "default"),
            ["--resume", selected["path"], "--skip_until_initial_step"],
            str(task.get("methods_subdir") or "gui-methods"),
            config_file=config_file,
            start_message=f"从检查点继续训练: {selected['name']}",
            command_label="续训命令",
            resume_info=resume_info,
            gpu_whitelist=gpu_whitelist,
            use_runtime_dir=False,
        )

        return {
            "ok": True,
            "message": "已从检查点继续训练",
            "task_id": self.current_task_id,
            "checkpoint": selected,
        }

    def _build_resume_payload(
        self,
        task_id: str,
        checkpoint: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
        payload = _load_history_task(task_id)
        task = payload.get("task") if isinstance(payload, dict) else {}
        if not isinstance(task, dict):
            raise ValueError("任务不存在")
        if task.get("job") != "training":
            raise ValueError("只能从训练任务继续训练")

        checkpoints = _list_resume_checkpoints(task)
        if not checkpoints:
            raise ValueError("这个训练任务没有可续训的检查点")

        selected = _select_resume_checkpoint(checkpoints, checkpoint)
        if selected is None:
            raise ValueError("未找到指定的检查点")

        snapshot_path = _history_snapshot_path(task_id)
        if snapshot_path is None:
            raise ValueError("历史任务缺少配置快照，无法安全续训")
        resume_info = {
            "source_task_id": task_id,
            "source_task_name": str(task.get("name") or ""),
            "history_group_key": str(task.get("history_group_key") or ""),
            "history_group_label": str(task.get("history_group_label") or ""),
            "history_source_config_file": str(task.get("history_source_config_file") or ""),
            "checkpoint": selected["path"],
            "checkpoint_name": selected["name"],
            "checkpoint_kind": selected["kind"],
            "checkpoint_epoch": selected.get("epoch"),
            "checkpoint_step": selected.get("step"),
        }
        return task, selected, snapshot_path, resume_info

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
        self._anchor = None
        self._reset_metric_runtime_state()
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

    def subscribe(self, ws: web.WebSocketResponse):
        self._ws_clients.add(ws)

    def unsubscribe(self, ws: web.WebSocketResponse):
        self._ws_clients.discard(ws)

    def get_metrics_history(self) -> list[dict]:
        return self._metrics_history[-500:]

    def get_log_records(self, after: int = 0, limit: int = 1000) -> list[dict[str, Any]]:
        limit = max(1, min(limit, MAX_LOG_RECORDS))
        records = [record for record in self._log_records if record["id"] > after]
        return records[-limit:]

    async def list_gpus(self) -> list[dict[str, Any]]:
        return await _list_available_gpus()

    def get_queue_snapshot(self) -> dict[str, Any]:
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
    ) -> dict[str, Any]:
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
        self._queue_items().append(item)
        self._compact_queue()
        self._save_queue()
        await self._broadcast_queue()
        self._schedule_queue_dispatch()
        return {"ok": True, "message": "已加入训练队列", "item": dict(item), **self.get_queue_snapshot()}

    async def enqueue_resume_from_history_task(
        self,
        task_id: str,
        checkpoint: str | None = None,
        *,
        gpu_whitelist: list[Any] | None = None,
    ) -> dict[str, Any]:
        task, selected, snapshot_path, resume_info = self._build_resume_payload(task_id, checkpoint)
        now = time.time()
        item = {
            "id": _new_queue_item_id("resume", str(task.get("methods_subdir") or "gui-methods"), str(task.get("variant") or "training")),
            "state": "queued",
            "kind": "resume",
            "requires_preprocess": False,
            "variant": str(task.get("variant") or ""),
            "preset": str(task.get("preset") or "default"),
            "methods_subdir": str(task.get("methods_subdir") or "gui-methods"),
            "runtime_config_file": _display_project_path(str(snapshot_path)),
            "source_config_file": str(task.get("history_source_config_file") or ""),
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
            "runtime_info": {},
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
            message = "已删除队列记录和运行缓存" if delete_runtime else "已删除队列记录"
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

    async def clear_finished_queue_items(self) -> dict[str, Any]:
        before = len(self._queue_items())
        self._queue["items"] = [
            item for item in self._queue_items()
            if item.get("state") not in QUEUE_CLEARABLE_STATES
        ]
        removed = before - len(self._queue["items"])
        if removed:
            self._save_queue()
            await self._broadcast_queue()
        return {"ok": True, "message": f"已清理 {removed} 条已结束记录", "removed": removed, **self.get_queue_snapshot()}

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

    def _clone_queue_item_for_retry(self, item: dict[str, Any]) -> dict[str, Any]:
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

    def list_history_tasks(self, *, include_archived: bool = False, limit: int | None = None) -> list[dict[str, Any]]:
        return _list_history_tasks(include_archived=include_archived, limit=limit)

    def get_history_task(self, task_id: str) -> dict[str, Any]:
        return _load_history_task(task_id)

    def get_config_group_timeline(
        self,
        methods_subdir: str,
        variant: str,
        preset: str,
        *,
        group_key: str = "",
        include_archived: bool = False,
        task_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        return _build_config_group_timeline(
            methods_subdir,
            variant,
            preset,
            group_key=group_key,
            include_archived=include_archived,
            task_ids=task_ids,
        )

    def get_history_collection_settings(self) -> dict[str, Any]:
        return _load_history_collection_settings()

    def save_history_collection_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        settings = _normalize_history_collection_settings(payload)
        settings["updated_at"] = datetime.now().timestamp()
        settings["updated_at_text"] = _format_ts(settings["updated_at"])
        _write_json_atomic(HISTORY_COLLECTIONS_FILE, settings)
        return {"ok": True, **settings}

    def get_resume_options(self, task_id: str) -> dict[str, Any]:
        payload = _load_history_task(task_id)
        task = payload.get("task") if isinstance(payload, dict) else {}
        if not isinstance(task, dict):
            raise FileNotFoundError("任务不存在")
        if task.get("job") != "training":
            raise ValueError("只能从训练任务读取续训检查点")
        checkpoints = _list_resume_checkpoints(task)
        diagnostic = _resume_checkpoint_diagnostic(task, checkpoints)
        default_checkpoint = checkpoints[0]["path"] if checkpoints else ""
        message = "选择一个保存了训练状态的目录继续训练。普通权重文件不能恢复优化器和步数。"
        if not checkpoints:
            message = diagnostic.get("reason") or "这个任务没有找到可续训的状态目录。只有保存了 train_state.json 的目录才能继续训练。"
        return {
            "ok": True,
            "task": {
                "id": task.get("id", task_id),
                "name": task.get("name", ""),
                "group": task.get("group", ""),
                "variant": task.get("variant", ""),
                "preset": task.get("preset", ""),
                "methods_subdir": task.get("methods_subdir", ""),
                "output_dir": task.get("output_dir", ""),
                "sample_dir": task.get("sample_dir", ""),
            },
            "checkpoints": checkpoints,
            "default_checkpoint": default_checkpoint,
            "message": message,
            "diagnostic": diagnostic,
        }

    def update_history_task(self, task_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        return _update_history_task(task_id, patch)

    def batch_update_history_tasks(self, payload: dict[str, Any]) -> dict[str, Any]:
        action = str(payload.get("action") or "").strip().lower()
        task_ids = _normalize_history_task_ids(payload.get("task_ids"))
        if not task_ids:
            raise ValueError("请先选择历史任务")
        if action in {"archive", "unarchive"}:
            return _batch_archive_history_tasks(task_ids, archived=(action == "archive"))
        if action == "set_group":
            return _batch_set_history_group(task_ids, payload.get("group"))
        if action == "delete":
            return self._batch_delete_history_tasks(payload, task_ids)
        raise ValueError("不支持的批量操作")

    def delete_history_task(self, task_id: str) -> dict[str, Any]:
        delete_task_ids = _history_task_ids_for_delete(task_id)
        if self.status == "running" and self.current_task_id in delete_task_ids:
            raise RuntimeError("当前运行中的任务不能删除")
        return _delete_history_tasks(delete_task_ids)

    def _batch_delete_history_tasks(self, payload: dict[str, Any], task_ids: list[str]) -> dict[str, Any]:
        delete_runtime_dirs = bool(payload.get("delete_runtime_dirs"))
        plan = self._plan_history_delete(task_ids, delete_runtime_dirs=delete_runtime_dirs)
        if payload.get("dry_run", False):
            return {"ok": True, "dry_run": True, **plan}
        if plan["blocked"]:
            raise RuntimeError("存在不能删除的任务或运行目录，请先处理阻止项")
        if delete_runtime_dirs and str(payload.get("confirm_text") or "").strip() != "彻底删除":
            raise ValueError("彻底删除需要输入确认文本：彻底删除")
        result = _delete_history_tasks([item["id"] for item in plan["tasks"]])
        deleted_runtime_dirs: list[str] = []
        runtime_cleanup_errors: dict[str, str] = {}
        if delete_runtime_dirs:
            for item in plan["runtime_dirs"]:
                path = _resolve_display_path(str(item.get("path") or ""))
                if path is None or not _path_exists(path):
                    continue
                try:
                    shutil.rmtree(path)
                    deleted_runtime_dirs.append(str(item.get("path") or ""))
                except OSError as exc:
                    runtime_cleanup_errors[str(item.get("path") or "")] = str(exc)
        result.update({
            "dry_run": False,
            "deleted_runtime_dirs": deleted_runtime_dirs,
            "runtime_cleanup_errors": runtime_cleanup_errors,
            "preview": plan,
        })
        if delete_runtime_dirs:
            result["message"] = f"已彻底删除 {len(result.get('deleted_task_ids') or [])} 个历史记录和 {len(deleted_runtime_dirs)} 个运行目录"
        return result

    def _plan_history_delete(self, task_ids: list[str], *, delete_runtime_dirs: bool) -> dict[str, Any]:
        tasks_by_id = {
            str(task.get("id") or ""): task
            for task in _list_history_tasks(include_archived=True, limit=0)
        }
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        blocked: list[dict[str, str]] = []
        for task_id in task_ids:
            task = tasks_by_id.get(task_id)
            if not task:
                blocked.append({"id": task_id, "reason": "任务不存在"})
                continue
            for linked_id in _history_task_ids_for_delete(task_id):
                if linked_id in seen:
                    continue
                linked = tasks_by_id.get(linked_id)
                if linked:
                    selected.append(linked)
                    seen.add(linked_id)
        if self.status == "running" and self.current_task_id in seen:
            blocked.append({"id": self.current_task_id, "reason": "当前运行中的任务不能删除"})

        runtime_dirs: list[dict[str, str]] = []
        if delete_runtime_dirs:
            run_keys = {
                _history_delete_run_key(task)
                for task in selected
                if _history_delete_run_key(task)
            }
            for candidate in tasks_by_id.values():
                candidate_id = str(candidate.get("id") or "")
                if not candidate_id or candidate_id in seen:
                    continue
                if _history_delete_run_key(candidate) not in run_keys:
                    continue
                selected.append(candidate)
                seen.add(candidate_id)
            if self.status == "running" and self.current_task_id in seen and not any(
                item.get("id") == self.current_task_id for item in blocked
            ):
                blocked.append({"id": self.current_task_id, "reason": "当前运行中的任务不能删除"})
            runtime_dirs, runtime_blocked = _history_runtime_delete_dirs_for_tasks(selected)
            blocked.extend(runtime_blocked)
            queue_refs = _queue_runtime_delete_blockers(self._queue_items(), runtime_dirs)
            blocked.extend(queue_refs)

        return {
            "tasks": [_history_delete_task_preview(task) for task in selected],
            "runtime_dirs": runtime_dirs,
            "blocked": blocked,
            "delete_runtime_dirs": delete_runtime_dirs,
            "task_count": len(selected),
            "runtime_dir_count": len(runtime_dirs),
        }

    def get_status_snapshot(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "variant": self.current_variant,
            "preset": self.current_preset,
            "methods_subdir": self.current_methods_subdir,
            "job": self.current_job,
            "output_dir": self.current_output_dir,
            "sample_dir": self.current_sample_dir,
            "sample_config": self.current_sample_config,
            **self.current_runtime_info,
            "task_id": self.current_task_id,
            "last_output_at": self._last_output_at,
            "last_log_line": self._last_log_line,
            "last_log_id": self._log_records[-1]["id"] if self._log_records else 0,
            "error_hint": self._detected_error_hint,
            "queue_paused": self._queue_paused,
            "queue_count": sum(1 for item in self._queue_items() if item.get("state") == "queued"),
            "queue_item_id": self._current_queue_item_id,
        }

    async def _read_output(self):
        assert self.process and self.process.stdout
        try:
            buffer = ""
            while True:
                raw = await self.process.stdout.read(OUTPUT_READ_SIZE)
                if not raw:
                    break
                decoded = raw.decode("utf-8", errors="replace")
                self._write_terminal(decoded)
                buffer += decoded
                buffer = await self._drain_output_buffer(buffer)
            if buffer.strip():
                await self._handle_output_record(buffer)
        except Exception:
            pass

        rc = await self.process.wait()
        job = self.current_job
        stop_requested = self._stop_requested
        pending_train = self._pending_train_after_preprocess
        queue_item_id = self._current_queue_item_id
        await self._ingest_progress_jsonl(final=True)
        self.status = "idle"
        self.current_job = ""
        self._stop_requested = False
        self._pending_train_after_preprocess = None
        self._current_queue_item_id = ""
        state = "idle" if rc == 0 or stop_requested else "error"
        if stop_requested and job == "preprocess":
            msg = "预处理已停止"
        elif stop_requested:
            msg = "训练已停止"
        elif job == "preprocess":
            msg = "预处理完成" if rc == 0 else f"预处理异常退出 (code={rc})"
        else:
            msg = "训练完成" if rc == 0 else f"训练异常退出 (code={rc})"
        if state == "error":
            msg = _message_with_error_hint(msg, self._detected_error_hint)
        self._remember_log("status", msg)
        self._finish_history_task(state=state, message=msg, returncode=rc)
        await self._broadcast({
            "type": "status",
            "state": state,
            "job": job,
            "message": msg,
            "output_dir": self.current_output_dir,
            "sample_dir": self.current_sample_dir,
            "sample_config": self.current_sample_config,
            **self.current_runtime_info,
            "task_id": self.current_task_id,
            "queue_item_id": queue_item_id,
        })
        if (
            job == "preprocess"
            and rc == 0
            and not stop_requested
            and pending_train is not None
        ):
            await self._start_pending_training(pending_train)
            return
        if queue_item_id:
            queue_failed = (not stop_requested) and rc != 0
            if stop_requested:
                self._update_queue_item(queue_item_id, {
                    "state": "canceled",
                    "message": msg,
                    "finished_at": time.time(),
                    "finished_at_text": _format_ts(time.time()),
                })
            elif queue_failed:
                self._pause_queue_after_failure()
                self._update_queue_item(queue_item_id, {
                    "state": "error",
                    "message": msg,
                    "finished_at": time.time(),
                    "finished_at_text": _format_ts(time.time()),
                })
            else:
                self._update_queue_item(queue_item_id, {
                    "state": "done",
                    "message": msg,
                    "finished_at": time.time(),
                    "finished_at_text": _format_ts(time.time()),
                })
            self._save_queue()
            await self._broadcast_queue()
        self._schedule_queue_dispatch()

    async def _start_pending_training(self, pending: dict[str, Any]) -> None:
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

    def _repair_queue_on_startup(self) -> None:
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
        self._normalize_queue()
        items = self._queue["items"]
        out: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, dict):
                out.append(item)
        self._queue["items"] = out
        return out

    def _find_queue_item(self, item_id: str) -> dict[str, Any] | None:
        needle = str(item_id or "").strip()
        for item in self._queue_items():
            if str(item.get("id") or "") == needle:
                return item
        return None

    def _update_queue_item(self, item_id: str, patch: dict[str, Any]) -> None:
        item = self._find_queue_item(item_id)
        if item is not None:
            item.update(patch)

    def _attach_history_task_to_queue_item(self, item_id: str, task_id: str) -> None:
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
        items = self._queue_items()
        if len(items) <= MAX_QUEUE_ITEMS:
            return
        protected = [item for item in items if item.get("state") in {"queued", "running"}]
        finished = [item for item in items if item.get("state") not in {"queued", "running"}]
        keep_finished = max(0, MAX_QUEUE_ITEMS - len(protected))
        self._queue["items"] = [*protected, *(finished[-keep_finished:] if keep_finished else [])]

    def _save_queue(self) -> None:
        self._normalize_queue()
        QUEUE_DIR.mkdir(parents=True, exist_ok=True)
        self._queue["paused"] = self._queue_paused
        self._queue["failure_policy"] = self._queue_failure_policy
        self._queue["updated_at"] = time.time()
        self._queue["updated_at_text"] = _format_ts(self._queue["updated_at"])
        _write_training_queue_state(self._queue)

    async def _broadcast_queue(self) -> None:
        await self._broadcast({"type": "queue", **self.get_queue_snapshot()})

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

    def _ensure_launch_allowed(self, queue_item_id: str = "") -> None:
        launching = self._queue_launching_item_id
        same_queue_item = launching and str(queue_item_id or "") == launching
        if self.status == "running" or (launching and not same_queue_item):
            raise RuntimeError("已有任务在运行中")

    async def _start_queue_item(self, item: dict[str, Any]) -> None:
        variant = str(item.get("variant") or "")
        preset = str(item.get("preset") or "default")
        methods_subdir = str(item.get("methods_subdir") or "gui-methods")
        extra_args = list(item.get("extra_args") or [])
        queue_item_id = str(item.get("id") or "")
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

        await self._start_unlocked(
            variant,
            preset,
            extra_args,
            methods_subdir,
            config_file=str(item.get("runtime_config_file") or ""),
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

    async def _drain_output_buffer(self, buffer: str) -> str:
        """同时处理普通换行和 tqdm 常用的回车刷新。"""
        while True:
            split_at = _first_record_separator(buffer)
            if split_at is None:
                return buffer
            record = buffer[:split_at]
            buffer = buffer[split_at + 1:]
            if record.strip():
                await self._handle_output_record(record)

    async def _handle_output_record(self, text: str):
        text = _clean_output_record(text)
        if not text:
            return

        now = time.time()
        self._last_output_at = now
        await self._maybe_note_error_hint(text, ts=now)

        m = TQDM_RE.search(text)
        if m:
            cur = int(m.group("cur"))
            tot = int(m.group("tot"))
            label = m.group("label").strip() or "Training"
            rate_str = self._compute_rate(cur, tot)
            await self._broadcast({
                "type": "progress",
                "current": cur,
                "total": tot,
                "label": label,
                "rate": rate_str,
                "ts": now,
            })
            self._remember_log("progress", text, ts=now)
            metrics = self._extract_metrics_from_tqdm(text, cur)
            if metrics:
                await self._record_metric(metrics)
            return

        self._last_log_line = text
        record = self._remember_log("log", text, ts=now)
        await self._broadcast({"type": "log", **record})
        metrics = self._extract_metrics_from_log(text)
        if metrics:
            await self._record_metric(metrics)

    async def _record_metric(self, metrics: dict[str, Any]) -> None:
        item = dict(metrics)
        item.setdefault("ts", time.time())
        key = _live_metric_key(item)
        if key in self._metric_seen_keys:
            return
        self._metric_seen_keys.add(key)
        self._metrics_history.append(item)
        self._append_history_jsonl("metrics.jsonl", item)
        lr_log_record = self._remember_lr_change_log(item)
        await self._broadcast({"type": "metrics", **item})
        if lr_log_record:
            await self._broadcast({"type": "log", **lr_log_record})

    def _reset_metric_runtime_state(self) -> None:
        self._metrics_history = []
        self._metric_seen_keys = set()
        self._last_lr_log_text = ""

    def _remember_lr_change_log(self, metric: dict[str, Any]) -> dict[str, Any] | None:
        lr = _float_or_none(metric.get("lr"))
        if lr is None:
            return None
        lr_text = f"{lr:.2e}"
        if lr_text == self._last_lr_log_text:
            return None
        previous = self._last_lr_log_text
        self._last_lr_log_text = lr_text
        step = _int_or_none(metric.get("step"))
        step_text = f"step {step}: " if step is not None else ""
        change_text = f"{previous} → {lr_text}" if previous else lr_text
        ts = _float_or_none(metric.get("ts"))
        return self._remember_log("metric", f"[学习率] {step_text}{change_text}", ts=ts)

    async def _tail_progress_jsonl(self) -> None:
        while self.status == "running" and self._progress_jsonl_path:
            await self._ingest_progress_jsonl()
            await asyncio.sleep(1.0)
        await self._ingest_progress_jsonl(final=True)

    async def _ingest_progress_jsonl(self, *, final: bool = False) -> None:
        path = self._progress_jsonl_path
        if path is None or not path.exists():
            return
        lock = self._progress_jsonl_lock
        if lock is None:
            self._progress_jsonl_lock = asyncio.Lock()
            lock = self._progress_jsonl_lock
        async with lock:
            try:
                size = path.stat().st_size
                if size < self._progress_jsonl_offset:
                    self._progress_jsonl_offset = 0
                with path.open("r", encoding="utf-8") as f:
                    f.seek(self._progress_jsonl_offset)
                    lines = f.readlines()
                    self._progress_jsonl_offset = f.tell()
            except OSError:
                return

        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                if final:
                    self._remember_log("log", f"[progress.jsonl] 无法解析: {line[:200]}")
                continue
            if not isinstance(event, dict):
                continue
            key = _progress_event_key(event)
            if key in self._progress_jsonl_seen:
                continue
            self._progress_jsonl_seen.add(key)
            await self._handle_progress_jsonl_event(event)

    async def _handle_progress_jsonl_event(self, event: dict[str, Any]) -> None:
        ev = str(event.get("ev") or "").strip()
        ts = _progress_event_wall_ts(event, self.current_task_dir)

        if ev == "run_start":
            total = _int_or_none(event.get("total_steps"))
            if total is not None and total > 0:
                self._progress_total_steps = total
                await self._broadcast({
                    "type": "progress",
                    "current": 0,
                    "total": total,
                    "label": "Training",
                    "rate": "",
                    "ts": ts,
                })
            record = self._remember_log("status", "结构化训练进度已开始", ts=ts)
            await self._broadcast({"type": "log", **record})
            return

        if ev in {"step", "val"}:
            metric = _metric_from_progress_jsonl_event(event, ts)
            if metric:
                await self._record_metric(metric)
            step = _int_or_none(event.get("global_step"))
            total = self._progress_total_steps
            if ev == "step" and step is not None and total:
                await self._broadcast({
                    "type": "progress",
                    "current": step,
                    "total": total,
                    "label": "Training",
                    "rate": "",
                    "ts": ts,
                })
            return

        if ev == "ckpt":
            ckpt_path = str(event.get("path") or "").strip()
            step = _int_or_none(event.get("global_step"))
            suffix = f" step={step}" if step is not None else ""
            record = self._remember_log("status", f"已保存检查点{suffix}: {ckpt_path}", ts=ts)
            await self._broadcast({"type": "log", **record})
            return

        if ev == "run_end":
            status = str(event.get("status") or "").strip() or "unknown"
            step = _int_or_none(event.get("final_step"))
            error = str(event.get("error") or "").strip()
            hint = await self._maybe_note_error_hint(error, ts=ts)
            line = f"结构化训练进度结束: {status}"
            if step is not None:
                line += f" final_step={step}"
            if error:
                line += f" error={_message_with_error_hint(error, hint)}"
            record = self._remember_log("status", line, ts=ts)
            await self._broadcast({"type": "log", **record})

    async def _maybe_note_error_hint(self, text: str, *, ts: float | None = None) -> str:
        hint = classify_training_error(text)
        if not hint:
            return self._detected_error_hint
        if self._detected_error_hint == hint:
            return hint
        self._detected_error_hint = hint
        record = self._remember_log("status", hint, ts=ts)
        await self._broadcast({"type": "log", **record})
        return hint

    def _remember_log(self, kind: str, line: str, ts: float | None = None) -> dict[str, Any]:
        record = {
            "id": self._next_log_id,
            "kind": kind,
            "line": line,
            "ts": ts if ts is not None else time.time(),
        }
        self._next_log_id += 1
        self._log_records.append(record)
        self._append_history_jsonl("logs.jsonl", record)
        if kind != "progress":
            self._last_log_line = line
        return record

    def _reserve_history_task_dir(self, job: str, methods_subdir: str, variant: str) -> Path:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        task_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{job}-{methods_subdir}-{variant}"
        task_id = _safe_task_id(task_id)
        task_dir = HISTORY_DIR / task_id
        suffix = 1
        while task_dir.exists():
            suffix += 1
            task_dir = HISTORY_DIR / f"{task_id}-{suffix}"
        task_dir.mkdir(parents=True, exist_ok=True)
        self.current_task_id = task_dir.name
        self.current_task_dir = task_dir
        return task_dir

    def _start_history_task(
        self,
        *,
        job: str,
        variant: str,
        preset: str,
        methods_subdir: str,
        output_dir: str,
        sample_dir: str,
        data_dirs: dict[str, str],
        sample_config: dict[str, Any],
        command: list[str],
        config_file: str | None = None,
        resume_info: dict[str, Any] | None = None,
        continue_info: dict[str, Any] | None = None,
        gpu_whitelist: list[int] | None = None,
        runtime_info: dict[str, str] | None = None,
        queue_info: dict[str, Any] | None = None,
    ) -> None:
        task_dir = self.current_task_dir or self._reserve_history_task_dir(job, methods_subdir, variant)
        now = time.time()
        runtime_meta = _runtime_meta(runtime_info)
        history_meta = _history_group_meta(
            methods_subdir,
            variant,
            preset,
            output_dir=output_dir,
            runtime_info=runtime_meta,
            resume_info=resume_info,
            task_id=task_dir.name,
        )
        default_name = _default_preprocess_history_name({
            "id": task_dir.name,
            "job": job,
            "output_dir": output_dir,
            **_continue_lora_history_meta(continue_info),
            **runtime_meta,
            **history_meta,
        })
        continue_meta = _continue_lora_history_meta(continue_info)
        queue_meta = queue_info if isinstance(queue_info, dict) else {}
        meta = {
            "id": task_dir.name,
            "name": default_name,
            "group": "",
            "archived": _default_history_archived(job),
            "job": job,
            "state": "running",
            "variant": variant,
            "preset": preset,
            "methods_subdir": methods_subdir,
            "output_dir": output_dir,
            "sample_dir": sample_dir,
            "source_image_dir": data_dirs.get("source_image_dir", ""),
            "resized_image_dir": data_dirs.get("resized_image_dir", ""),
            "lora_cache_dir": data_dirs.get("lora_cache_dir", ""),
            "data_dirs": data_dirs,
            "sample_config": sample_config,
            "command": command,
            "resume_from": resume_info or {},
            **continue_meta,
            "gpu_whitelist": gpu_whitelist or [],
            **runtime_meta,
            **history_meta,
            **queue_meta,
            "started_at": now,
            "started_at_text": _format_ts(now),
            "finished_at": None,
            "finished_at_text": "",
            "message": "",
            "returncode": None,
            "log_count": 0,
            "metric_count": 0,
        }
        _write_json_atomic(task_dir / "meta.json", meta)
        _write_config_snapshot(
            task_dir / "config.snapshot.toml",
            variant,
            preset,
            methods_subdir,
            config_file=config_file,
            continue_info=continue_info,
        )

    def _finish_history_task(self, *, state: str, message: str, returncode: int) -> None:
        if not self.current_task_dir:
            return
        meta = _read_json(self.current_task_dir / "meta.json")
        now = time.time()
        meta.update({
            "state": state,
            "finished_at": now,
            "finished_at_text": _format_ts(now),
            "message": message,
            "returncode": returncode,
            "log_count": _count_jsonl(self.current_task_dir / "logs.jsonl"),
            "metric_count": _count_jsonl(self.current_task_dir / "metrics.jsonl"),
        })
        _write_json_atomic(self.current_task_dir / "meta.json", meta)

    def _append_history_jsonl(self, filename: str, payload: dict[str, Any]) -> None:
        if not self.current_task_dir:
            return
        try:
            with (self.current_task_dir / filename).open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _write_terminal(self, text: str) -> None:
        try:
            sys.stdout.write(text)
            sys.stdout.flush()
        except Exception:
            pass

    def _compute_rate(self, cur: int, tot: int) -> str:
        now = time.monotonic()
        if self._anchor is None or cur <= 1:
            if cur >= 1:
                self._anchor = (now, cur)
            return ""
        anchor_time, anchor_step = self._anchor
        steps = cur - anchor_step
        if steps <= 0:
            return ""
        spi = (now - anchor_time) / steps
        return f"{spi:.2f}s/step"

    def _extract_metrics_from_tqdm(self, line: str, step: int) -> dict | None:
        parts = line.split(",")
        metrics: dict[str, Any] = {"step": step, "ts": time.time()}
        found = False
        for part in parts:
            part = part.strip()
            if "loss" in part.lower():
                try:
                    val = _extract_float_metric(part, ("avr_loss", "loss"))
                    if val is None:
                        continue
                    metrics["loss"] = val
                    found = True
                except ValueError:
                    continue
            elif "lr" in part.lower():
                try:
                    val = _extract_float_metric(part, ("lr", "learning_rate"))
                    if val is None:
                        continue
                    metrics["lr"] = val
                    found = True
                except ValueError:
                    continue
        return metrics if found else None

    def _extract_metrics_from_log(self, line: str) -> dict | None:
        metrics: dict[str, Any] = {"ts": time.time()}
        found = False
        lower = line.lower()
        if "loss" in lower:
            for m in re.finditer(r"(?:avr_)?loss[=:/\s]+([\d.eE\-+]+)", line, re.IGNORECASE):
                metrics["loss"] = float(m.group(1))
                found = True
                break
        if "cmmd" in lower or "val_" in lower:
            for m in re.finditer(r"(?:cmmd|val_[\w/]+)[=:/\s]+([\d.eE\-+]+)", line, re.IGNORECASE):
                try:
                    metrics["cmmd"] = float(m.group(1))
                    metrics["kind"] = "val"
                    found = True
                except ValueError:
                    pass
                break
        if "lr" in lower:
            for m in re.finditer(r"lr[=:/\s]+([\d.eE\-+]+)", line, re.IGNORECASE):
                try:
                    metrics["lr"] = float(m.group(1))
                    found = True
                except ValueError:
                    pass
                break
        if "step" in lower:
            for m in re.finditer(r"step[=:/\s]+(\d+)", line, re.IGNORECASE):
                metrics["step"] = int(m.group(1))
                break
        return metrics if found else None

    async def _monitor_system(self):
        while self.status == "running":
            stats = await _get_gpu_stats()
            if stats:
                stats["last_output_at"] = self._last_output_at
                stats["ts"] = time.time()
                self._append_history_jsonl("system.jsonl", stats)
                await self._broadcast({"type": "system", **stats})
            await asyncio.sleep(5)

    async def _broadcast(self, msg: dict):
        import json
        data = json.dumps(msg, ensure_ascii=False)
        dead = set()
        for ws in self._ws_clients:
            try:
                await ws.send_str(data)
            except (ConnectionResetError, RuntimeError):
                dead.add(ws)
        self._ws_clients -= dead


async def _get_gpu_stats() -> dict:
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        parts = stdout.decode().strip().split(", ")
        if len(parts) >= 4:
            return {
                "vram_used_gb": round(int(parts[0]) / 1024, 2),
                "vram_total_gb": round(int(parts[1]) / 1024, 2),
                "gpu_util": int(parts[2]),
                "gpu_temp": int(parts[3]),
            }
    except Exception:
        pass
    return {}


async def _list_available_gpus() -> list[dict[str, Any]]:
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
    except Exception:
        return []

    gpus: list[dict[str, Any]] = []
    for line in stdout.decode(errors="replace").splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) < 2:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        memory_total_mb = _int_or_none(parts[2]) if len(parts) >= 3 else None
        item: dict[str, Any] = {
            "index": index,
            "name": parts[1],
            "label": f"GPU {index} · {parts[1]}",
        }
        if memory_total_mb is not None:
            item["memory_total_mb"] = memory_total_mb
            item["memory_total_gb"] = round(memory_total_mb / 1024, 1)
        gpus.append(item)
    return gpus


def _normalize_gpu_whitelist(value: Any) -> list[int]:
    if value is None or value == "":
        return []
    raw_items = value if isinstance(value, list) else [value]
    out: list[int] = []
    for item in raw_items:
        try:
            index = int(str(item).strip())
        except (TypeError, ValueError):
            continue
        if index < 0 or index in out:
            continue
        out.append(index)
    return out


def inspect_continue_lora_weight(
    path: str,
    *,
    variant: str = "lora",
    preset: str = "default",
    methods_subdir: str = "gui-methods",
    config_file: str | None = None,
) -> dict[str, Any]:
    raw_path = str(path or "").strip()
    if not raw_path:
        raise ValueError("请填写 LoRA/LoKr 权重路径")
    weight_path = _resolve_display_path(raw_path)
    if weight_path is None:
        raise ValueError("权重路径不合法")
    if not _path_exists(weight_path):
        raise FileNotFoundError("权重文件不存在")
    if not weight_path.is_file():
        raise ValueError("权重路径不是文件")
    if weight_path.suffix.lower() != ".safetensors":
        raise ValueError("只支持 .safetensors 权重文件")
    if not os.access(weight_path, os.R_OK):
        raise ValueError("权重文件不可读取")

    metadata, keys = _read_safetensors_header(weight_path)
    kind = _detect_continue_lora_kind(keys, metadata)
    if kind not in CONTINUE_LORA_KINDS:
        raise ValueError("这个 safetensors 未识别为 LoRA 或 LoKr 权重")

    compatible, message = _continue_lora_compatibility(
        kind,
        variant=variant,
        preset=preset,
        methods_subdir=methods_subdir,
        config_file=config_file,
    )
    display_path = _display_project_path(str(weight_path))
    return {
        "ok": True,
        "name": weight_path.name,
        "abs_path": str(weight_path),
        "path": display_path,
        "kind": kind,
        "metadata": _safe_continue_lora_metadata(metadata),
        "compatible": compatible,
        "message": message,
    }


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
        raise ValueError(inspected.get("message") or "当前训练配置与继续训练权重不兼容")
    return {
        "continue_from_weight_abs_path": inspected["abs_path"],
        "continue_from_weight_name": inspected["name"],
        "continue_from_weight_kind": inspected["kind"],
    }


def _read_safetensors_header(path: Path) -> tuple[dict[str, str], list[str]]:
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as f:
            metadata = {str(k): str(v) for k, v in (f.metadata() or {}).items()}
            keys = list(f.keys())
        return metadata, keys
    except Exception as exc:
        raise ValueError(f"读取 safetensors 权重失败: {exc}") from exc


def _detect_continue_lora_kind(keys: list[str], metadata: dict[str, str]) -> str:
    meta_spec = str(metadata.get("ss_network_spec") or "").strip().lower()
    lowered_keys = [str(key).lower() for key in keys]
    has_lokr_keys = any("lokr_w1" in key or "lokr_w2" in key for key in lowered_keys)
    if has_lokr_keys:
        return "LoKr"
    if _continue_lora_has_unsupported_structure(lowered_keys, metadata, meta_spec):
        return ""
    has_plain_lora_keys = any(
        key.endswith(".lora_down.weight") or key.endswith(".lora_up.weight")
        for key in lowered_keys
    )
    if has_plain_lora_keys and meta_spec in CONTINUE_LORA_ACCEPTED_LORA_SPECS:
        return "LoRA"
    return ""


def _continue_lora_has_unsupported_structure(
    lowered_keys: list[str],
    metadata: dict[str, str],
    meta_spec: str,
) -> bool:
    if any(token in meta_spec for token in CONTINUE_LORA_UNSUPPORTED_SPEC_TOKENS):
        return True
    use_moe_style = str(metadata.get("ss_use_moe_style") or "").strip().lower()
    if use_moe_style not in {"", "false", "none"}:
        return True
    router_source = str(metadata.get("ss_router_source") or "").strip().lower()
    if router_source not in {"", "false", "none"}:
        return True
    if _truthy(metadata.get("ss_use_chimera_hydra")):
        return True
    if any(key in metadata for key in ("ss_num_experts_content", "ss_num_experts_freq")):
        return True
    for key in lowered_keys:
        if key.startswith("reft_"):
            return True
        if key.endswith(".s_p") or key.endswith(".s_q"):
            return True
        if any(fragment in key for fragment in CONTINUE_LORA_UNSUPPORTED_KEY_FRAGMENTS):
            return True
    return False


def _safe_continue_lora_metadata(metadata: dict[str, str]) -> dict[str, str]:
    allowed = (
        "ss_network_spec",
        "ss_output_name",
        "ss_epoch",
        "ss_steps",
        "ss_num_epochs",
        "ss_max_train_steps",
        "ss_learning_rate",
        "ss_network_dim",
        "ss_network_alpha",
        "modelspec.architecture",
        "modelspec.implementation",
    )
    return {key: str(metadata[key]) for key in allowed if key in metadata}


def _continue_lora_compatibility(
    kind: str,
    *,
    variant: str,
    preset: str,
    methods_subdir: str,
    config_file: str | None = None,
) -> tuple[bool, str]:
    cfg = _load_config_file_config(config_file) if config_file else {}
    if not cfg:
        try:
            cfg = load_merged_config(variant, preset, methods_subdir)
        except Exception as exc:
            return False, f"无法读取当前训练配置用于兼容性检查: {exc}"
    current_kind = _continue_lora_config_kind(variant, methods_subdir, cfg)
    if current_kind == "LoKr":
        if kind == "LoKr":
            return True, "兼容：当前变体为 LoKr，会基于该 LoKr 权重继续训练"
        return False, "LoRA 权重不能直接用于 LoKr 变体；请切换到 LoRA 家族配置"
    if current_kind == "LoRA":
        if kind == "LoRA":
            return True, "兼容：当前配置属于 LoRA 家族，会基于该 LoRA 权重继续训练"
        return False, "LoKr 权重需要当前变体为 lokr，请先切换到 LoKr 变体"
    return False, "第一版只支持 LoRA / LoKr 家族配置继续训练"


def _continue_lora_config_kind(variant: str, methods_subdir: str, cfg: dict[str, Any]) -> str:
    module_name = str(cfg.get("network_module") or "")
    variant_key = str(variant or "").strip().lower()
    if _truthy(cfg.get("use_lokr")) or variant_key == "lokr":
        return "LoKr"
    if module_name and "lora_anima" not in module_name:
        return ""
    if str(methods_subdir or "") == "gui-methods":
        blocked = (
            "hydra",
            "fera",
            "reft",
            "ip_adapter",
            "easycontrol",
            "soft_tokens",
            "postfix",
            "chimera",
        )
        if any(token in variant_key for token in blocked):
            return ""
    if _truthy(cfg.get("use_chimera_hydra")):
        return ""
    if _truthy(cfg.get("add_reft")):
        return ""
    if str(cfg.get("use_moe_style") or "").strip().lower() not in {"", "false", "none"}:
        return ""
    return "LoRA"


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


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
    if whitelist:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in whitelist)


def _resolve_training_runtime_info(
    variant: str,
    preset: str,
    methods_subdir: str,
    extra_args: list[str],
    config_file: str | None = None,
) -> tuple[str, str, dict[str, Any]]:
    output_dir = "output/ckpt"
    cfg: dict[str, Any] = {}
    try:
        if config_file:
            cfg = _load_config_file_config(config_file)
        else:
            cfg = load_merged_config(variant, preset, methods_subdir)
        output_dir = str(cfg.get("output_dir") or output_dir)
    except Exception:
        pass

    for idx, arg in enumerate(extra_args):
        if arg == "--output_dir" and idx + 1 < len(extra_args):
            output_dir = str(extra_args[idx + 1])
            break
        if arg.startswith("--output_dir="):
            output_dir = arg.split("=", 1)[1]
            break

    rel_output = _display_project_path(output_dir) or "output/ckpt"
    return rel_output, f"{rel_output.rstrip('/')}/sample", _sample_config_from_cfg(cfg, extra_args)


def _ensure_training_data_dirs(
    variant: str,
    preset: str,
    methods_subdir: str,
    *,
    config_file: str | None = None,
) -> dict[str, str]:
    if config_file:
        cfg = apply_auto_data_dirs(_load_config_file_config(config_file), create=True)
    else:
        cfg = apply_auto_data_dirs(load_merged_config(variant, preset, methods_subdir), create=True)
    return {
        "source_image_dir": str(cfg.get("source_image_dir") or ""),
        "resized_image_dir": str(cfg.get("resized_image_dir") or ""),
        "lora_cache_dir": str(cfg.get("lora_cache_dir") or ""),
    }


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
        "# WebUI 继续训练来源",
        '# training_mode = "continue_lora"',
        f'# continue_from_weight_kind = "{_toml_comment_string(continue_info.get("continue_from_weight_kind"))}"',
        f'# continue_from_weight_name = "{_toml_comment_string(continue_info.get("continue_from_weight_name"))}"',
        f'# continue_from_weight_abs_path = "{_toml_comment_string(continue_info.get("continue_from_weight_abs_path"))}"',
        "",
    ]
    return base + "\n".join(lines)


def _toml_comment_string(value: Any) -> str:
    return str(value or "").replace("\\", "\\\\").replace('"', '\\"')


def _load_config_file_config(config_file: str) -> dict[str, Any]:
    path = _resolve_display_path(config_file)
    if path is None or not _path_exists(path):
        return {}
    try:
        return toml.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def toml_dumps_sorted(data: dict[str, Any]) -> str:
    try:
        import toml
        return toml.dumps({key: data[key] for key in sorted(data)})
    except Exception:
        return json.dumps(data, ensure_ascii=False, indent=2)


def _prepare_web_runtime_config(
    variant: str,
    preset: str,
    methods_subdir: str,
    *,
    source_config_file: str | None,
) -> dict[str, Any]:
    source_path = _resolve_display_path(source_config_file or "") if source_config_file else None
    if source_config_file and (source_path is None or not _path_exists(source_path) or not source_path.is_file()):
        raise FileNotFoundError(f"训练配置不存在: {source_config_file}")

    stem_source = source_path.stem if source_path is not None else variant
    run_stem = _safe_run_stem(stem_source or variant or "run")
    run_dir = _unique_runtime_dir(resolve_output_root(), run_stem)

    model_cache_dir = run_dir / "model_cache"
    dataset_cache_dir = run_dir / "dataset_cache"
    training_output_dir = run_dir / "training_output"
    sample_dir = training_output_dir / "sample"
    logs_dir = model_cache_dir / "logs"
    torchinductor_dir = model_cache_dir / "torchinductor"
    triton_dir = model_cache_dir / "triton"

    for path in (
        model_cache_dir,
        dataset_cache_dir,
        training_output_dir,
        sample_dir,
        logs_dir,
        torchinductor_dir,
        triton_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    cfg = load_merged_config(variant, preset, methods_subdir)
    if source_path is not None:
        source_cfg = _load_config_file_config(_display_settings_path(source_path))
        if source_cfg:
            cfg.update(source_cfg)
    source_rows = _dataset_rows_for_estimate(cfg)
    if not source_rows:
        raise ValueError("请先配置至少一个数据集路径")

    runtime_rows: list[dict[str, Any]] = []
    for index, row in enumerate(source_rows, start=1):
        group_dir = dataset_cache_dir / f"dataset-{index:02d}"
        resized_dir = group_dir / "resized"
        lora_dir = group_dir / "lora"
        resized_dir.mkdir(parents=True, exist_ok=True)
        lora_dir.mkdir(parents=True, exist_ok=True)
        source_dir = str(
            row.get("source_dir")
            or row.get("source_image_dir")
            or row.get("image_dir")
            or ""
        ).strip()
        source_dir = _prepare_runtime_nl_tag_mix_source(row, group_dir, source_dir)
        runtime_rows.append({
            "source_dir": source_dir,
            "image_dir": _display_settings_path(resized_dir),
            "cache_dir": _display_settings_path(lora_dir),
            "num_repeats": row.get("num_repeats") or 1,
            "recursive": _bool_value_for_row(row.get("recursive"), True),
            "settings": row.get("settings") if isinstance(row.get("settings"), dict) else {},
        })
        trigger_clone = _normalize_trigger_clone(row.get("trigger_clone"))
        if trigger_clone["enabled"]:
            clone_source_dir = _prepare_runtime_trigger_clone_source(row, group_dir, source_dir)
            clone_resized_dir = group_dir / "trigger-clone-resized"
            clone_lora_dir = group_dir / "trigger-clone-lora"
            clone_resized_dir.mkdir(parents=True, exist_ok=True)
            clone_lora_dir.mkdir(parents=True, exist_ok=True)
            clone_settings = dict(row.get("settings") if isinstance(row.get("settings"), dict) else {})
            clone_settings["caption_source_mode"] = CAPTION_SOURCE_CAPTIONS_JSON
            clone_settings["prefer_json_caption"] = False
            runtime_rows.append({
                "source_dir": clone_source_dir,
                "image_dir": _display_settings_path(clone_resized_dir),
                "cache_dir": _display_settings_path(clone_lora_dir),
                "num_repeats": trigger_clone["num_repeats"],
                "recursive": _bool_value_for_row(row.get("recursive"), True),
                "settings": clone_settings,
            })

    original_config_path = run_dir / "config.original.toml"
    if source_path is not None:
        shutil.copy2(source_path, original_config_path)
    else:
        original_config_path.write_text(toml_dumps_sorted(cfg), encoding="utf-8")

    dataset_config_path = run_dir / "dataset.runtime.toml"
    runtime_cfg = dict(cfg)
    dataset_doc = _build_dataset_config_doc(
        runtime_rows,
        runtime_cfg,
        prefer_train_batch_size=True,
        include_preprocess_settings=False,
    )
    dataset_config_path.write_text(dataset_doc, encoding="utf-8")

    first_row = runtime_rows[0]
    runtime_cfg.update({
        "output_dir": _display_settings_path(training_output_dir),
        "logging_dir": _display_settings_path(logs_dir),
        "dataset_config": _display_settings_path(dataset_config_path),
        "source_image_dir": first_row["source_dir"],
        "resized_image_dir": first_row["image_dir"],
        "lora_cache_dir": first_row["cache_dir"],
    })
    runtime_config_path = run_dir / "config.runtime.toml"
    runtime_config_path.write_text(toml_dumps_sorted(runtime_cfg), encoding="utf-8")

    data_dirs = {
        "source_image_dir": first_row["source_dir"],
        "resized_image_dir": first_row["image_dir"],
        "lora_cache_dir": first_row["cache_dir"],
    }
    history_source_config_file = _display_settings_path(source_path) if source_path is not None else ""
    _write_runtime_run_meta(
        run_dir,
        {
            "history_source_config_file": history_source_config_file,
            "source_config_file": history_source_config_file,
            "run_dir": _display_settings_path(run_dir),
            "runtime_config_file": _display_settings_path(runtime_config_path),
            "original_config_file": _display_settings_path(original_config_path),
            "dataset_config_file": _display_settings_path(dataset_config_path),
        },
    )
    return {
        "run_dir": _display_settings_path(run_dir),
        "runtime_config_file": _display_settings_path(runtime_config_path),
        "original_config_file": _display_settings_path(original_config_path),
        "dataset_config_file": _display_settings_path(dataset_config_path),
        "output_dir": runtime_cfg["output_dir"],
        "sample_dir": _display_settings_path(sample_dir),
        "model_cache_dir": _display_settings_path(model_cache_dir),
        "dataset_cache_dir": _display_settings_path(dataset_cache_dir),
        "training_output_dir": runtime_cfg["output_dir"],
        "logs_dir": runtime_cfg["logging_dir"],
        "torchinductor_cache_dir": _display_settings_path(torchinductor_dir),
        "triton_cache_dir": _display_settings_path(triton_dir),
        "history_source_config_file": history_source_config_file,
        "data_dirs": data_dirs,
        "dataset_dirs": runtime_rows,
        "sample_config": _sample_config_from_cfg(runtime_cfg, []),
    }


def _apply_runtime_env(env: dict[str, str], runtime: dict[str, Any] | None) -> None:
    if not runtime:
        return
    env["ANIMA_RUNTIME_CONFIG"] = str(runtime.get("runtime_config_file") or "")
    env["TORCHINDUCTOR_CACHE_DIR"] = str(runtime.get("torchinductor_cache_dir") or "")
    env["TRITON_CACHE_DIR"] = str(runtime.get("triton_cache_dir") or "")


def _runtime_meta(runtime: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(runtime, dict):
        return {}
    return {
        key: str(runtime.get(key) or "")
        for key in RUNTIME_META_KEYS
        if str(runtime.get(key) or "").strip()
    }


def _delete_queue_item_runtime_dir(item: dict[str, Any]) -> dict[str, Any]:
    run_dir = _queue_item_runtime_delete_dir(item)
    if run_dir is None:
        return {"deleted": False, "runtime_dir": ""}

    runtime_dir = _display_settings_path(run_dir)
    if not _path_exists(run_dir):
        return {"deleted": False, "runtime_dir": runtime_dir}
    if not run_dir.is_dir():
        raise ValueError("运行缓存路径不是目录，已阻止删除")

    output_root = resolve_output_root()
    if run_dir == output_root or not _path_is_relative_to(run_dir, output_root):
        raise ValueError("运行缓存目录不在 WebUI 输出根目录内，已阻止删除")
    if not _is_web_runtime_dir(run_dir):
        raise ValueError("运行缓存目录缺少 WebUI runtime 标记，已阻止删除")
    _validate_queue_runtime_dir_match(item, run_dir)

    shutil.rmtree(run_dir)
    return {"deleted": True, "runtime_dir": runtime_dir}


def _queue_item_runtime_dir_label(item: dict[str, Any]) -> str:
    run_dir = _queue_item_runtime_delete_dir(item)
    return _display_settings_path(run_dir) if run_dir is not None else ""


def _queue_item_runtime_delete_dir(item: dict[str, Any]) -> Path | None:
    runtime_info = item.get("runtime_info") if isinstance(item.get("runtime_info"), dict) else {}
    run_dir = _resolve_display_path(str(runtime_info.get("run_dir") or ""))
    if run_dir is not None:
        return run_dir
    for value in (
        str(runtime_info.get("runtime_config_file") or ""),
        str(item.get("runtime_config_file") or ""),
    ):
        path = _resolve_display_path(value)
        if path is not None and path.name == "config.runtime.toml":
            return path.parent
    output_dir = _resolve_display_path(str(runtime_info.get("training_output_dir") or runtime_info.get("output_dir") or ""))
    if output_dir is not None and output_dir.name == "training_output":
        return output_dir.parent
    return None


def _validate_queue_runtime_dir_match(item: dict[str, Any], run_dir: Path) -> None:
    expected_config = _resolve_display_path(str(item.get("runtime_config_file") or ""))
    runtime_info = item.get("runtime_info") if isinstance(item.get("runtime_info"), dict) else {}
    info_config = _resolve_display_path(str(runtime_info.get("runtime_config_file") or ""))
    valid_configs = [path.resolve() for path in (expected_config, info_config) if path is not None]
    actual_config = (run_dir / "config.runtime.toml").resolve()
    if not valid_configs:
        raise ValueError("队列记录缺少 runtime 配置，已阻止删除")
    if actual_config not in valid_configs:
        raise ValueError("运行缓存目录与队列记录的 runtime 配置不匹配，已阻止删除")
    run_meta = _read_runtime_run_meta(run_dir)
    meta_config = _resolve_display_path(str(run_meta.get("runtime_config_file") or ""))
    if meta_config is not None and meta_config.resolve() != actual_config:
        raise ValueError("运行缓存目录的 runtime 元数据不匹配，已阻止删除")


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _write_runtime_run_meta(run_dir: Path, payload: dict[str, Any]) -> None:
    meta = {key: value for key, value in payload.items() if str(value or "").strip()}
    _write_json(run_dir / RUN_META_FILE, meta)


def _read_runtime_run_meta(run_dir: Path) -> dict[str, Any]:
    meta = _read_json(run_dir / RUN_META_FILE)
    return meta if isinstance(meta, dict) else {}


def _runtime_from_config_file(
    config_file: str | None,
    *,
    source_config_file: str | None = None,
) -> dict[str, Any] | None:
    if not config_file:
        return None
    config_path = _resolve_display_path(config_file)
    if config_path is None or not _path_exists(config_path) or not config_path.is_file():
        return None
    run_dir = config_path.parent
    model_cache_dir = run_dir / "model_cache"
    training_output_dir = run_dir / "training_output"
    dataset_cache_dir = run_dir / "dataset_cache"
    if not model_cache_dir.is_dir() or not training_output_dir.is_dir():
        return None

    cfg = _load_config_file_config(_display_settings_path(config_path))
    run_meta = _read_runtime_run_meta(run_dir)
    source_config_path = _resolve_display_path(source_config_file or "") if source_config_file else None
    history_source_config_file = (
        _display_settings_path(source_config_path)
        if source_config_path is not None
        else str(
            run_meta.get("history_source_config_file")
            or run_meta.get("source_config_file")
            or ""
        )
    )
    history_source_config_file = _display_project_path(history_source_config_file)
    source_dir = str(cfg.get("source_image_dir") or "")
    resized_dir = str(cfg.get("resized_image_dir") or "")
    lora_dir = str(cfg.get("lora_cache_dir") or "")
    sample_dir = training_output_dir / "sample"
    logs_dir = model_cache_dir / "logs"
    torchinductor_dir = model_cache_dir / "torchinductor"
    triton_dir = model_cache_dir / "triton"
    for path in (sample_dir, logs_dir, torchinductor_dir, triton_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": _display_settings_path(run_dir),
        "runtime_config_file": _display_settings_path(config_path),
        "original_config_file": _display_settings_path(run_dir / "config.original.toml"),
        "dataset_config_file": str(cfg.get("dataset_config") or ""),
        "output_dir": str(cfg.get("output_dir") or _display_settings_path(training_output_dir)),
        "sample_dir": _display_settings_path(sample_dir),
        "model_cache_dir": _display_settings_path(model_cache_dir),
        "dataset_cache_dir": _display_settings_path(dataset_cache_dir),
        "training_output_dir": str(cfg.get("output_dir") or _display_settings_path(training_output_dir)),
        "logs_dir": str(cfg.get("logging_dir") or _display_settings_path(logs_dir)),
        "torchinductor_cache_dir": _display_settings_path(torchinductor_dir),
        "triton_cache_dir": _display_settings_path(triton_dir),
        "history_source_config_file": history_source_config_file,
        "data_dirs": {
            "source_image_dir": source_dir,
            "resized_image_dir": resized_dir,
            "lora_cache_dir": lora_dir,
        },
    }


def _clone_frozen_runtime_config(
    config_file: str,
    *,
    source_config_file: str = "",
    reset_data_dirs: bool = False,
) -> dict[str, Any]:
    config_path = _resolve_display_path(config_file)
    if config_path is None or not _path_exists(config_path) or not config_path.is_file():
        raise FileNotFoundError(f"冻结运行配置不存在: {config_file}")

    cfg = _load_config_file_config(_display_settings_path(config_path))
    if not cfg:
        raise ValueError("冻结运行配置为空或无法解析")

    previous_run_dir = config_path.parent
    run_stem = _safe_run_stem(f"{previous_run_dir.name or config_path.stem}-retry")
    run_dir = _unique_runtime_dir(resolve_output_root(), run_stem)
    model_cache_dir = run_dir / "model_cache"
    dataset_cache_dir = run_dir / "dataset_cache"
    training_output_dir = run_dir / "training_output"
    sample_dir = training_output_dir / "sample"
    logs_dir = model_cache_dir / "logs"
    torchinductor_dir = model_cache_dir / "torchinductor"
    triton_dir = model_cache_dir / "triton"
    for path in (
        model_cache_dir,
        dataset_cache_dir,
        training_output_dir,
        sample_dir,
        logs_dir,
        torchinductor_dir,
        triton_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    original_config_path = run_dir / "config.original.toml"
    old_original = previous_run_dir / "config.original.toml"
    shutil.copy2(old_original if _path_exists(old_original) else config_path, original_config_path)

    dataset_config_path = run_dir / "dataset.runtime.toml"
    runtime_rows = _dataset_rows_for_estimate(cfg)
    if not runtime_rows:
        raise ValueError("冻结运行配置缺少数据集路径，无法重新预处理" if reset_data_dirs else "冻结运行配置缺少数据集路径，无法重新入队")
    cloned_rows = _clone_runtime_dataset_rows(
        runtime_rows,
        dataset_cache_dir,
        copy_existing=not reset_data_dirs,
    )
    dataset_config_path.write_text(
        _build_dataset_config_doc(
            cloned_rows,
            cfg,
            prefer_train_batch_size=True,
            include_preprocess_settings=False,
        ),
        encoding="utf-8",
    )
    first_row = cloned_rows[0]
    data_dirs = {
        "source_image_dir": first_row["source_dir"],
        "resized_image_dir": first_row["image_dir"],
        "lora_cache_dir": first_row["cache_dir"],
    }

    runtime_cfg = dict(cfg)
    runtime_cfg.update({
        "output_dir": _display_settings_path(training_output_dir),
        "logging_dir": _display_settings_path(logs_dir),
        "dataset_config": _display_settings_path(dataset_config_path),
    })
    runtime_cfg.update({key: value for key, value in data_dirs.items() if value})

    runtime_config_path = run_dir / "config.runtime.toml"
    runtime_config_path.write_text(toml_dumps_sorted(runtime_cfg), encoding="utf-8")

    run_meta = _read_runtime_run_meta(previous_run_dir)
    history_source_config_file = _display_project_path(
        source_config_file
        or str(run_meta.get("history_source_config_file") or run_meta.get("source_config_file") or "")
    )
    _write_runtime_run_meta(
        run_dir,
        {
            "history_source_config_file": history_source_config_file,
            "source_config_file": history_source_config_file,
            "run_dir": _display_settings_path(run_dir),
            "runtime_config_file": _display_settings_path(runtime_config_path),
            "original_config_file": _display_settings_path(original_config_path),
            "dataset_config_file": _display_settings_path(dataset_config_path),
        },
    )
    return {
        "run_dir": _display_settings_path(run_dir),
        "runtime_config_file": _display_settings_path(runtime_config_path),
        "original_config_file": _display_settings_path(original_config_path),
        "dataset_config_file": _display_settings_path(dataset_config_path),
        "output_dir": runtime_cfg["output_dir"],
        "sample_dir": _display_settings_path(sample_dir),
        "model_cache_dir": _display_settings_path(model_cache_dir),
        "dataset_cache_dir": _display_settings_path(dataset_cache_dir),
        "training_output_dir": runtime_cfg["output_dir"],
        "logs_dir": runtime_cfg["logging_dir"],
        "torchinductor_cache_dir": _display_settings_path(torchinductor_dir),
        "triton_cache_dir": _display_settings_path(triton_dir),
        "history_source_config_file": history_source_config_file,
        "data_dirs": data_dirs,
        "sample_config": _sample_config_from_cfg(runtime_cfg, []),
    }


def _clone_runtime_dataset_rows(
    runtime_rows: list[dict[str, Any]],
    dataset_cache_dir: Path,
    *,
    copy_existing: bool,
) -> list[dict[str, Any]]:
    cloned_rows: list[dict[str, Any]] = []
    for index, row in enumerate(runtime_rows, start=1):
        group_dir = dataset_cache_dir / f"dataset-{index:02d}"
        resized_dir = group_dir / _runtime_dataset_child_name(
            str(row.get("image_dir") or row.get("resized_image_dir") or ""),
            default="resized",
            allowed={"resized", "trigger-clone-resized"},
        )
        lora_dir = group_dir / _runtime_dataset_child_name(
            str(row.get("cache_dir") or row.get("lora_cache_dir") or ""),
            default="lora",
            allowed={"lora", "trigger-clone-lora"},
        )
        resized_dir.mkdir(parents=True, exist_ok=True)
        lora_dir.mkdir(parents=True, exist_ok=True)
        if copy_existing:
            _copy_runtime_dataset_dir(str(row.get("image_dir") or row.get("resized_image_dir") or ""), resized_dir)
            _copy_runtime_dataset_dir(str(row.get("cache_dir") or row.get("lora_cache_dir") or ""), lora_dir)
        source_dir = str(row.get("source_dir") or row.get("source_image_dir") or row.get("image_dir") or "")
        source_path = _resolve_display_path(source_dir)
        source_target = group_dir / _runtime_dataset_child_name(
            source_dir,
            default="source",
            allowed={"source", "trigger-clone-source"},
        )
        if (
            copy_existing
            and source_path
            and _is_materialized_runtime_source_dir(source_path)
            and source_path.resolve() != source_target.resolve()
        ):
            _copy_runtime_dataset_dir(source_dir, source_target)
            source_dir = _display_settings_path(source_target)
        cloned_rows.append({
            "source_dir": source_dir,
            "image_dir": _display_settings_path(resized_dir),
            "cache_dir": _display_settings_path(lora_dir),
            "num_repeats": row.get("num_repeats") or 1,
            "recursive": _bool_value_for_row(row.get("recursive"), True),
            "settings": row.get("settings") if isinstance(row.get("settings"), dict) else {},
        })
    return cloned_rows


def _runtime_dataset_child_name(value: str, *, default: str, allowed: set[str]) -> str:
    path = _resolve_display_path(value)
    name = path.name if path is not None else ""
    return name if name in allowed else default


def _bool_value_for_row(value: Any, fallback: bool = False) -> bool:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _prepare_runtime_nl_tag_mix_source(row: dict[str, Any], group_dir: Path, source_dir: str) -> str:
    mix = _normalize_nl_tag_mix(row.get("nl_tag_mix"))
    if not mix.get("enabled"):
        return source_dir
    source_path = _resolve_display_path(source_dir)
    if source_path is None:
        raise ValueError("captions格式nl/tag权重调整需要填写原始数据集路径")
    if not source_path.is_dir():
        raise ValueError(f"captions格式nl/tag权重调整失败: {source_dir} 不是目录")
    target_dir = group_dir / "source"
    target_dir.mkdir(parents=True, exist_ok=True)
    caption_settings = _nl_tag_mix_caption_settings(row)
    manifest = _build_nl_tag_mix_source(
        source_path,
        target_dir,
        tag_ratio=float(mix.get("tag_ratio") or 0.0),
        recursive=_bool_value_for_row(row.get("recursive"), True),
        **caption_settings,
    )
    (target_dir / "results.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return _display_settings_path(target_dir)


def _prepare_runtime_trigger_clone_source(row: dict[str, Any], group_dir: Path, source_dir: str) -> str:
    clone = _normalize_trigger_clone(row.get("trigger_clone"))
    if not clone["enabled"]:
        return source_dir
    prompt = clone["prompt"]
    if not prompt:
        raise ValueError("触发提示词图像克隆需要填写触发提示词")
    source_path = _resolve_display_path(source_dir)
    if source_path is None:
        raise ValueError("触发提示词图像克隆需要填写原始数据集路径")
    if not source_path.is_dir():
        raise ValueError(f"触发提示词图像克隆失败: {source_dir} 不是目录")
    target_dir = group_dir / "trigger-clone-source"
    target_dir.mkdir(parents=True, exist_ok=True)
    recursive = _bool_value_for_row(row.get("recursive"), True)
    images = _nl_tag_mix_image_files(source_path, DATASET_IMAGE_EXTS, recursive=recursive)
    if not images:
        raise ValueError("触发提示词图像克隆失败: 数据集目录里没有可训练图片")

    captions_json: dict[str, list[str]] = {}
    items: list[dict[str, str]] = []
    for image_path in images:
        rel_image = _nl_tag_mix_relative_image_path(image_path, source_path)
        target_image = target_dir / rel_image
        target_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_image)
        rel_key = rel_image.as_posix()
        captions_json[rel_key] = [prompt]
        items.append({
            "image": _display_settings_path(image_path),
            "target": _display_settings_path(target_image),
            "caption_key": rel_key,
        })

    (target_dir / CAPTIONS_JSON_FILE).write_text(
        json.dumps(captions_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "prompt": prompt,
        "num_repeats": clone["num_repeats"],
        "recursive": recursive,
        "source_dir": _display_settings_path(source_path),
        "target_dir": _display_settings_path(target_dir),
        "caption_source_mode": CAPTION_SOURCE_CAPTIONS_JSON,
        "total": len(items),
        "items": items,
    }
    (target_dir / "trigger-clone-results.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return _display_settings_path(target_dir)


def _nl_tag_mix_caption_settings(row: dict[str, Any]) -> dict[str, Any]:
    settings = row.get("settings") if isinstance(row.get("settings"), dict) else {}
    prefer_json_caption = bool(settings.get("prefer_json_caption"))
    return {
        "caption_source_mode": normalize_caption_source_mode(
            settings.get("caption_source_mode"),
            prefer_json_caption,
        ),
        "caption_extension": str(settings.get("caption_extension") or ".txt"),
        "prefer_json_caption": prefer_json_caption,
    }


def _build_nl_tag_mix_source(
    source_dir: Path,
    target_dir: Path,
    *,
    tag_ratio: float,
    recursive: bool = True,
    caption_source_mode: str = "auto",
    caption_extension: str = ".txt",
    prefer_json_caption: bool = False,
) -> dict[str, Any]:
    samples = _classify_nl_tag_mix_samples(
        source_dir,
        recursive=recursive,
        caption_source_mode=caption_source_mode,
        caption_extension=caption_extension,
        prefer_json_caption=prefer_json_caption,
    )
    if not samples:
        raise ValueError("captions格式nl/tag权重调整失败: 数据集目录里没有可训练图片")
    captions_json_samples = [sample for sample in samples if sample.get("caption_entries")]
    plain_samples = [sample for sample in samples if not sample.get("caption_entries")]
    selected = [
        *captions_json_samples,
        *_select_nl_tag_mix_samples(plain_samples, tag_ratio=tag_ratio),
    ]
    items: list[dict[str, Any]] = []
    counts = {"tag": 0, "nl": 0}
    available_counts = {"tag": 0, "nl": 0}
    caption_available_counts = {"tag": 0, "nl": 0}
    caption_counts = {"tag": 0, "nl": 0}
    missing_caption_count = 0
    captions_json: dict[str, list[str]] = {}
    captions_json_target = target_dir / CAPTIONS_JSON_FILE
    for sample in samples:
        available_counts[sample["source"]] += 1
        available_entry_counts = _nl_tag_mix_source_counts(sample.get("caption_entries") or [])
        if available_entry_counts["tag"] or available_entry_counts["nl"]:
            caption_available_counts["tag"] += available_entry_counts["tag"]
            caption_available_counts["nl"] += available_entry_counts["nl"]
        elif sample.get("caption_path"):
            caption_available_counts[sample["source"]] += 1
        if not sample.get("caption_path"):
            missing_caption_count += 1
    for sample in sorted(selected, key=lambda item: item["image_path"].as_posix()):
        image_path = sample["image_path"]
        rel_image = _nl_tag_mix_relative_image_path(image_path, source_dir)
        target_image = target_dir / rel_image
        target_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_image)
        caption_source = sample.get("caption_source")
        selected_entries: list[dict[str, Any]] = []
        selected_entry_counts = {"tag": 0, "nl": 0}
        if getattr(caption_source, "from_captions_json", False):
            selected_entries = _select_nl_tag_caption_entries(
                sample.get("caption_entries") or [],
                tag_ratio=tag_ratio,
            )
            selected_entry_counts = _nl_tag_mix_source_counts(selected_entries)
            caption_counts["tag"] += selected_entry_counts["tag"]
            caption_counts["nl"] += selected_entry_counts["nl"]
            captions_json[rel_image.as_posix()] = [
                str(entry.get("text") or "")
                for entry in selected_entries
                if str(entry.get("text") or "").strip()
            ]
        elif sample.get("caption_path"):
            caption_counts[sample["source"]] += 1
        copied_captions = _copy_nl_tag_caption_sidecars(
            image_path,
            target_image,
            target_dir,
            caption_source,
            captions_json_path=captions_json_target,
        )
        source_kind = (
            _nl_tag_mix_dominant_source(selected_entry_counts)
            if selected_entries
            else sample["source"]
        )
        counts[source_kind] += 1
        item = {
            "stem": image_path.stem,
            "source": source_kind,
            "classification": sample["classification"],
            "image": _display_settings_path(image_path),
            "target": _display_settings_path(target_image),
            "caption": _display_settings_path(sample["caption_path"]) if sample.get("caption_path") else "",
            "captions": copied_captions,
            "caption_source_mode": sample.get("caption_source_mode", ""),
        }
        if selected_entries:
            item["caption_entry_count"] = len(sample.get("caption_entries") or [])
            item["weighted_caption_count"] = len(selected_entries)
            item["available_caption_counts"] = _nl_tag_mix_source_counts(sample.get("caption_entries") or [])
            item["actual_caption_counts"] = selected_entry_counts
            item["selected_caption_indices"] = [int(entry.get("index", 0)) for entry in selected_entries]
        items.append(item)
    if captions_json:
        captions_json_target.write_text(
            json.dumps(captions_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return {
        "tag_ratio": min(1.0, max(0.0, tag_ratio)),
        "classification_method": NL_TAG_MIX_CLASSIFICATION_METHOD,
        "caption_source_mode": caption_source_mode,
        "recursive": bool(recursive),
        "source_dir": _display_settings_path(source_dir),
        "available_tag_count": available_counts["tag"],
        "available_nl_count": available_counts["nl"],
        "actual_tag_count": counts["tag"],
        "actual_nl_count": counts["nl"],
        "available_tag_caption_count": caption_available_counts["tag"],
        "available_nl_caption_count": caption_available_counts["nl"],
        "actual_tag_caption_count": caption_counts["tag"],
        "actual_nl_caption_count": caption_counts["nl"],
        "total": len(items),
        "missing_caption_count": missing_caption_count,
        "items": items,
    }


def _classify_nl_tag_mix_samples(
    source_dir: Path,
    *,
    recursive: bool = True,
    caption_source_mode: str = "auto",
    caption_extension: str = ".txt",
    prefer_json_caption: bool = False,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for image_path in _nl_tag_mix_image_files(source_dir, DATASET_IMAGE_EXTS, recursive=recursive):
        caption_source = _nl_tag_mix_caption_source(
            image_path,
            caption_source_mode=caption_source_mode,
            caption_extension=caption_extension,
            prefer_json_caption=prefer_json_caption,
            captions_root=source_dir,
        )
        caption_texts = caption_source.caption_texts()
        caption_entries = (
            _nl_tag_mix_caption_entries(caption_texts)
            if getattr(caption_source, "from_captions_json", False)
            else []
        )
        if caption_entries:
            entry_counts = _nl_tag_mix_source_counts(caption_entries)
            source_kind = _nl_tag_mix_dominant_source(entry_counts)
            classification = {
                "kind": source_kind,
                "reason": "captions_json_caption_entries_majority",
                "method": NL_TAG_MIX_CLASSIFICATION_METHOD,
                "metrics": {
                    "caption_count": len(caption_entries),
                    "tag_caption_count": entry_counts["tag"],
                    "nl_caption_count": entry_counts["nl"],
                },
            }
        else:
            caption_text = "\n".join(caption_texts)
            classification = _classify_nl_tag_caption_text(caption_text)
            source_kind = classification["kind"]
        samples.append({
            "image_path": image_path,
            "caption_path": caption_source.path,
            "caption_source": caption_source,
            "caption_source_mode": caption_source.detected_mode,
            "caption_texts": caption_texts,
            "caption_entries": caption_entries,
            "source": source_kind,
            "classification": classification,
        })
    return samples


def _nl_tag_mix_caption_entries(caption_texts: list[str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for index, text in enumerate(caption_texts):
        clean_text = str(text or "").strip()
        if not clean_text:
            continue
        classification = _classify_nl_tag_caption_text(clean_text)
        entries.append({
            "index": index,
            "text": clean_text,
            "source": classification["kind"],
            "classification": classification,
        })
    return entries


def _nl_tag_mix_source_counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"tag": 0, "nl": 0}
    for entry in entries:
        source = str(entry.get("source") or "")
        if source in counts:
            counts[source] += 1
    return counts


def _nl_tag_mix_dominant_source(counts: dict[str, int]) -> str:
    return "nl" if int(counts.get("nl") or 0) > int(counts.get("tag") or 0) else "tag"


def _cycle_nl_tag_entries(entries: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or not entries:
        return []
    return [entries[index % len(entries)] for index in range(count)]


def _select_nl_tag_caption_entries(
    entries: list[dict[str, Any]],
    *,
    tag_ratio: float,
) -> list[dict[str, Any]]:
    if not entries:
        return []
    ratio = min(1.0, max(0.0, tag_ratio))
    total = len(entries)
    tag_entries = [entry for entry in entries if entry["source"] == "tag"]
    nl_entries = [entry for entry in entries if entry["source"] == "nl"]
    tag_quota = int(round(total * ratio))
    nl_quota = total - tag_quota
    selected = [
        *_cycle_nl_tag_entries(tag_entries, tag_quota),
        *_cycle_nl_tag_entries(nl_entries, nl_quota),
    ]
    if len(selected) < total:
        fallback = tag_entries + nl_entries or entries
        selected.extend(_cycle_nl_tag_entries(fallback, total - len(selected)))
    return selected[:total]


def _nl_tag_mix_relative_image_path(image_path: Path, source_dir: Path) -> Path:
    try:
        return image_path.resolve().relative_to(source_dir.resolve())
    except ValueError:
        return Path(image_path.name)


def _select_nl_tag_mix_samples(samples: list[dict[str, Any]], *, tag_ratio: float) -> list[dict[str, Any]]:
    ratio = min(1.0, max(0.0, tag_ratio))
    tag_samples = [sample for sample in samples if sample["source"] == "tag"]
    nl_samples = [sample for sample in samples if sample["source"] == "nl"]
    tag_quota = int(round(len(samples) * ratio))
    nl_quota = len(samples) - tag_quota
    selected = [*tag_samples[:tag_quota], *nl_samples[:nl_quota]]
    if len(selected) < len(samples):
        selected_ids = {id(sample) for sample in selected}
        fill = [sample for sample in samples if id(sample) not in selected_ids]
        selected.extend(fill[:len(samples) - len(selected)])
    return sorted(selected, key=lambda sample: sample["image_path"].name)


def _copy_nl_tag_caption_sidecars(
    image_path: Path,
    target_image: Path,
    target_dir: Path,
    caption_source=None,
    *,
    captions_json_path: Path | None = None,
) -> list[str]:
    if getattr(caption_source, "from_captions_json", False):
        return [_display_settings_path(captions_json_path)] if captions_json_path is not None else []
    copied: list[str] = []
    copied_sources: set[Path] = set()
    if getattr(caption_source, "path", None) is not None:
        source_path = caption_source.path
        if source_path.is_file():
            target = target_image.with_suffix(source_path.suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target)
            copied.append(_display_settings_path(target))
            copied_sources.add(source_path.resolve())
    for ext in DATASET_CAPTION_EXTS:
        source = image_path.with_suffix(ext)
        if not source.is_file() or source.resolve() in copied_sources:
            continue
        target = target_image.with_suffix(ext)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(_display_settings_path(target))
    return copied


def _copy_runtime_dataset_dir(source: str, target: Path) -> None:
    source_path = _resolve_display_path(source)
    if source_path is None or not _path_exists(source_path) or not source_path.is_dir():
        return
    if source_path.resolve() == target.resolve():
        return
    shutil.copytree(source_path, target, dirs_exist_ok=True)


def _is_materialized_runtime_source_dir(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return path.name in {"source", "trigger-clone-source"} and "dataset_cache" in parts


def _unique_runtime_dir(output_root: Path, stem: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = output_root / f"{stem}-{timestamp}"
    candidate = base
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = output_root / f"{stem}-{timestamp}-{suffix}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


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


def _safe_run_stem(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip()).strip(".-")
    return clean[:80] or "run"


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
    if not HISTORY_DIR.exists():
        return []
    meta_paths = [
        meta_path for meta_path in HISTORY_DIR.glob("*/meta.json")
        if not _is_deleting_history_dir(meta_path.parent)
    ]
    for meta_path in meta_paths:
        meta = _read_json(meta_path)
        if meta:
            _repair_history_meta(meta_path, meta)
    _sync_bound_history_collection_groups(meta_paths)

    tasks = []
    for meta_path in meta_paths:
        if _is_deleting_history_dir(meta_path.parent):
            continue
        meta = _read_json(meta_path)
        if meta:
            task = _history_summary(meta, meta_path.parent)
            if include_archived or not task.get("archived"):
                tasks.append(task)
    tasks.sort(key=lambda item: item.get("started_at") or 0, reverse=True)
    if limit is None:
        limit = MAX_HISTORY_ITEMS
    if limit and limit > 0:
        return tasks[:limit]
    return tasks


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


def _history_meta_records(meta_paths: list[Path] | None = None) -> list[dict[str, Any]]:
    paths = meta_paths if meta_paths is not None else list(HISTORY_DIR.glob("*/meta.json")) if HISTORY_DIR.exists() else []
    records: list[dict[str, Any]] = []
    for meta_path in paths:
        if _is_deleting_history_dir(meta_path.parent):
            continue
        meta = _read_json(meta_path)
        if not meta:
            continue
        task_id = str(meta.get("id") or meta_path.parent.name).strip()
        work = dict(meta)
        _fill_history_runtime_meta(work)
        _fill_history_group_meta(work)
        records.append({
            "id": task_id,
            "path": meta_path,
            "meta": meta,
            "group_key": str(work.get("history_group_key") or "").strip(),
            "job": str(meta.get("job") or "").strip(),
            "group": _clean_history_text(meta.get("group"), max_len=48),
            "updated_at": float(meta.get("updated_at") or 0),
            "started_at": float(meta.get("started_at") or 0),
        })
    return records


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


def _sync_bound_history_collection_groups(meta_paths: list[Path] | None = None) -> int:
    records = _history_meta_records(meta_paths)
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
    if not HISTORY_DIR.exists():
        return 0
    count = 0
    for meta_path in HISTORY_DIR.glob("*/meta.json"):
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
        _write_json_atomic(meta_path, meta)
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
    snapshot_path = task_dir / "config.snapshot.toml"
    logs, logs_total, logs_truncated = _read_jsonl_limited(
        task_dir / "logs.jsonl",
        limit=MAX_HISTORY_DETAIL_LOG_RECORDS,
    )
    system, system_total, system_truncated = _read_jsonl_limited(
        task_dir / "system.jsonl",
        limit=MAX_HISTORY_DETAIL_SYSTEM_RECORDS,
    )
    metrics = _read_jsonl(task_dir / "metrics.jsonl")
    return {
        "ok": True,
        "task": _history_summary(meta, task_dir),
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
        task_metrics = _timeline_training_metrics(_metrics_from_history(task_logs, _read_jsonl(task_dir / "metrics.jsonl")))
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


def _metrics_from_history(logs: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[int | None, float | None, float | None]] = set()
    for item in metrics:
        normalized = _normalize_metric_record(item)
        if normalized is None:
            continue
        key = _metric_seen_key(normalized)
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)

    for record in logs:
        if record.get("kind") != "progress":
            continue
        parsed = _metric_from_progress_line(str(record.get("line") or ""))
        if parsed is None:
            continue
        if record.get("ts") is not None:
            parsed["ts"] = record.get("ts")
        key = _metric_seen_key(parsed)
        if key in seen:
            continue
        seen.add(key)
        out.append(parsed)

    out.sort(key=lambda item: (float(item.get("ts") or 0), int(item.get("step") or 0)))
    return out


def _timeline_training_metrics(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    max_step: int | None = None
    for item in metrics:
        step = _int_or_none(item.get("step"))
        if step is not None:
            if max_step is not None and step < max_step:
                continue
            max_step = step if max_step is None else max(max_step, step)
        out.append(item)
    return out


def _normalize_metric_record(item: dict[str, Any]) -> dict[str, Any] | None:
    out: dict[str, Any] = {}
    step = _int_or_none(item.get("step"))
    if step is not None:
        out["step"] = step
    for key in ("loss", "lr", "cmmd"):
        value = _float_or_none(item.get(key))
        if value is not None:
            out[key] = value
    if item.get("kind"):
        out["kind"] = str(item.get("kind"))
    if item.get("rate"):
        out["rate"] = str(item.get("rate"))
    ts = _float_or_none(item.get("ts"))
    if ts is not None:
        out["ts"] = ts
    if not any(key in out for key in ("loss", "lr", "cmmd")):
        return None
    return out


def _metric_from_progress_line(line: str) -> dict[str, Any] | None:
    out: dict[str, Any] = {}
    step_match = re.search(r"\|\s*(\d+)\/\d+\s*\[", line) or re.search(r"step[=:/\s]+(\d+)", line, re.IGNORECASE)
    if step_match:
        out["step"] = int(step_match.group(1))
    loss = _extract_float_metric(line, ("avr_loss", "loss"))
    if loss is not None:
        out["loss"] = loss
    lr = _extract_float_metric(line, ("lr", "learning_rate"))
    if lr is not None:
        out["lr"] = lr
    rate_match = re.search(r"([\d.]+\s*(?:s/it|it/s|s/step))", line, re.IGNORECASE)
    if rate_match:
        out["rate"] = rate_match.group(1).replace(" ", "")
    return out if any(key in out for key in ("loss", "lr")) else None


def _metric_seen_key(item: dict[str, Any]) -> tuple[int | None, float | None, float | None, float | None, str]:
    step = _int_or_none(item.get("step"))
    loss = _float_or_none(item.get("loss"))
    lr = _float_or_none(item.get("lr"))
    cmmd = _float_or_none(item.get("cmmd"))
    return (
        step,
        round(loss, 8) if loss is not None else None,
        round(lr, 12) if lr is not None else None,
        round(cmmd, 8) if cmmd is not None else None,
        str(item.get("kind") or ""),
    )


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
    start_step: int | None = None
    last_step: int | None = None
    for item in metrics:
        raw_step = _int_or_none(item.get("step"))
        display_step = (offset + raw_step) if raw_step is not None else ((last_step or offset) + 1)
        if last_step is not None and display_step <= last_step:
            display_step = last_step + 1
        item["display_step"] = display_step
        item["display_step_offset"] = offset
        if start_step is None:
            start_step = display_step
        last_step = display_step
    return start_step, last_step


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

    run_key = _history_delete_run_key(task)
    if not run_key:
        return task_ids
    seen = {task_id}
    for candidate in _list_history_tasks(include_archived=True):
        candidate_id = str(candidate.get("id") or "").strip()
        if not candidate_id or candidate_id in seen:
            continue
        if str(candidate.get("job") or "").strip() != "preprocess":
            continue
        if _history_delete_run_key(candidate) != run_key:
            continue
        task_ids.append(candidate_id)
        seen.add(candidate_id)
    return task_ids


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


def _is_web_runtime_dir(path: Path) -> bool:
    return (
        ((path / "config.runtime.toml").is_file() or (path / RUN_META_FILE).is_file())
        and (path / "model_cache").is_dir()
        and (path / "dataset_cache").is_dir()
        and (path / "training_output").is_dir()
    )


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
    _fill_history_group_meta(out)
    if not out["name"]:
        out["name"] = _default_preprocess_history_name(out)
    out["log_count"] = int(out.get("log_count") or _count_jsonl(task_dir / "logs.jsonl"))
    out["metric_count"] = int(out.get("metric_count") or _count_jsonl(task_dir / "metrics.jsonl"))
    return out


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
        return f"继续训练 {kind}{suffix}"
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
        })

    items.sort(key=_resume_state_sort_key)
    return items[:MAX_RESUME_CHECKPOINTS]


def _resume_checkpoint_diagnostic(task: dict[str, Any], checkpoints: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    raw_output_dir = str(task.get("output_dir") or "")
    output_dir = _resolve_display_path(raw_output_dir)
    diagnostic: dict[str, Any] = {
        "output_dir": raw_output_dir,
        "output_dir_resolved": _display_project_path(str(output_dir)) if output_dir is not None else "",
        "output_dir_valid": output_dir is not None,
        "output_dir_exists": bool(output_dir is not None and _path_exists(output_dir)),
        "output_dir_is_dir": bool(output_dir is not None and _path_exists(output_dir) and output_dir.is_dir()),
        "state_dir_count": 0,
        "train_state_count": 0,
        "checkpoint_count": len(checkpoints or []),
        "reason": "",
        "recommendation": "如需继续训练，可回到配置页选择这个任务导出的 LoRA/LoKr 权重做热启动；热启动不会恢复 optimizer、scheduler 和已完成步数。",
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

    state_dirs = [
        child
        for child in output_dir.iterdir()
        if child.is_dir() and not _is_transient_resume_state_dir(child.name)
    ]
    diagnostic["state_dir_count"] = len(state_dirs)
    diagnostic["train_state_count"] = sum(1 for child in state_dirs if _path_exists(child / "train_state.json"))
    if checkpoints:
        diagnostic["reason"] = "已找到可完整续训的状态目录。"
    elif diagnostic["train_state_count"]:
        diagnostic["reason"] = "输出目录里存在 train_state.json 状态目录，但不属于当前历史任务时间范围。"
    elif diagnostic["state_dir_count"]:
        diagnostic["reason"] = "输出目录里有子目录，但没有包含 train_state.json 的完整续训状态目录。"
    else:
        diagnostic["reason"] = "输出目录里没有完整续训状态目录；训练完成时 checkpoint-state 可能已被清理，或该配置未写出训练状态。"
    return diagnostic


def _is_transient_resume_state_dir(name: str) -> bool:
    return name.endswith((".tmp", ".backup"))


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


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


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
        "sample_sampler": "ddim",
        "message": "未启用训练中采样",
    }


def _sample_config_from_cfg(cfg: dict[str, Any], extra_args: list[str]) -> dict[str, Any]:
    sample_prompts = cfg.get("sample_prompts")
    sample_every_n_epochs = cfg.get("sample_every_n_epochs")
    sample_every_n_steps = cfg.get("sample_every_n_steps")
    sample_at_first = bool(cfg.get("sample_at_first", False))
    sample_sampler = str(cfg.get("sample_sampler") or "ddim")

    overrides = _cli_arg_overrides(extra_args)
    if "sample_prompts" in overrides:
        sample_prompts = overrides["sample_prompts"]
    if "sample_every_n_epochs" in overrides:
        sample_every_n_epochs = overrides["sample_every_n_epochs"]
    if "sample_every_n_steps" in overrides:
        sample_every_n_steps = overrides["sample_every_n_steps"]
    if "sample_at_first" in overrides:
        sample_at_first = True
    if "sample_sampler" in overrides:
        sample_sampler = str(overrides["sample_sampler"] or sample_sampler)

    epoch_freq = _positive_int_or_none(sample_every_n_epochs)
    step_freq = _positive_int_or_none(sample_every_n_steps)
    prompt_path = _resolve_display_path(str(sample_prompts or ""))
    prompt_exists = prompt_path.is_file() if prompt_path else False
    enabled = bool(prompt_path and prompt_exists and (epoch_freq is not None or step_freq is not None or sample_at_first))

    if not sample_prompts:
        message = "未设置 sample_prompts，训练不会生成样张"
    elif not prompt_exists:
        message = f"sample_prompts 文件不存在: {sample_prompts}"
    elif epoch_freq is None and step_freq is None and not sample_at_first:
        message = "未设置 sample_every_n_epochs 或 sample_every_n_steps，训练不会生成样张"
    else:
        message = "训练中采样已配置"

    return {
        "enabled": enabled,
        "sample_prompts": str(sample_prompts or ""),
        "sample_prompts_exists": prompt_exists,
        "sample_every_n_epochs": epoch_freq,
        "sample_every_n_steps": step_freq,
        "sample_at_first": sample_at_first,
        "sample_sampler": sample_sampler,
        "message": message,
    }


def _cli_arg_overrides(extra_args: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = {
        "--sample_prompts": "sample_prompts",
        "--sample_every_n_epochs": "sample_every_n_epochs",
        "--sample_every_n_steps": "sample_every_n_steps",
        "--sample_sampler": "sample_sampler",
    }
    for idx, arg in enumerate(extra_args):
        if arg == "--sample_at_first":
            out["sample_at_first"] = True
            continue
        if arg in keys and idx + 1 < len(extra_args):
            out[keys[arg]] = extra_args[idx + 1]
            continue
        for cli_key, config_key in keys.items():
            prefix = cli_key + "="
            if arg.startswith(prefix):
                out[config_key] = arg.split("=", 1)[1]
                break
    return out


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


def _resolve_display_path(value: str) -> Path | None:
    raw = str(value or "").replace("\\", "/").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (ROOT / path).resolve()


def _display_project_path(value: str) -> str:
    raw = str(value or "").replace("\\", "/").strip()
    if not raw:
        return ""
    path = Path(raw)
    if not path.is_absolute():
        return path.as_posix().strip("/")
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return raw


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


def _live_metric_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _int_or_none(item.get("step")),
        _int_or_none(item.get("epoch")),
        round(_float_or_none(item.get("loss")) or 0.0, 8) if _float_or_none(item.get("loss")) is not None else None,
        round(_float_or_none(item.get("lr")) or 0.0, 12) if _float_or_none(item.get("lr")) is not None else None,
        round(_float_or_none(item.get("cmmd")) or 0.0, 8) if _float_or_none(item.get("cmmd")) is not None else None,
        str(item.get("kind") or ""),
    )


def _progress_event_key(event: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(event.get("ev") or ""),
        event.get("ts"),
        event.get("global_step"),
        event.get("epoch"),
        event.get("val_step"),
        event.get("path"),
        event.get("status"),
        event.get("final_step"),
    )


def _progress_event_wall_ts(event: dict[str, Any], task_dir: Path | None) -> float:
    rel_ts = _float_or_none(event.get("ts"))
    started_at = None
    if task_dir is not None:
        meta = _read_json(task_dir / "meta.json")
        if isinstance(meta, dict):
            started_at = _float_or_none(meta.get("started_at"))
    if rel_ts is not None and started_at is not None:
        return started_at + rel_ts
    if rel_ts is not None and rel_ts > 1_000_000_000:
        return rel_ts
    return time.time()


def _metric_from_progress_jsonl_event(event: dict[str, Any], ts: float) -> dict[str, Any] | None:
    metric: dict[str, Any] = {"ts": ts}
    step = _int_or_none(event.get("global_step"))
    if step is not None:
        metric["step"] = step
    epoch = _int_or_none(event.get("epoch"))
    if epoch is not None:
        metric["epoch"] = epoch

    if str(event.get("ev") or "") == "val":
        metric["kind"] = "val"
        cmmd = _float_or_none(event.get("cmmd"))
        if cmmd is not None:
            metric["cmmd"] = cmmd
            metric["loss"] = cmmd
        val_step = _int_or_none(event.get("val_step"))
        if val_step is not None:
            metric["val_step"] = val_step
    else:
        loss = _float_or_none(event.get("loss"))
        if loss is not None:
            metric["loss"] = loss
        lr = _float_or_none(event.get("lr"))
        if lr is not None:
            metric["lr"] = lr

    return metric if any(key in metric for key in ("loss", "lr", "cmmd")) else None


def classify_training_error(text: str) -> str:
    """Return a short user-facing hint for known high-signal training failures."""
    if text and CUDA_OOM_RE.search(text):
        return OOM_HINT
    return ""


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
    for name in names:
        match = re.search(rf"{re.escape(name)}[=:/\s]+([\d.eE\-+]+)", text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None
