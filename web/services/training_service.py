"""Training subprocess management and output parsing."""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import psutil
from aiohttp import web

from library.env import load_dotenv
from library.runtime.launch import accelerate_training_command_prefix
from web.services.config_service import apply_auto_data_dirs, load_merged_config, preflight_training_config

ROOT = Path(__file__).resolve().parents[2]
HISTORY_DIR = ROOT / "configs" / "web-training-history"
OUTPUT_READ_SIZE = 4096
MAX_LOG_RECORDS = 3000
MAX_HISTORY_ITEMS = 100

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
        self._log_records: deque[dict[str, Any]] = deque(maxlen=MAX_LOG_RECORDS)
        self._next_log_id = 1

    async def start(
        self,
        variant: str,
        preset: str,
        extra_args: list[str] | None = None,
        methods_subdir: str = "gui-methods",
        *,
        reset_logs: bool = True,
    ):
        if self.status == "running":
            raise RuntimeError("已有任务在运行中")

        venv_python = str(ROOT / ".venv" / "bin" / "python")
        if not Path(venv_python).exists():
            venv_python = sys.executable

        env = os.environ.copy()
        cmd = [
            *accelerate_training_command_prefix(venv_python, ROOT / "train.py", env),
            "--method", variant,
            "--preset", preset,
            "--methods_subdir", methods_subdir,
        ]
        if extra_args:
            cmd.extend(extra_args)

        env["PYTHONUNBUFFERED"] = "1"
        env["PATH"] = str(ROOT / ".venv" / "bin") + ":" + env.get("PATH", "")

        output_dir, sample_dir, sample_config = _resolve_training_runtime_info(
            variant,
            preset,
            methods_subdir,
            extra_args or [],
        )
        data_dirs = _ensure_training_data_dirs(variant, preset, methods_subdir)
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
            start_message=f"训练启动: {methods_subdir}/{variant} / {preset}",
            command_label="训练命令",
            reset_logs=reset_logs,
        )

    async def start_preprocess(
        self,
        variant: str,
        preset: str,
        methods_subdir: str = "gui-methods",
        extra_args: list[str] | None = None,
        train_after: bool = False,
    ):
        if self.status == "running":
            raise RuntimeError("已有任务在运行中")

        venv_python = str(ROOT / ".venv" / "bin" / "python")
        if not Path(venv_python).exists():
            venv_python = sys.executable

        cmd = [venv_python, "tasks.py", "preprocess"]
        if extra_args:
            cmd.extend(extra_args)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PATH"] = str(ROOT / ".venv" / "bin") + ":" + env.get("PATH", "")
        env["METHOD"] = variant
        env["METHODS_SUBDIR"] = methods_subdir
        env["PRESET"] = preset

        output_dir, sample_dir, sample_config = _resolve_training_runtime_info(
            variant,
            preset,
            methods_subdir,
            [],
        )
        data_dirs = _ensure_training_data_dirs(variant, preset, methods_subdir)
        self._pending_train_after_preprocess = {
            "variant": variant,
            "preset": preset,
            "methods_subdir": methods_subdir,
            "extra_args": [],
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
    ):
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=str(ROOT),
            start_new_session=True,
        )
        self.status = "running"
        self.current_job = job
        self.current_variant = variant
        self.current_preset = preset
        self.current_methods_subdir = methods_subdir
        self.current_output_dir = output_dir
        self.current_sample_dir = sample_dir
        self.current_sample_config = sample_config
        self._anchor = None
        self._metrics_history = []
        self._stop_requested = False
        if job != "preprocess":
            self._pending_train_after_preprocess = None
        self._last_output_at = time.time()
        self._last_log_line = ""
        if reset_logs:
            self._log_records.clear()
            self._next_log_id = 1

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
        )
        self._remember_log("status", f"{command_label}: {' '.join(cmd)}")

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
            "task_id": self.current_task_id,
        })
        asyncio.create_task(self._read_output())
        asyncio.create_task(self._monitor_system())

    async def stop(self):
        if not self.process or self.process.returncode is not None:
            self.status = "idle"
            return
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
            "task_id": self.current_task_id,
        })

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

    def list_history_tasks(self) -> list[dict[str, Any]]:
        return _list_history_tasks()

    def get_history_task(self, task_id: str) -> dict[str, Any]:
        return _load_history_task(task_id)

    def update_history_task(self, task_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        return _update_history_task(task_id, patch)

    def delete_history_task(self, task_id: str) -> dict[str, Any]:
        if task_id == self.current_task_id and self.status == "running":
            raise RuntimeError("当前运行中的任务不能删除")
        return _delete_history_task(task_id)

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
            "task_id": self.current_task_id,
            "last_output_at": self._last_output_at,
            "last_log_line": self._last_log_line,
            "last_log_id": self._log_records[-1]["id"] if self._log_records else 0,
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
        self.status = "idle"
        self.current_job = ""
        self._stop_requested = False
        self._pending_train_after_preprocess = None
        state = "idle" if rc == 0 or stop_requested else "error"
        if stop_requested and job == "preprocess":
            msg = "预处理已停止"
        elif stop_requested:
            msg = "训练已停止"
        elif job == "preprocess":
            msg = "预处理完成" if rc == 0 else f"预处理异常退出 (code={rc})"
        else:
            msg = "训练完成" if rc == 0 else f"训练异常退出 (code={rc})"
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
            "task_id": self.current_task_id,
        })
        if (
            job == "preprocess"
            and rc == 0
            and not stop_requested
            and pending_train is not None
        ):
            await self._start_pending_training(pending_train)

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
            )
        except Exception as e:
            msg = f"自动开始训练失败: {e}"
            self._remember_log("status", msg)
            await self._broadcast({
                "type": "status",
                "state": "error",
                "job": "training",
                "message": msg,
                "output_dir": self.current_output_dir,
                "sample_dir": self.current_sample_dir,
                "sample_config": self.current_sample_config,
                "task_id": self.current_task_id,
            })

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
                self._metrics_history.append(metrics)
                self._append_history_jsonl("metrics.jsonl", metrics)
                await self._broadcast({"type": "metrics", **metrics})
            return

        self._last_log_line = text
        record = self._remember_log("log", text, ts=now)
        await self._broadcast({"type": "log", **record})
        metrics = self._extract_metrics_from_log(text)
        if metrics:
            self._metrics_history.append(metrics)
            self._append_history_jsonl("metrics.jsonl", metrics)
            await self._broadcast({"type": "metrics", **metrics})

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
    ) -> None:
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
        now = time.time()
        meta = {
            "id": task_dir.name,
            "name": "",
            "group": "",
            "archived": False,
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
            "started_at": now,
            "started_at_text": _format_ts(now),
            "finished_at": None,
            "finished_at_text": "",
            "message": "",
            "returncode": None,
            "log_count": 0,
            "metric_count": 0,
        }
        _write_json(task_dir / "meta.json", meta)
        _write_config_snapshot(task_dir / "config.snapshot.toml", variant, preset, methods_subdir)

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
        _write_json(self.current_task_dir / "meta.json", meta)

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
        if "loss" in line.lower():
            for m in re.finditer(r"(?:avr_)?loss[=:/\s]+([\d.eE\-+]+)", line, re.IGNORECASE):
                metrics["loss"] = float(m.group(1))
                found = True
                break
        if "lr" in line.lower():
            for m in re.finditer(r"lr[=:/\s]+([\d.eE\-+]+)", line, re.IGNORECASE):
                try:
                    metrics["lr"] = float(m.group(1))
                    found = True
                except ValueError:
                    pass
                break
        if "step" in line.lower():
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


def _resolve_training_runtime_info(
    variant: str,
    preset: str,
    methods_subdir: str,
    extra_args: list[str],
) -> tuple[str, str, dict[str, Any]]:
    output_dir = "output/ckpt"
    cfg: dict[str, Any] = {}
    try:
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


def _ensure_training_data_dirs(variant: str, preset: str, methods_subdir: str) -> dict[str, str]:
    cfg = apply_auto_data_dirs(load_merged_config(variant, preset, methods_subdir), create=True)
    return {
        "source_image_dir": str(cfg.get("source_image_dir") or ""),
        "resized_image_dir": str(cfg.get("resized_image_dir") or ""),
        "lora_cache_dir": str(cfg.get("lora_cache_dir") or ""),
    }


def _write_config_snapshot(path: Path, variant: str, preset: str, methods_subdir: str) -> None:
    try:
        cfg = apply_auto_data_dirs(load_merged_config(variant, preset, methods_subdir))
        path.write_text(toml_dumps_sorted(cfg), encoding="utf-8")
    except Exception as e:
        path.write_text(f"# 无法生成配置快照: {e}\n", encoding="utf-8")


def toml_dumps_sorted(data: dict[str, Any]) -> str:
    try:
        import toml
        return toml.dumps({key: data[key] for key in sorted(data)})
    except Exception:
        return json.dumps(data, ensure_ascii=False, indent=2)


def _list_history_tasks() -> list[dict[str, Any]]:
    if not HISTORY_DIR.exists():
        return []
    tasks = []
    for meta_path in HISTORY_DIR.glob("*/meta.json"):
        meta = _read_json(meta_path)
        if meta:
            tasks.append(_history_summary(meta, meta_path.parent))
    tasks.sort(key=lambda item: item.get("started_at") or 0, reverse=True)
    return tasks[:MAX_HISTORY_ITEMS]


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
    if not task_dir.exists():
        raise FileNotFoundError("任务不存在")
    meta = _read_json(task_dir / "meta.json")
    if not meta:
        raise FileNotFoundError("任务元信息不存在")
    return {
        "ok": True,
        "task": _history_summary(meta, task_dir),
        "logs": _read_jsonl(task_dir / "logs.jsonl"),
        "metrics": _read_jsonl(task_dir / "metrics.jsonl"),
        "system": _read_jsonl(task_dir / "system.jsonl"),
        "config_toml": (task_dir / "config.snapshot.toml").read_text(encoding="utf-8")
            if (task_dir / "config.snapshot.toml").exists()
            else "",
    }


def _update_history_task(task_id: str, patch: dict[str, Any]) -> dict[str, Any]:
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
    _write_json(meta_path, meta)
    return {"ok": True, "task": _history_summary(meta, task_dir)}


def _delete_history_task(task_id: str) -> dict[str, Any]:
    task_dir = _history_task_dir(task_id)
    if not task_dir.exists():
        raise FileNotFoundError("任务不存在")
    for child in task_dir.iterdir():
        if child.is_dir():
            raise ValueError("任务目录包含子目录，已拒绝删除")
        child.unlink()
    task_dir.rmdir()
    return {"ok": True, "message": "任务已删除"}


def _history_summary(meta: dict[str, Any], task_dir: Path) -> dict[str, Any]:
    out = dict(meta)
    out["id"] = task_dir.name
    out["name"] = str(out.get("name") or "")
    out["group"] = str(out.get("group") or "")
    out["archived"] = bool(out.get("archived", False))
    out["history_dir"] = _display_project_path(str(task_dir))
    out["history_dir_abs"] = str(task_dir)
    out["config_snapshot"] = _display_project_path(str(task_dir / "config.snapshot.toml"))
    out["logs_path"] = _display_project_path(str(task_dir / "logs.jsonl"))
    out["metrics_path"] = _display_project_path(str(task_dir / "metrics.jsonl"))
    out["system_path"] = _display_project_path(str(task_dir / "system.jsonl"))
    data_dirs = out.get("data_dirs") if isinstance(out.get("data_dirs"), dict) else {}
    for key in ("source_image_dir", "resized_image_dir", "lora_cache_dir"):
        out[key] = str(out.get(key) or data_dirs.get(key) or "")
    out["log_count"] = int(out.get("log_count") or _count_jsonl(task_dir / "logs.jsonl"))
    out["metric_count"] = int(out.get("metric_count") or _count_jsonl(task_dir / "metrics.jsonl"))
    return out


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                out.append(value)
        except Exception:
            continue
    return out


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())
    except Exception:
        return 0


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
