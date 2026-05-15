"""Training subprocess management and output parsing."""

from __future__ import annotations

import asyncio
from collections import deque
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import psutil
from aiohttp import web

from library.env import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_READ_SIZE = 4096
MAX_LOG_RECORDS = 3000

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
    ):
        if self.status == "running":
            raise RuntimeError("训练已在运行中")

        venv_python = str(ROOT / ".venv" / "bin" / "python")
        if not Path(venv_python).exists():
            venv_python = sys.executable

        cmd = [
            venv_python, "-m", "accelerate.commands.accelerate_cli",
            "launch", "--num_cpu_threads_per_process", "3",
            "--mixed_precision", "bf16",
            str(ROOT / "train.py"),
            "--method", variant,
            "--preset", preset,
            "--methods_subdir", methods_subdir,
        ]
        if extra_args:
            cmd.extend(extra_args)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PATH"] = str(ROOT / ".venv" / "bin") + ":" + env.get("PATH", "")

        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=str(ROOT),
            start_new_session=True,
        )
        self.status = "running"
        self.current_variant = variant
        self.current_preset = preset
        self.current_methods_subdir = methods_subdir
        self._anchor = None
        self._metrics_history = []
        self._last_output_at = time.time()
        self._last_log_line = ""
        self._log_records.clear()
        self._next_log_id = 1

        self._remember_log("status", f"训练命令: {' '.join(cmd)}")

        await self._broadcast({"type": "status", "state": "running",
                               "message": f"训练启动: {methods_subdir}/{variant} / {preset}"})
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
        self.status = "idle"
        await self._broadcast({"type": "status", "state": "idle", "message": "训练已停止"})

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

    def get_status_snapshot(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "variant": self.current_variant,
            "preset": self.current_preset,
            "methods_subdir": self.current_methods_subdir,
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
        self.status = "idle"
        state = "idle" if rc == 0 else "error"
        msg = "训练完成" if rc == 0 else f"训练异常退出 (code={rc})"
        self._remember_log("status", msg)
        await self._broadcast({"type": "status", "state": state, "message": msg})

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
                await self._broadcast({"type": "metrics", **metrics})
            return

        self._last_log_line = text
        record = self._remember_log("log", text, ts=now)
        await self._broadcast({"type": "log", **record})
        metrics = self._extract_metrics_from_log(text)
        if metrics:
            self._metrics_history.append(metrics)
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
        if kind != "progress":
            self._last_log_line = line
        return record

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
