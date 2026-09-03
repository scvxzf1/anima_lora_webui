"""Delegated training service methods.

This module is a mechanical extraction from ``web.services.training_service``.
The public ``TrainingService`` class keeps the same method names and delegates
here so HTTP routes and WebSocket payloads remain unchanged.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

from web.services.training.anomalies import (
    _message_with_error_hint,
    classify_training_error,
    format_training_anomaly,
)
from web.services.training.common import _float_or_none, _format_ts, _int_or_none
from web.services.training.constants import (
    OUTPUT_READ_SIZE,
    SYSTEM_MONITOR_INTERVAL_SECONDS,
    TQDM_RE,
)
from web.services.training.history_timeline import _step_rate_text_from_sample
from web.services.training.live_utils import (
    _clean_output_record,
    _extract_float_metric,
    _first_record_separator,
    _json_safe_training_payload,
    _live_metric_key,
    _metric_from_progress_jsonl_event,
    _progress_event_key,
    _progress_event_wall_ts,
)
from web.services.training.task_lifecycle import cancel_run_tasks, track_run_task

def get_status_snapshot(self) -> dict[str, Any]:
    last_log_id = self._log_records[-1]["id"] if self._log_records else 0
    snapshot = {
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
        "last_log_id": last_log_id,
        "log_count": self._current_history_log_count,
        "metric_count": len(self._metrics_history),
        "latest_progress": dict(self._latest_progress or {}),
        "latest_metric": dict(self._metrics_history[-1]) if self._metrics_history else {},
        "latest_system": dict(self._latest_system_stats or {}),
        "error_hint": self._detected_error_hint,
        "queue_paused": self._queue_paused,
        "queue_count": sum(1 for item in self._queue_items() if item.get("state") == "queued"),
        "queue_item_id": self._current_queue_item_id,
        "gpu_whitelist": self.current_gpu_whitelist,
    }

    # 检测并附加格式化的训练异常提示
    anomaly_message = format_training_anomaly(snapshot)
    if anomaly_message:
        snapshot["anomaly_message"] = anomaly_message

    return _json_safe_training_payload(snapshot)

async def _read_output(self, process=None, generation: int | None = None):
    process = process or self.process
    generation = self._run_generation if generation is None else generation
    assert process and process.stdout
    try:
        buffer = ""
        while True:
            raw = await process.stdout.read(OUTPUT_READ_SIZE)
            if not raw:
                break
            decoded = raw.decode("utf-8", errors="replace")
            self._write_terminal(decoded)
            buffer += decoded
            buffer = await self._drain_output_buffer(buffer)
        if buffer.strip():
            await self._handle_output_record(buffer)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if generation == self._run_generation:
            self._remember_log("error", f"读取训练输出失败: {exc}")

    rc = await process.wait()
    if generation != self._run_generation or self.process is not process:
        return
    job = self.current_job
    stop_requested = self._stop_requested
    pending_train = self._pending_train_after_preprocess
    queue_item_id = self._current_queue_item_id
    await self._ingest_progress_jsonl(final=True)
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
    start_pending = (
        job == "preprocess"
        and rc == 0
        and not stop_requested
        and pending_train is not None
    )
    if queue_item_id and not start_pending:
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
            failed_item = self._find_queue_item(queue_item_id)
            self._maybe_auto_retry(failed_item, reason="process_exit")
        else:
            self._update_queue_item(queue_item_id, {
                "state": "done",
                "message": msg,
                "finished_at": time.time(),
                "finished_at_text": _format_ts(time.time()),
            })
        self._save_queue()
        await self._broadcast_queue()
    await cancel_run_tasks(
        self,
        generation,
        exclude=asyncio.current_task(),
    )
    if generation != self._run_generation or self.process is not process:
        return
    stop_in_progress = self._stopping or self._stop_requested
    self.status = "idle"
    self.current_job = ""
    self._pending_train_after_preprocess = None
    self._current_queue_item_id = ""
    if stop_in_progress:
        return
    self._stop_requested = False
    if start_pending:
        track_run_task(
            self,
            self._start_pending_training(pending_train),
            generation=generation,
        )
        return
    self._schedule_queue_dispatch()

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
        # Structured progress uses the resumed global step, while tqdm starts
        # this process at zero. Keep tqdm only as a log once JSONL is active.
        if self._progress_total_steps is not None:
            self._remember_log("progress", text, ts=now)
            return
        cur = int(m.group("cur"))
        tot = int(m.group("tot"))
        label = m.group("label").strip() or "Training"
        rate_str = self._compute_rate(cur, tot)
        await self._broadcast_progress({
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

def _reset_progress_rate_state(self) -> None:
    self._anchor = None
    self._stdout_rate_last = None
    self._stdout_rate_samples.clear()
    self._structured_rate_last = None
    self._structured_rate_samples.clear()

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

async def _tail_progress_jsonl(self, generation: int | None = None) -> None:
    generation = self._run_generation if generation is None else generation
    while (
        generation == self._run_generation
        and self.status == "running"
        and self._progress_jsonl_path
    ):
        await self._ingest_progress_jsonl()
        await asyncio.sleep(1.0)
    if generation == self._run_generation:
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
            await self._broadcast_progress({
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
        step = _int_or_none(event.get("global_step"))
        rate_str = self._compute_structured_rate(step, ts) if ev == "step" and step is not None else ""
        metric = _metric_from_progress_jsonl_event(event, ts, rate=rate_str)
        if metric:
            await self._record_metric(metric)
        total = self._progress_total_steps
        if ev == "step" and step is not None and total:
            await self._broadcast_progress({
                "current": step,
                "total": total,
                "label": "Training",
                "rate": rate_str,
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
    if self.current_task_dir:
        self._current_history_log_count += 1
    if kind != "progress":
        self._last_log_line = line
    return record

def _compute_rate(self, cur: int, tot: int) -> str:
    del tot
    rate, last = _step_rate_text_from_sample(
        self._stdout_rate_last,
        self._stdout_rate_samples,
        cur,
        time.monotonic(),
    )
    self._stdout_rate_last = last
    return rate

def _compute_structured_rate(self, step: int, ts: float) -> str:
    rate, last = _step_rate_text_from_sample(
        self._structured_rate_last,
        self._structured_rate_samples,
        step,
        ts,
    )
    self._structured_rate_last = last
    return rate

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
        loss = _extract_float_metric(line, ("avr_loss", "loss"))
        if loss is not None:
            metrics["loss"] = loss
            found = True
    if "cmmd" in lower or "val_" in lower:
        metric_number = r"([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?|[+\-]?nan|[+\-]?inf(?:inity)?)"
        for m in re.finditer(rf"(?:cmmd|val_[\w/]+)[=:/\s]+{metric_number}", line, re.IGNORECASE):
            try:
                metrics["cmmd"] = float(m.group(1))
                metrics["kind"] = "val"
                found = True
            except ValueError:
                pass
            break
    if "lr" in lower or "learning_rate" in lower:
        lr = _extract_float_metric(line, ("lr", "learning_rate"))
        if lr is not None:
            metrics["lr"] = lr
            found = True
    if "step" in lower:
        for m in re.finditer(r"step[=:/\s]+(\d+)", line, re.IGNORECASE):
            metrics["step"] = int(m.group(1))
            break
    return metrics if found else None

async def _monitor_system(
    self,
    generation: int | None = None,
    gpu_whitelist: tuple[int, ...] | None = None,
):
    generation = self._run_generation if generation is None else generation
    gpu_whitelist = tuple(self.current_gpu_whitelist) if gpu_whitelist is None else gpu_whitelist
    # Sample immediately on launch, then keep a short cadence so the
    # dashboard "资源与活动" panel tracks preprocess/training GPU load.
    while generation == self._run_generation and self.status == "running":
        from web.services.training.gpu_async import get_gpu_stats

        stats = await get_gpu_stats(list(gpu_whitelist))
        if generation != self._run_generation or self.status != "running":
            return
        if stats:
            stats["last_output_at"] = self._last_output_at
            stats["ts"] = time.time()
            self._latest_system_stats = dict(stats)
            self._append_history_jsonl("system.jsonl", stats)
            await self._broadcast({"type": "system", **stats})
        await asyncio.sleep(SYSTEM_MONITOR_INTERVAL_SECONDS)

async def _broadcast_progress(self, msg: dict[str, Any]) -> None:
    payload = {"type": "progress", **msg}
    self._latest_progress = dict(payload)
    await self._broadcast(payload)

async def _broadcast(self, msg: dict):
    import json
    data = json.dumps(_json_safe_training_payload(msg), ensure_ascii=False)
    dead = set()
    for ws in self._ws_clients:
        try:
            await ws.send_str(data)
        except (ConnectionResetError, RuntimeError):
            dead.add(ws)
    self._ws_clients -= dead
