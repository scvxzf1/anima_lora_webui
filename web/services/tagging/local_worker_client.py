"""Async controller for one short-lived local tagging worker process."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from web.services.config_service import ROOT
from web.services.project_python import resolve_web_python_executable

from .local_worker_protocol import (
    MAX_EVENT_BYTES,
    PROTOCOL_VERSION,
    decode_event,
    encode_event,
    safe_error,
)
from .providers.base import LocalTaggingError

WORKER_STOP_TIMEOUT_SECONDS = 3.0
MAX_STDERR_CHARS = 16_000
_LOCAL_SETTING_KEYS = {
    "provider",
    "asset_id",
    "model_id",
    "device",
    "gpu_index",
    "batch_size",
    "general_threshold",
    "character_threshold",
    "blacklist",
    "add_copyright_tag",
    "add_artist_tag",
    "add_meta_tag",
    "add_model_tag",
    "add_rating_tag",
    "add_quality_tag",
}


class LocalWorkerError(LocalTaggingError):
    def __init__(self, message: str, *, phase: str = "inference"):
        super().__init__(message)
        self.phase = phase if phase in {"protocol", "initialization", "inference"} else "inference"


def selected_gpu_index(settings: Mapping[str, Any]) -> int | None:
    if str(settings.get("device") or "auto").strip().lower() != "cuda":
        return None
    value = settings.get("gpu_index")
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    return index if index >= 0 else None


def build_worker_environment(
    settings: Mapping[str, Any],
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    env["PYTHONUNBUFFERED"] = "1"
    index = selected_gpu_index(settings)
    if index is not None:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(index)
    return env


def _worker_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    return {key: settings[key] for key in _LOCAL_SETTING_KEYS if key in settings}


class LocalTaggingWorkerClient:
    def __init__(
        self,
        *,
        provider: str,
        settings: Mapping[str, Any],
        job_id: str = "",
        command: Sequence[str] | None = None,
        root: Path | None = None,
    ):
        self.provider = str(provider or "").strip().lower()
        self.settings = dict(settings)
        self.job_id = str(job_id or "")[:64]
        self.command = list(command) if command is not None else None
        self.root = Path(root or ROOT).resolve()
        self.runtime_warning = ""
        self.last_progress = (0, 0)
        self.last_pid: int | None = None
        self.last_returncode: int | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._stop_lock = asyncio.Lock()
        self._running = False

    @property
    def pid(self) -> int | None:
        process = self._process
        return process.pid if process is not None and process.returncode is None else None

    async def run(self, paths: list[Path]) -> list[dict[str, Any]]:
        if self._running:
            raise RuntimeError("本地打标 Worker 客户端不能并发复用")
        self._running = True
        stderr_task: asyncio.Task[str] | None = None
        process: asyncio.subprocess.Process | None = None
        try:
            process = await self._spawn()
            self._process = process
            self.last_pid = process.pid
            if process.stdout is None or process.stdin is None or process.stderr is None:
                raise LocalWorkerError("无法建立本地打标 Worker 通信管道", phase="initialization")
            stderr_task = asyncio.create_task(_capture_stderr(process.stderr))
            request = {
                "version": PROTOCOL_VERSION,
                "job_id": self.job_id,
                "provider": self.provider,
                "settings": _worker_settings(self.settings),
                "images": [str(Path(path)) for path in paths],
            }
            process.stdin.write(f"{encode_event(request)}\n".encode("utf-8"))
            await process.stdin.drain()
            process.stdin.close()
            await process.stdin.wait_closed()
            results, terminal_error = await self._read_events(process.stdout, expected_total=len(paths))
            returncode = await process.wait()
            self.last_returncode = returncode
            stderr = await stderr_task
            stderr_task = None
            if terminal_error is not None:
                message, phase = terminal_error
                raise LocalWorkerError(message, phase=phase)
            if returncode != 0:
                detail = safe_error(stderr) if stderr else f"退出码 {returncode}"
                raise LocalWorkerError(f"本地打标 Worker 异常退出：{detail}")
            return results
        except asyncio.CancelledError:
            await asyncio.shield(self.stop())
            raise
        except LocalWorkerError:
            await self.stop()
            raise
        except (OSError, ValueError, RuntimeError) as exc:
            await self.stop()
            raise LocalWorkerError(safe_error(exc), phase="initialization" if process is None else "protocol") from exc
        finally:
            if process is not None and process.returncode is None:
                await self.stop()
            if stderr_task is not None:
                try:
                    await stderr_task
                except asyncio.CancelledError:
                    pass
            if process is not None:
                self.last_returncode = process.returncode
            if self._process is process:
                self._process = None
            self._running = False

    async def stop(self) -> None:
        async with self._stop_lock:
            process = self._process
            if process is None or process.returncode is not None:
                return
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(
                    asyncio.shield(process.wait()),
                    timeout=WORKER_STOP_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
            self.last_returncode = process.returncode

    async def _spawn(self) -> asyncio.subprocess.Process:
        spawn_task = asyncio.create_task(self._create_process())
        try:
            return await asyncio.shield(spawn_task)
        except asyncio.CancelledError as canceled:
            # Process creation itself is not cancel-safe.  Retrieve a process
            # that won the race, then reap it before propagating cancellation.
            try:
                process = await asyncio.shield(spawn_task)
            except Exception:  # noqa: BLE001 - cancellation remains authoritative
                raise canceled
            self._process = process
            self.last_pid = process.pid
            await asyncio.shield(self.stop())
            if self._process is process:
                self._process = None
            raise canceled

    async def _create_process(self) -> asyncio.subprocess.Process:
        command = self.command or [
            resolve_web_python_executable(self.root),
            "-m",
            "web.services.tagging.local_worker",
        ]
        try:
            return await asyncio.create_subprocess_exec(
                *command,
                cwd=str(self.root),
                env=build_worker_environment(self.settings),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=MAX_EVENT_BYTES + 1,
            )
        except (OSError, ValueError) as exc:
            raise LocalWorkerError(
                f"无法启动本地打标 Worker：{safe_error(exc)}",
                phase="initialization",
            ) from exc

    async def _read_events(
        self,
        stream: asyncio.StreamReader,
        *,
        expected_total: int,
    ) -> tuple[list[dict[str, Any]], tuple[str, str] | None]:
        results: list[dict[str, Any]] = []
        expected_sequence = 1
        terminal_error: tuple[str, str] | None = None
        completed = False
        ready = False
        while raw := await stream.readline():
            event = decode_event(raw)
            if str(event.get("job_id") or "") != self.job_id:
                raise LocalWorkerError("本地打标 Worker 返回了错误的任务 ID", phase="protocol")
            if event.get("seq") != expected_sequence:
                raise LocalWorkerError("本地打标 Worker 事件序号不连续", phase="protocol")
            expected_sequence += 1
            event_type = str(event.get("type") or "")
            if completed or terminal_error is not None:
                raise LocalWorkerError("本地打标 Worker 在终态后继续输出事件", phase="protocol")
            if event_type == "ready":
                if ready or expected_sequence != 2:
                    raise LocalWorkerError("本地打标 Worker ready 事件顺序无效", phase="protocol")
                ready = True
                self.runtime_warning = str(event.get("runtime_warning") or "")[:500]
            elif event_type == "progress":
                if not ready:
                    raise LocalWorkerError("本地打标 Worker 在 ready 前返回进度", phase="protocol")
                done = _nonnegative_int(event.get("done"))
                total = _nonnegative_int(event.get("total"))
                if total != expected_total or done > total:
                    raise LocalWorkerError("本地打标 Worker 进度范围无效", phase="protocol")
                self.last_progress = (done, total)
            elif event_type == "result":
                if not ready:
                    raise LocalWorkerError("本地打标 Worker 在 ready 前返回结果", phase="protocol")
                result = event.get("result")
                if not isinstance(result, dict):
                    raise LocalWorkerError("本地打标 Worker 返回了无效结果", phase="protocol")
                if event.get("index") != len(results) or len(results) >= expected_total:
                    raise LocalWorkerError("本地打标 Worker 结果序号无效", phase="protocol")
                results.append(dict(result))
            elif event_type == "error":
                terminal_error = (
                    safe_error(event.get("error")),
                    str(event.get("phase") or "inference"),
                )
            elif event_type == "complete":
                if not ready:
                    raise LocalWorkerError("本地打标 Worker 在 ready 前完成", phase="protocol")
                if _nonnegative_int(event.get("total")) != expected_total:
                    raise LocalWorkerError("本地打标 Worker 完成数量无效", phase="protocol")
                if _nonnegative_int(event.get("results")) != len(results):
                    raise LocalWorkerError("本地打标 Worker 结果数量不一致", phase="protocol")
                completed = True
            else:
                raise LocalWorkerError(f"未知的本地打标 Worker 事件：{event_type or '空值'}", phase="protocol")
        if terminal_error is None and not completed:
            raise LocalWorkerError("本地打标 Worker 未返回完成事件", phase="protocol")
        return results, terminal_error


async def _capture_stderr(stream: asyncio.StreamReader) -> str:
    chunks: list[str] = []
    size = 0
    while raw := await stream.readline():
        text = raw.decode("utf-8", errors="replace")
        chunks.append(text)
        size += len(text)
        while size > MAX_STDERR_CHARS and chunks:
            size -= len(chunks.pop(0))
    return "".join(chunks)[-MAX_STDERR_CHARS:]


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


__all__ = [
    "LocalTaggingWorkerClient",
    "LocalWorkerError",
    "build_worker_environment",
    "selected_gpu_index",
]
