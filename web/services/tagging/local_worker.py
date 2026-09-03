"""Short-lived subprocess entry point for local ONNX tagging jobs."""

from __future__ import annotations

import ctypes
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any, TextIO

from .local_worker_protocol import PROTOCOL_VERSION, encode_event, safe_error, sanitize_result


class _EventWriter:
    def __init__(self, stream: TextIO):
        self.stream = stream
        self.job_id = ""
        self.sequence = 0

    def emit(self, event_type: str, **payload: Any) -> None:
        self.sequence += 1
        event = {
            "version": PROTOCOL_VERSION,
            "job_id": self.job_id,
            "seq": self.sequence,
            "type": event_type,
            **payload,
        }
        self.stream.write(f"{encode_event(event)}\n")
        self.stream.flush()


def _protocol_stream() -> TextIO:
    """Reserve the original stdout pipe, then send incidental prints to stderr."""

    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    return os.fdopen(protocol_fd, "w", encoding="utf-8", buffering=1)


def _arm_parent_death_signal() -> None:
    """Ensure Linux workers do not survive an abruptly terminated Web process."""

    if not sys.platform.startswith("linux"):
        return
    try:
        libc = ctypes.CDLL(None)
        if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
            return
        if os.getppid() == 1:
            os.kill(os.getpid(), signal.SIGTERM)
    except (AttributeError, OSError):
        return


def _read_request() -> dict[str, Any]:
    raw = sys.stdin.buffer.readline()
    if not raw:
        raise ValueError("本地打标 Worker 未收到任务参数")
    try:
        request = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("本地打标 Worker 任务参数不是有效 JSON") from exc
    if not isinstance(request, dict) or request.get("version") != PROTOCOL_VERSION:
        raise ValueError("本地打标 Worker 任务协议无效")
    return request


def _validate_request(request: dict[str, Any]) -> tuple[str, dict[str, Any], list[Path]]:
    provider = str(request.get("provider") or "").strip().lower()
    if provider not in {"wd14", "cltagger"}:
        raise ValueError(f"不支持的本地打标 provider：{provider or '空值'}")
    settings = request.get("settings")
    images = request.get("images")
    if not isinstance(settings, dict) or not isinstance(images, list):
        raise ValueError("本地打标 Worker 任务参数不完整")
    if not images or len(images) > 500:
        raise ValueError("本地打标 Worker 图片数量无效")
    paths = [Path(str(value)) for value in images]
    return provider, settings, paths


def _selected_gpu(settings: dict[str, Any]) -> int | None:
    if str(settings.get("device") or "auto").lower() != "cuda":
        return None
    value = settings.get("gpu_index")
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    return index if index >= 0 else None


def _close_tagger(tagger: Any) -> None:
    close = getattr(tagger, "close", None)
    if callable(close):
        close()
        return
    clear = getattr(tagger, "_clear_session", None)
    if callable(clear):
        clear()


def run() -> int:
    _arm_parent_death_signal()
    with _protocol_stream() as stream:
        writer = _EventWriter(stream)
        tagger: Any = None
        phase = "protocol"
        try:
            request = _read_request()
            writer.job_id = str(request.get("job_id") or "")[:64]
            provider, settings, paths = _validate_request(request)
            phase = "initialization"
            from .providers.factory import get_tagger

            tagger = get_tagger(provider, settings)
            tagger.prepare()
            selected_gpu = _selected_gpu(settings)
            if selected_gpu is not None and not bool(getattr(tagger, "_using_gpu", False)):
                raise RuntimeError(f"所选 GPU {selected_gpu} 未能启用 CUDAExecutionProvider")
            writer.emit(
                "ready",
                provider=provider,
                runtime_warning=str(getattr(tagger, "runtime_warning", "") or "")[:500],
                message="本地模型已就绪",
            )
            phase = "inference"

            def on_progress(done: int, total: int) -> None:
                writer.emit("progress", done=int(done), total=int(total))

            count = 0
            for index, result in enumerate(tagger.tag(paths, on_progress=on_progress)):
                writer.emit("result", index=index, result=sanitize_result(result))
                count += 1
            writer.emit("complete", results=count, total=len(paths))
            return 0
        except BaseException as exc:  # noqa: BLE001 - process boundary normalization
            writer.emit("error", phase=phase, error=safe_error(exc))
            return 1
        finally:
            if tagger is not None:
                try:
                    _close_tagger(tagger)
                except Exception:  # noqa: BLE001 - process exit is authoritative cleanup
                    pass


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
