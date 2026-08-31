"""In-memory tagging job lifecycle and review/write-back orchestration."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .client import OpenAICompatibleClient, TaggingApiError
from .memory_log import DEFAULT_LOG_RETENTION_LINES, TaggingMemoryLog
from .settings import load_settings
from .storage import CaptionWriteConflict, resolve_tagging_image, write_caption

MAX_ITEMS_PER_JOB = 500
MAX_RETAINED_JOBS = 40
SHUTDOWN_TIMEOUT_SECONDS = 2.0


class TaggingJobManager:
    """Own short-lived jobs for one WebUI process.

    Jobs are intentionally not mixed with the training queue.  They contain
    review drafts and may be canceled while an external provider is in flight;
    the browser polls the sanitized snapshots returned by this class.
    """

    def __init__(self, *, log_retention_lines: int = DEFAULT_LOG_RETENTION_LINES):
        self.jobs: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._tasks: dict[str, asyncio.Task] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._commit_locks: dict[str, asyncio.Lock] = {}
        self.logs = TaggingMemoryLog(log_retention_lines)

    async def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        dataset_file = str(payload.get("dataset_file") or payload.get("file") or "").strip()
        if not dataset_file:
            raise ValueError("请先选择数据集预设")
        try:
            dataset_index = int(payload.get("dataset_index", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("数据集序号无效") from exc
        source = "source" if str(payload.get("source") or "source").lower() == "source" else "training"
        prompt = str(payload.get("user_prompt") or payload.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("请输入打标提示词")
        provider_settings = load_settings()
        system_prompt = str(payload.get("system_prompt") or provider_settings.get("system_prompt") or "").strip()
        if not system_prompt:
            raise ValueError("请输入系统提示词")
        if len(system_prompt) > 10_000 or len(prompt) > 10_000:
            raise ValueError("单条提示词最多支持 10000 个字符")
        provider_settings = {**provider_settings, "system_prompt": system_prompt}
        raw_items = payload.get("items")
        if raw_items is None:
            raw_items = payload.get("images")
        if not isinstance(raw_items, list) or not raw_items:
            raise ValueError("请至少选择一张图片")
        if len(raw_items) > MAX_ITEMS_PER_JOB:
            raise ValueError(f"单次最多提交 {MAX_ITEMS_PER_JOB} 张图片")

        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in raw_items:
            item = raw if isinstance(raw, dict) else {"file": raw}
            image_file = str(item.get("file") or item.get("image") or "").strip()
            if not image_file or image_file in seen:
                continue
            seen.add(image_file)
            resolved = await asyncio.to_thread(
                resolve_tagging_image,
                dataset_file,
                dataset_index,
                image_file,
                source=source,
            )
            current = resolved["caption"]
            items.append(
                {
                    "id": uuid.uuid4().hex[:12],
                    "file": resolved["file"],
                    "name": resolved["name"],
                    "url": str(item.get("url") or resolved.get("url") or ""),
                    "thumbnail_url": str(item.get("thumbnail_url") or resolved.get("thumbnail_url") or ""),
                    "state": "queued",
                    "caption": str(current.get("text") or ""),
                    "proposed_caption": "",
                    "error": "",
                    "commit_error": "",
                    "attempts": 0,
                    "elapsed_ms": None,
                    "_path": resolved["path"],
                }
            )
        if not items:
            raise ValueError("没有可提交的有效图片")

        job_id = uuid.uuid4().hex[:16]
        now = time.time()
        job = {
            "id": job_id,
            "state": "queued",
            "created_at": now,
            "started_at": None,
            "finished_at": None,
            "dataset_file": dataset_file,
            "dataset_index": dataset_index,
            "source": source,
            "prompt": prompt[:10000],
            "system_prompt": system_prompt[:10000],
            "total": len(items),
            "completed": 0,
            "failed": 0,
            "canceled": 0,
            "items": items,
            "error": "",
            "settings": _public_job_settings(provider_settings),
            "_provider_settings": provider_settings,
        }
        self.jobs[job_id] = job
        self._cancel_events[job_id] = asyncio.Event()
        self._commit_locks[job_id] = asyncio.Lock()
        self._prune()
        self._log(job, f"任务已加入队列，共 {len(items)} 张图片", event="job_queued")
        self._tasks[job_id] = asyncio.create_task(self._run(job), name=f"tagging:{job_id}")
        return self.snapshot(job_id)

    def list(self) -> list[dict[str, Any]]:
        return [self._summary(job) for job in reversed(self.jobs.values())]

    def snapshot(self, job_id: str) -> dict[str, Any]:
        job = self._get(job_id)
        return {"ok": True, "job": self._public_job(job)}

    def get_logs(self, *, after: Any = 0, limit: Any = None, job_id: str = "") -> dict[str, Any]:
        return self.logs.snapshot(after=after, limit=limit, job_id=job_id)

    def clear_logs(self) -> dict[str, Any]:
        return self.logs.clear()

    def set_log_retention(self, value: Any) -> int:
        return self.logs.set_retention(value)

    async def cancel(self, job_id: str) -> dict[str, Any]:
        job = self._get(job_id)
        if job["state"] not in {"queued", "running"}:
            raise RuntimeError("当前任务已经结束")
        self._cancel_events[job_id].set()
        job["error"] = "用户请求停止"
        self._log(job, "收到停止请求", level="warning", event="job_cancel_requested")
        # Cancel the owning coroutine as well as setting the cooperative flag.
        # The provider client is async, so this interrupts an in-flight HTTP
        # request instead of waiting for its timeout/retry cycle to finish.
        if job["state"] == "queued":
            self._mark_canceled(job)
        task = self._tasks.get(job_id)
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self._tasks.pop(job_id, None)
        elif job["state"] in {"queued", "running"}:
            self._mark_canceled(job)
        return self.snapshot(job_id)

    def update_item(self, job_id: str, item_id: str, text: str) -> dict[str, Any]:
        job = self._get(job_id)
        if job["state"] in {"queued", "running"}:
            raise RuntimeError("任务运行中，暂不能编辑候选 caption")
        item = next((entry for entry in job["items"] if entry["id"] == item_id), None)
        if item is None:
            raise KeyError("找不到打标条目")
        value = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if len(value) > 100_000:
            raise ValueError("caption 过长，最多支持 100000 个字符")
        item["proposed_caption"] = value
        item["error"] = ""
        item["state"] = "ready" if value else "empty"
        self._refresh_counts(job)
        self._log(
            job,
            f"已更新候选标注：{item['name']}",
            event="item_edited",
            item=item,
        )
        return self.snapshot(job_id)

    async def commit(
        self,
        job_id: str,
        *,
        all_items: bool = False,
        item_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        job = self._get(job_id)
        if job["state"] in {"queued", "running"}:
            raise RuntimeError("任务尚未完成，不能写回 caption")
        lock = self._commit_locks.setdefault(job_id, asyncio.Lock())
        async with lock:
            return await self._commit_locked(job, all_items=all_items, item_ids=item_ids)

    async def _commit_locked(
        self,
        job: dict[str, Any],
        *,
        all_items: bool,
        item_ids: list[str] | None,
    ) -> dict[str, Any]:
        wanted = set(str(item_id) for item_id in (item_ids or []))
        candidates = [
            item
            for item in job["items"]
            if (all_items or item["id"] in wanted)
            and str(item.get("proposed_caption") or "").strip()
            and (
                item.get("state") in {"ready", "failed"}
                or (
                    item.get("state") == "committed"
                    and str(item.get("caption") or "") != str(item.get("proposed_caption") or "")
                )
            )
        ]
        if not candidates:
            return {"ok": True, "written": 0, "conflicts": 0, "skipped": 0, "errors": [], "job": self._public_job(job)}
        written = 0
        conflicts = 0
        skipped = 0
        errors: list[dict[str, str]] = []
        for item in candidates:
            try:
                result = await asyncio.to_thread(
                    write_caption,
                    job["dataset_file"],
                    job["dataset_index"],
                    item["file"],
                    item["proposed_caption"],
                    source=job["source"],
                )
            except CaptionWriteConflict as exc:
                conflicts += 1
                item["commit_error"] = str(exc)
                errors.append({"item_id": item["id"], "file": item["file"], "error": str(exc)})
                self._log(job, f"写回冲突：{item['name']} - {exc}", level="warning", event="commit_conflict", item=item)
            except (OSError, ValueError) as exc:
                skipped += 1
                item["commit_error"] = str(exc)
                errors.append({"item_id": item["id"], "file": item["file"], "error": str(exc)})
                self._log(job, f"写回失败：{item['name']} - {exc}", level="error", event="commit_failed", item=item)
            else:
                written += 1
                item["state"] = "committed"
                item["commit_error"] = ""
                item["caption"] = result["text"]
                item["caption_file"] = result["caption_file"]
                self._log(job, f"已写回：{item['name']}", level="success", event="commit_succeeded", item=item)
        self._refresh_counts(job)
        self._log(
            job,
            f"写回完成：成功 {written}，冲突 {conflicts}，跳过 {skipped}",
            level="success" if written and not errors else "warning",
            event="commit_finished",
        )
        return {
            "ok": True,
            "written": written,
            "conflicts": conflicts,
            "skipped": skipped,
            "errors": errors,
            "job": self._public_job(job),
        }

    async def shutdown(self) -> None:
        for event in self._cancel_events.values():
            event.set()
        tasks = list(self._tasks.values())
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=SHUTDOWN_TIMEOUT_SECONDS)
            for task in pending:
                task.cancel()
            # Retrieve exceptions from already-finished tasks too.  Otherwise
            # a provider failure that races with shutdown is reported as an
            # unhandled Task exception by asyncio.
            await asyncio.gather(*(tuple(done) + tuple(pending)), return_exceptions=True)
        self._tasks.clear()

    async def _run(self, job: dict[str, Any]) -> None:
        job_id = job["id"]
        try:
            await self._run_job(job)
        finally:
            self._tasks.pop(job_id, None)

    async def _run_job(self, job: dict[str, Any]) -> None:
        job_id = job["id"]
        cancel_event = self._cancel_events[job_id]
        if cancel_event.is_set():
            self._mark_canceled(job)
            return
        job["state"] = "running"
        job["started_at"] = time.time()
        self._log(job, "任务开始", event="job_started")
        settings = dict(job.get("_provider_settings") or load_settings())
        try:
            client = OpenAICompatibleClient(settings)
        except (ValueError, OSError) as exc:
            job["error"] = str(exc)
            self._log(job, f"外部 API 初始化失败：{exc}", level="error", event="provider_failed")
            for item in job["items"]:
                item["state"] = "failed"
                item["error"] = str(exc)
            self._finish(job, "failed")
            return

        semaphore = asyncio.Semaphore(int(settings.get("concurrency", 2) or 2))

        async def run_item(item: dict[str, Any]) -> None:
            if cancel_event.is_set():
                item["state"] = "canceled"
                return
            async with semaphore:
                if cancel_event.is_set():
                    item["state"] = "canceled"
                    return
                item["state"] = "running"
                item["attempts"] = 1
                self._log(job, f"开始处理：{item['name']}", level="debug", event="item_started", item=item)
                started = time.perf_counter()
                try:
                    result = await client.describe_image(Path(item["_path"]), job["prompt"])
                except asyncio.CancelledError:
                    item["state"] = "canceled"
                    raise
                except TaggingApiError as exc:
                    item["state"] = "failed"
                    item["error"] = str(exc)
                    item["attempts"] = exc.attempts
                    self._log(job, f"处理失败：{item['name']} - {exc}", level="error", event="item_failed", item=item)
                except (OSError, ValueError) as exc:
                    item["state"] = "failed"
                    item["error"] = str(exc)
                    self._log(job, f"处理失败：{item['name']} - {exc}", level="error", event="item_failed", item=item)
                else:
                    item["state"] = "ready"
                    item["proposed_caption"] = result["caption"]
                    item["attempts"] = int(result.get("attempts", 1) or 1)
                    item["elapsed_ms"] = result.get("elapsed_ms") or round((time.perf_counter() - started) * 1000)
                    self._log(
                        job,
                        f"处理完成：{item['name']}（{item['elapsed_ms']} ms）",
                        level="success",
                        event="item_succeeded",
                        item=item,
                    )

        try:
            await asyncio.gather(*(run_item(item) for item in job["items"]))
        except asyncio.CancelledError:
            for item in job["items"]:
                if item["state"] in {"queued", "running"}:
                    item["state"] = "canceled"
            job["error"] = "任务已停止"
            self._finish(job, "canceled")
            raise
        if cancel_event.is_set():
            self._mark_canceled(job)
        else:
            states = [item["state"] for item in job["items"]]
            if all(state == "ready" for state in states):
                final_state = "completed"
            elif any(state == "ready" for state in states):
                final_state = "partial"
            else:
                final_state = "failed"
            self._finish(job, final_state)

    def _finish(self, job: dict[str, Any], state: str) -> None:
        job["state"] = state
        job["finished_at"] = time.time()
        self._refresh_counts(job)
        self._log_finished(job, state)

    def _refresh_counts(self, job: dict[str, Any]) -> None:
        job["completed"] = sum(item["state"] in {"ready", "committed"} for item in job["items"])
        job["failed"] = sum(item["state"] == "failed" for item in job["items"])
        job["canceled"] = sum(item["state"] == "canceled" for item in job["items"])

    def _log_finished(self, job: dict[str, Any], state: str) -> None:
        level = "success" if state == "completed" else "warning" if state in {"partial", "canceled"} else "error"
        self._log(
            job,
            f"任务结束：{state}，完成 {job['completed']}，失败 {job['failed']}，取消 {job['canceled']}",
            level=level,
            event="job_finished",
        )

    def _mark_canceled(self, job: dict[str, Any]) -> None:
        for item in job["items"]:
            if item["state"] in {"queued", "running"}:
                item["state"] = "canceled"
        if job["state"] in {"queued", "running"}:
            self._finish(job, "canceled")

    def _get(self, job_id: str) -> dict[str, Any]:
        job = self.jobs.get(str(job_id))
        if job is None:
            raise KeyError("打标任务不存在")
        return job

    def _summary(self, job: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": job["id"],
            "state": job["state"],
            "created_at": job["created_at"],
            "created_at_text": _format_timestamp(job["created_at"]),
            "dataset_file": job["dataset_file"],
            "dataset_index": job["dataset_index"],
            "source": job["source"],
            "total": job["total"],
            "completed": job["completed"],
            "failed": job["failed"],
            "canceled": job["canceled"],
            "model": job["settings"].get("model", ""),
        }

    def _public_job(self, job: dict[str, Any]) -> dict[str, Any]:
        result = {key: value for key, value in job.items() if key != "items" and not key.startswith("_")}
        result["created_at_text"] = _format_timestamp(job["created_at"])
        result["started_at_text"] = _format_timestamp(job.get("started_at"))
        result["finished_at_text"] = _format_timestamp(job.get("finished_at"))
        result["items"] = []
        for item in job["items"]:
            result["items"].append({key: value for key, value in item.items() if not key.startswith("_")})
        return result

    def _prune(self) -> None:
        while len(self.jobs) > MAX_RETAINED_JOBS:
            job_id, job = next(iter(self.jobs.items()))
            if job["state"] in {"queued", "running"}:
                break
            self.jobs.pop(job_id, None)
            self._cancel_events.pop(job_id, None)
            self._commit_locks.pop(job_id, None)

    def _log(
        self,
        job: dict[str, Any],
        message: str,
        *,
        level: str = "info",
        event: str = "",
        item: dict[str, Any] | None = None,
    ) -> None:
        self.logs.append(
            message,
            level=level,
            event=event,
            job_id=job.get("id", ""),
            item_id=item.get("id", "") if item else "",
        )


def _public_job_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": settings.get("provider", "openai_compatible"),
        "base_url": settings.get("base_url", ""),
        "model": settings.get("model", ""),
        "concurrency": settings.get("concurrency", 1),
    }


def _format_timestamp(value: Any) -> str:
    if value is None:
        return ""
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError, OverflowError):
        return ""
