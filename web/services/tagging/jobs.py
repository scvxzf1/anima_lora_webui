"""In-memory tagging job lifecycle and review/write-back orchestration."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .client import OpenAICompatibleClient, TaggingApiError
from .local_worker_client import LocalTaggingWorkerClient, LocalWorkerError
from .memory_log import DEFAULT_LOG_RETENTION_LINES, TaggingMemoryLog
from .profiles import get_effective_settings
from .settings import load_settings
from .storage import CaptionWriteConflict, resolve_tagging_image, write_caption

MAX_ITEMS_PER_JOB = 500
DEFAULT_MAX_RETAINED_JOBS = 40
SHUTDOWN_TIMEOUT_SECONDS = 2.0


class TaggingJobManager:
    """Own short-lived jobs for one WebUI process.

    Jobs are intentionally not mixed with the training queue.  They contain
    review drafts and may be canceled while an external provider is in flight;
    the browser polls the sanitized snapshots returned by this class.
    """

    def __init__(
        self,
        *,
        log_retention_lines: int = DEFAULT_LOG_RETENTION_LINES,
        max_retained_jobs: int = DEFAULT_MAX_RETAINED_JOBS,
        local_worker_factory: Callable[..., LocalTaggingWorkerClient] | None = None,
    ):
        self.jobs: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._tasks: dict[str, asyncio.Task] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._commit_locks: dict[str, asyncio.Lock] = {}
        self._rerun_locks: dict[str, asyncio.Lock] = {}
        self._local_workers: dict[str, LocalTaggingWorkerClient] = {}
        self._local_worker_factory = local_worker_factory or LocalTaggingWorkerClient
        self.logs = TaggingMemoryLog(log_retention_lines)
        self.max_retained_jobs = _normalize_job_retention(max_retained_jobs)

    async def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        dataset_file = str(payload.get("dataset_file") or payload.get("file") or "").strip()
        if not dataset_file:
            raise ValueError("请先选择数据集预设")
        try:
            dataset_index = int(payload.get("dataset_index", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("数据集序号无效") from exc
        source = "source" if str(payload.get("source") or "source").lower() == "source" else "training"
        profile_id = str(payload.get("profile_id") or "").strip()
        provider_settings = get_effective_settings(profile_id) if profile_id else load_settings()
        provider = str(provider_settings.get("provider") or "openai_compatible").strip().lower()
        prompt = str(payload.get("user_prompt") or payload.get("prompt") or "").strip()
        system_prompt = str(payload.get("system_prompt") or provider_settings.get("system_prompt") or "").strip()
        # Local taggers do not consume prompts.  Keep the fields in the job
        # snapshot for compatibility, but do not force users to fill them.
        if provider == "openai_compatible" and not prompt:
            raise ValueError("请输入打标提示词")
        if provider == "openai_compatible" and not system_prompt:
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
            "profile_id": provider_settings.get("_profile_id", "legacy-openai"),
            "profile_name": provider_settings.get("_profile_name", "默认外部 API"),
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
        self._rerun_locks[job_id] = asyncio.Lock()
        self._prune()
        self._log(job, f"任务已加入队列，共 {len(items)} 张图片", event="job_queued")
        self._tasks[job_id] = asyncio.create_task(self._run(job), name=f"tagging:{job_id}")
        return self.snapshot(job_id)

    async def rerun(
        self,
        job_id: str,
        *,
        profile_id: str = "",
        item_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        lock = self._rerun_locks.setdefault(str(job_id), asyncio.Lock())
        async with lock:
            return await self._rerun_in_place(job_id, profile_id=profile_id, item_ids=item_ids)

    async def _rerun_in_place(
        self,
        job_id: str,
        *,
        profile_id: str = "",
        item_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        source_job = self._get(job_id)
        if source_job["state"] in {"queued", "running"}:
            raise RuntimeError("任务仍在运行，结束后才能重新打标")
        # A terminal state is recorded before the owning coroutine reaches its
        # cleanup block.  Let that short tail finish before replacing the task
        # reference so the previous run cannot race the in-place rerun.
        previous_task = self._tasks.get(job_id)
        if previous_task is not None and not previous_task.done():
            await asyncio.gather(previous_task, return_exceptions=True)
            source_job = self._get(job_id)
            if source_job["state"] in {"queued", "running"}:
                raise RuntimeError("任务仍在运行，结束后才能重新打标")
        selected_profile = str(profile_id or source_job.get("profile_id") or "").strip()
        provider_settings = get_effective_settings(selected_profile) if selected_profile else dict(source_job.get("_provider_settings") or load_settings())
        provider = str(provider_settings.get("provider") or "openai_compatible").strip().lower()
        prompt = str(source_job.get("prompt") or "").strip()
        system_prompt = str(source_job.get("system_prompt") or provider_settings.get("system_prompt") or "").strip()
        if provider == "openai_compatible" and not prompt:
            raise ValueError("请输入打标提示词")
        if provider == "openai_compatible" and not system_prompt:
            raise ValueError("请输入系统提示词")
        if len(system_prompt) > 10_000 or len(prompt) > 10_000:
            raise ValueError("单条提示词最多支持 10000 个字符")
        provider_settings = {**provider_settings, "system_prompt": system_prompt}
        source_items = list(source_job.get("items") or [])
        requested_ids = [str(item_id).strip() for item_id in (item_ids or []) if str(item_id).strip()]
        if requested_ids:
            source_by_id = {
                str(item.get("id") or ""): item
                for item in source_items
                if str(item.get("id") or "").strip()
            }
            unknown = [item_id for item_id in requested_ids if item_id not in source_by_id]
            if unknown:
                raise ValueError(f"找不到打标图片：{unknown[0]}")
            selected = []
            seen: set[str] = set()
            for item_id in requested_ids:
                if item_id in seen:
                    continue
                seen.add(item_id)
                selected.append(source_by_id[item_id])
        else:
            selected = source_items
        if not selected:
            raise ValueError("没有可重新打标的图片")
        for item in selected:
            if not item.get("_path"):
                resolved = await asyncio.to_thread(
                    resolve_tagging_image,
                    source_job["dataset_file"],
                    source_job["dataset_index"],
                    item["file"],
                    source=source_job["source"],
                )
                item["_path"] = resolved["path"]
            item["state"] = "queued"
            item["proposed_caption"] = ""
            item["error"] = ""
            item["commit_error"] = ""
            item["attempts"] = 0
            item["elapsed_ms"] = None

        job_id = str(source_job["id"])
        source_job["state"] = "queued"
        source_job["started_at"] = None
        source_job["finished_at"] = None
        source_job["error"] = ""
        source_job["profile_id"] = provider_settings.get("_profile_id", selected_profile or "legacy-openai")
        source_job["profile_name"] = provider_settings.get("_profile_name", "默认外部 API")
        source_job["prompt"] = prompt[:10_000]
        source_job["system_prompt"] = system_prompt[:10_000]
        source_job["settings"] = _public_job_settings(provider_settings)
        source_job["_provider_settings"] = provider_settings
        source_job["_run_item_ids"] = [str(item.get("id") or "") for item in selected]
        self._cancel_events[job_id] = asyncio.Event()
        self._refresh_counts(source_job)
        self._log(source_job, f"已在当前任务中重新加入 {len(selected)} 张图片", event="job_rerun")
        self._tasks[job_id] = asyncio.create_task(self._run(source_job), name=f"tagging:{job_id}")
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

    def set_job_retention(self, value: Any) -> int:
        self.max_retained_jobs = _normalize_job_retention(value)
        self._prune()
        return self.max_retained_jobs

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
        workers = list(self._local_workers.values())
        if workers:
            await asyncio.gather(*(worker.stop() for worker in workers), return_exceptions=True)
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
        self._local_workers.clear()

    async def _run(self, job: dict[str, Any]) -> None:
        job_id = job["id"]
        try:
            await self._run_job(job)
        finally:
            # Do not remove a newer in-place rerun task that reused this ID.
            current_task = self._tasks.get(job_id)
            if current_task is asyncio.current_task():
                self._tasks.pop(job_id, None)

    async def _run_job(self, job: dict[str, Any]) -> None:
        job_id = job["id"]
        cancel_event = self._cancel_events[job_id]
        run_items = self._run_items(job)
        if cancel_event.is_set():
            self._mark_canceled(job)
            return
        job["state"] = "running"
        job["started_at"] = time.time()
        self._log(job, "任务开始", event="job_started")
        settings = dict(job.get("_provider_settings") or load_settings())
        provider = str(settings.get("provider") or "openai_compatible").strip().lower()
        if provider in {"wd14", "cltagger"}:
            await self._run_local_job(job, settings, cancel_event, run_items)
            return
        try:
            client = OpenAICompatibleClient(settings)
        except (ValueError, OSError) as exc:
            job["error"] = str(exc)
            self._log(job, f"外部 API 初始化失败：{exc}", level="error", event="provider_failed")
            for item in run_items:
                item["state"] = "failed"
                item["error"] = str(exc)
            self._finish(job, self._final_state(job))
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
            await asyncio.gather(*(run_item(item) for item in run_items))
        except asyncio.CancelledError:
            for item in run_items:
                if item["state"] in {"queued", "running"}:
                    item["state"] = "canceled"
            job["error"] = "任务已停止"
            self._finish(job, "canceled")
            raise
        if cancel_event.is_set():
            self._mark_canceled(job)
        else:
            self._finish(job, self._final_state(job))

    async def _run_local_job(
        self,
        job: dict[str, Any],
        settings: dict[str, Any],
        cancel_event: asyncio.Event,
        run_items: list[dict[str, Any]] | None = None,
    ) -> None:
        """Run one local provider in a disposable subprocess."""

        provider = str(settings.get("provider") or "").strip().lower()
        job_id = str(job.get("id") or "")
        run_items = run_items if run_items is not None else self._run_items(job)
        try:
            worker = self._local_worker_factory(
                provider=provider,
                settings=settings,
                job_id=job_id,
            )
        except Exception as exc:  # noqa: BLE001 - injected factories may vary
            self._fail_local_job(job, provider, exc, phase="initialization", run_items=run_items)
            return
        if cancel_event.is_set():
            self._mark_local_canceled(job)
            return

        paths = [Path(item["_path"]) for item in run_items]
        for item in run_items:
            item["state"] = "running"
            item["attempts"] = 1
        started = time.perf_counter()
        self._local_workers[job_id] = worker
        try:
            results = await worker.run(paths)
        except asyncio.CancelledError:
            self._mark_local_canceled(job)
            raise
        except LocalWorkerError as exc:
            if cancel_event.is_set():
                self._mark_local_canceled(job)
                return
            phase = "initialization" if exc.phase in {"initialization", "protocol"} else "inference"
            self._fail_local_job(job, provider, exc, phase=phase, run_items=run_items)
            return
        except Exception as exc:  # noqa: BLE001 - normalize inference failures
            if cancel_event.is_set():
                self._mark_local_canceled(job)
                return
            self._fail_local_job(job, provider, exc, phase="inference", run_items=run_items)
            return
        finally:
            self._local_workers.pop(job_id, None)
        runtime_warning = str(getattr(worker, "runtime_warning", "") or "").strip()
        if runtime_warning:
            job["settings"]["runtime_warning"] = runtime_warning[:500]
            self._log(job, runtime_warning[:500], level="warning", event="provider_warning")
        elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))

        result_by_path: dict[str, dict[str, Any]] = {}
        for result in results:
            image = result.get("image") if isinstance(result, dict) else None
            if image is not None:
                result_by_path[_path_key(image)] = result
        for index, item in enumerate(run_items):
            if cancel_event.is_set():
                item["state"] = "canceled"
                continue
            item["state"] = "running"
            item["attempts"] = 1
            result = result_by_path.get(_path_key(item["_path"]))
            if result is None and index < len(results):
                # Positional fallback is only safe for providers that omit
                # the image field entirely.  Never reuse an explicitly tagged
                # result for a different file when a path cannot be resolved.
                candidate = results[index]
                if isinstance(candidate, dict) and not candidate.get("image"):
                    result = candidate
            if not result:
                item["state"] = "failed"
                item["error"] = "本地模型没有返回结果"
                self._log(job, f"处理失败：{item['name']} - {item['error']}", level="error", event="item_failed", item=item)
                continue
            error = str(result.get("error") or "").strip()
            if error:
                item["state"] = "failed"
                item["error"] = error[:500]
                self._log(job, f"处理失败：{item['name']} - {item['error']}", level="error", event="item_failed", item=item)
                continue
            caption = str(result.get("caption") or "").strip()
            tags = result.get("tags")
            if not caption and isinstance(tags, list):
                caption = ", ".join(str(tag).strip() for tag in tags if str(tag).strip())
            item["elapsed_ms"] = elapsed_ms
            if not caption:
                item["state"] = "empty"
                item["error"] = "模型未返回标签"
                self._log(job, f"未生成标签：{item['name']}", level="warning", event="item_empty", item=item)
                continue
            item["state"] = "ready"
            item["proposed_caption"] = caption[:100_000]
            item["error"] = ""
            self._log(job, f"处理完成：{item['name']}（{elapsed_ms} ms）", level="success", event="item_succeeded", item=item)

        if cancel_event.is_set():
            self._mark_canceled(job)
            return
        self._finish(job, self._final_state(job))

    def _mark_local_canceled(self, job: dict[str, Any]) -> None:
        job["error"] = "任务已停止"
        self._mark_canceled(job)

    def _fail_local_job(
        self,
        job: dict[str, Any],
        provider: str,
        exc: BaseException,
        *,
        phase: str,
        run_items: list[dict[str, Any]] | None = None,
    ) -> None:
        message = str(exc).strip() or (
            "本地模型初始化失败" if phase == "initialization" else "本地模型推理失败"
        )
        job["error"] = message[:500]
        initializing = phase == "initialization"
        label = "初始化失败" if initializing else "推理失败"
        event = "provider_failed" if initializing else "inference_failed"
        self._log(
            job,
            f"{provider} {label}：{job['error']}",
            level="error",
            event=event,
        )
        targets = run_items if run_items is not None else self._run_items(job)
        for item in targets:
            if item["state"] != "canceled":
                item["state"] = "failed"
                item["error"] = job["error"]
        self._finish(job, self._final_state(job))

    def _run_items(self, job: dict[str, Any]) -> list[dict[str, Any]]:
        item_ids = job.get("_run_item_ids")
        if not isinstance(item_ids, list) or not item_ids:
            return list(job.get("items") or [])
        wanted = {str(item_id) for item_id in item_ids if str(item_id).strip()}
        if not wanted:
            return list(job.get("items") or [])
        return [item for item in job.get("items") or [] if str(item.get("id") or "") in wanted]

    def _final_state(self, job: dict[str, Any]) -> str:
        states = [str(item.get("state") or "") for item in job.get("items") or []]
        successful = {"ready", "committed"}
        if states and all(state in successful for state in states):
            return "completed"
        if any(state in successful for state in states):
            return "partial"
        return "failed"

    def _finish(self, job: dict[str, Any], state: str) -> None:
        job["state"] = state
        job["finished_at"] = time.time()
        self._refresh_counts(job)
        self._log_finished(job, state)
        self._prune()

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
            "profile_id": job.get("profile_id", "legacy-openai"),
            "profile_name": job.get("profile_name", "默认外部 API"),
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
        while len(self.jobs) > self.max_retained_jobs:
            candidate = next(
                (
                    (job_id, job)
                    for job_id, job in self.jobs.items()
                    if job["state"] not in {"queued", "running"}
                ),
                None,
            )
            if candidate is None:
                break
            job_id, _job = candidate
            self.jobs.pop(job_id, None)
            self._cancel_events.pop(job_id, None)
            self._commit_locks.pop(job_id, None)
            self._rerun_locks.pop(job_id, None)

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
    result = {
        "provider": settings.get("provider", "openai_compatible"),
        "profile_id": settings.get("_profile_id", "legacy-openai"),
        "profile_name": settings.get("_profile_name", "默认外部 API"),
        "base_url": settings.get("base_url", ""),
        "model": settings.get("model", ""),
        "concurrency": settings.get("concurrency", 1),
    }
    if result["provider"] in {"wd14", "cltagger"}:
        result.update(
            {
                "asset_id": settings.get("asset_id", ""),
                "device": settings.get("device", "auto"),
                "gpu_index": settings.get("gpu_index"),
                "batch_size": settings.get("batch_size", 1),
                "general_threshold": settings.get("general_threshold", 0.35),
                "character_threshold": settings.get("character_threshold", 0.85),
                "blacklist": list(settings.get("blacklist") or []) if isinstance(settings.get("blacklist"), list) else [],
            }
        )
        if result["provider"] == "cltagger":
            result.update(
                {
                    key: bool(settings.get(key, default))
                    for key, default in {
                        "add_copyright_tag": True,
                        "add_artist_tag": False,
                        "add_meta_tag": False,
                        "add_model_tag": False,
                        "add_rating_tag": False,
                        "add_quality_tag": False,
                    }.items()
                }
            )
    return result


def _normalize_job_retention(value: Any) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        count = DEFAULT_MAX_RETAINED_JOBS
    return max(1, min(500, count))


def _path_key(value: Any) -> str:
    try:
        return str(Path(value).resolve())
    except (OSError, TypeError, ValueError):
        return str(value or "")


def _format_timestamp(value: Any) -> str:
    if value is None:
        return ""
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError, OverflowError):
        return ""
