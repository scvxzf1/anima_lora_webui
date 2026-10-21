"""External API tagging service facade."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from web.services.settings_service import get_global_settings

from .caption_snapshots import (
    create_snapshot as _create_caption_snapshot,
    delete_snapshot as _delete_caption_snapshot,
    list_snapshots as _list_caption_snapshots,
    read_snapshot as _read_caption_snapshot,
    restore_snapshot as _restore_caption_snapshot,
)
from .client import OpenAICompatibleClient
from .download_jobs import ModelDownloadService
from .jobs import TaggingJobManager, caption_scope_key
from .prompt_presets import (
    create_prompt_preset,
    delete_prompt_preset,
    list_prompt_presets,
    update_prompt_preset,
)
from .profiles import (
    activate_profile,
    create_profile,
    delete_profile,
    list_profiles,
    update_profile,
)
from .provider_registry import list_provider_types
from .providers.factory import get_tagger
from .settings import get_public_settings, load_settings, save_settings
from .tag_dictionary import TagDictionaryService


class TaggingService:
    """Compose provider settings, connectivity checks, and job management."""

    def __init__(self, app=None):
        self.app = app
        current = load_settings()
        global_settings = get_global_settings()
        self.jobs = TaggingJobManager(
            log_retention_lines=current.get("log_retention_lines", 200),
            max_retained_jobs=global_settings.get("tagging_max_retained_jobs", 40),
        )
        self.model_downloads = ModelDownloadService()
        self.tag_dictionary = TagDictionaryService()
        # Keep a short alias for embedded callers that used the service as a
        # composition root before the public facade methods were added.
        self.downloads = self.model_downloads

    def get_settings(self) -> dict[str, Any]:
        return get_public_settings()

    def save_settings(self, data: dict[str, Any] | None) -> dict[str, Any]:
        saved = save_settings(data)
        self.jobs.set_log_retention(saved.get("log_retention_lines"))
        return saved

    def list_prompt_presets(self) -> dict[str, Any]:
        return list_prompt_presets(include_builtins=True)

    def create_prompt_preset(self, data: dict[str, Any] | None) -> dict[str, Any]:
        return create_prompt_preset(data, include_builtins=True)

    def update_prompt_preset(self, preset_id: str, data: dict[str, Any] | None) -> dict[str, Any]:
        return update_prompt_preset(preset_id, data, include_builtins=True)

    def delete_prompt_preset(self, preset_id: str) -> dict[str, Any]:
        return delete_prompt_preset(preset_id, include_builtins=True)

    def list_provider_types(self) -> dict[str, Any]:
        return {"ok": True, "provider_types": list_provider_types()}

    def list_profiles(self) -> dict[str, Any]:
        return list_profiles()

    def create_profile(self, data: dict[str, Any] | None) -> dict[str, Any]:
        return create_profile(data)

    def update_profile(self, profile_id: str, data: dict[str, Any] | None) -> dict[str, Any]:
        return update_profile(profile_id, data)

    def delete_profile(self, profile_id: str) -> dict[str, Any]:
        return delete_profile(profile_id)

    def activate_profile(self, profile_id: str) -> dict[str, Any]:
        return activate_profile(profile_id)

    async def list_model_assets(self) -> dict[str, Any]:
        return await self.model_downloads.list_assets()

    async def get_model_asset(self, asset_id: str) -> dict[str, Any]:
        return await self.model_downloads.get_asset(asset_id)

    async def start_model_download(self, asset_id: str) -> dict[str, Any]:
        return await self.model_downloads.start_download(asset_id)

    def list_model_downloads(self) -> dict[str, Any]:
        return self.model_downloads.list_downloads()

    def get_model_download(self, download_id: str) -> dict[str, Any]:
        return self.model_downloads.get_download(download_id)

    async def cancel_model_download(self, download_id: str) -> dict[str, Any]:
        return await self.model_downloads.cancel_download(download_id)

    def get_tag_dictionary_status(self) -> dict[str, Any]:
        return self.tag_dictionary.status()

    async def start_tag_dictionary_download(self) -> dict[str, Any]:
        return await self.tag_dictionary.start_download()

    def get_tag_dictionary_download(self, download_id: str) -> dict[str, Any]:
        return self.tag_dictionary.get_download(download_id)

    async def cancel_tag_dictionary_download(self, download_id: str) -> dict[str, Any]:
        return await self.tag_dictionary.cancel_download(download_id)

    async def translate_tags(self, tags: list[Any], target_language: str) -> dict[str, Any]:
        return await self.tag_dictionary.translate(tags, target_language)

    def get_logs(self, *, after: Any = 0, limit: Any = None, job_id: str = "") -> dict[str, Any]:
        return self.jobs.get_logs(after=after, limit=limit, job_id=job_id)

    def clear_logs(self) -> dict[str, Any]:
        return self.jobs.clear_logs()

    def set_job_retention(self, value: Any) -> int:
        return self.jobs.set_job_retention(value)

    async def test_provider(self, mode: str = "ping", profile_id: str | None = None) -> dict[str, Any]:
        settings = load_settings()
        if profile_id:
            from .profiles import get_effective_settings

            settings = get_effective_settings(profile_id)
        provider = str(settings.get("provider") or "openai_compatible").strip().lower()
        if provider in {"wd14", "cltagger"}:
            started = time.perf_counter()
            tagger = None
            try:
                tagger = get_tagger(provider, settings)
                available, reason = await asyncio.to_thread(tagger.is_available)
                if not available:
                    return {
                        "ok": False,
                        "available": False,
                        "provider": provider,
                        "elapsed_ms": round((time.perf_counter() - started) * 1000),
                        "error": reason,
                    }
                if str(mode or "ping").lower() == "actual":
                    await asyncio.to_thread(tagger.prepare)
                return {
                    "ok": True,
                    "available": True,
                    "provider": provider,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000),
                    "message": reason,
                    "runtime_warning": getattr(tagger, "runtime_warning", ""),
                }
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                return {
                    "ok": False,
                    "available": False,
                    "provider": provider,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000),
                    "error": str(exc),
                }
            except Exception as exc:  # noqa: BLE001 - optional runtimes vary
                return {
                    "ok": False,
                    "available": False,
                    "provider": provider,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000),
                    "error": str(exc) or "本地 provider 测试失败",
                }
            finally:
                if tagger is not None:
                    close = getattr(tagger, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:  # noqa: BLE001 - test cleanup is best effort
                            pass
        client = OpenAICompatibleClient(settings)
        if str(mode or "ping").lower() == "actual":
            return await client.actual()
        return await client.ping()

    async def create_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self.jobs.create(payload)

    async def rerun_job(
        self,
        job_id: str,
        profile_id: str = "",
        item_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        return await self.jobs.rerun(job_id, profile_id=profile_id, item_ids=item_ids)

    def list_jobs(self) -> dict[str, Any]:
        return {"ok": True, "jobs": self.jobs.list()}

    def clear_finished_jobs(self, job_ids: list[str]) -> dict[str, Any]:
        return self.jobs.clear_finished(job_ids)

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self.jobs.snapshot(job_id)

    async def cancel_job(self, job_id: str) -> dict[str, Any]:
        return await self.jobs.cancel(job_id)

    def update_item(self, job_id: str, item_id: str, text: str) -> dict[str, Any]:
        return self.jobs.update_item(job_id, item_id, text)

    def update_items(self, job_id: str, updates: Any) -> dict[str, Any]:
        return self.jobs.update_items(job_id, updates)

    async def commit_job(
        self,
        job_id: str,
        *,
        all_items: bool = False,
        item_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        return await self.jobs.commit(job_id, all_items=all_items, item_ids=item_ids)

    def list_caption_snapshots(
        self,
        *,
        dataset_file: str = "",
        dataset_index: int | None = None,
    ) -> dict[str, Any]:
        return {
            "ok": True,
            "snapshots": _list_caption_snapshots(dataset_file=dataset_file, dataset_index=dataset_index),
        }

    async def create_caption_snapshot(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        data = payload if isinstance(payload, dict) else {}
        dataset_file = str(data.get("dataset_file") or data.get("file") or "").strip()
        if not dataset_file:
            raise ValueError("请先选择数据集预设")
        try:
            dataset_index = int(data.get("dataset_index", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("dataset_index 必须是整数") from exc
        source = "source" if str(data.get("source") or "source").lower() == "source" else "training"
        label = str(data.get("label") or "").strip()[:80]
        # Manual restore points share the scope write lock with commit, so a
        # backup can never capture a half-written caption set.  The key is the
        # resolved caption directory, so rows over one directory share the lock;
        # inside a commit the order is job lock -> scope lock.
        scope_key = caption_scope_key(dataset_file, dataset_index, source=source)
        async with self.jobs.dataset_write_lock(scope_key):
            snapshot = await asyncio.to_thread(
                _create_caption_snapshot,
                dataset_file,
                dataset_index,
                source=source,
                reason="manual",
                label=label,
            )
        return {"ok": True, "snapshot": snapshot}

    async def restore_caption_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        # Read the manifest first: it carries the scope_dir the restore will touch
        # and is validated there (missing/broken documents keep their 404/400).
        # Restore takes ONLY that scope lock; it never takes a job lock, so the
        # global order job lock -> scope lock is never reversed.
        manifest = await asyncio.to_thread(_read_caption_snapshot, snapshot_id)
        scope_key = str(manifest.get("scope_dir") or "").strip() or f"snapshot:{snapshot_id}"
        async with self.jobs.dataset_write_lock(scope_key):
            return await asyncio.to_thread(_restore_caption_snapshot, snapshot_id)

    def delete_caption_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        return _delete_caption_snapshot(snapshot_id)

    async def shutdown(self) -> None:
        await self.tag_dictionary.shutdown()
        await self.model_downloads.shutdown()
        await self.jobs.shutdown()


__all__ = ["TaggingService"]
