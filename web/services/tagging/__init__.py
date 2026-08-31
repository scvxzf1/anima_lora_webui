"""External API tagging service facade."""

from __future__ import annotations

from typing import Any

from .client import OpenAICompatibleClient
from .jobs import TaggingJobManager
from .prompt_presets import (
    create_prompt_preset,
    delete_prompt_preset,
    list_prompt_presets,
    update_prompt_preset,
)
from .settings import get_public_settings, load_settings, save_settings


class TaggingService:
    """Compose provider settings, connectivity checks, and job management."""

    def __init__(self, app=None):
        self.app = app
        current = load_settings()
        self.jobs = TaggingJobManager(log_retention_lines=current.get("log_retention_lines", 200))

    def get_settings(self) -> dict[str, Any]:
        return get_public_settings()

    def save_settings(self, data: dict[str, Any] | None) -> dict[str, Any]:
        saved = save_settings(data)
        self.jobs.set_log_retention(saved.get("log_retention_lines"))
        return saved

    def list_prompt_presets(self) -> dict[str, Any]:
        return list_prompt_presets()

    def create_prompt_preset(self, data: dict[str, Any] | None) -> dict[str, Any]:
        return create_prompt_preset(data)

    def update_prompt_preset(self, preset_id: str, data: dict[str, Any] | None) -> dict[str, Any]:
        return update_prompt_preset(preset_id, data)

    def delete_prompt_preset(self, preset_id: str) -> dict[str, Any]:
        return delete_prompt_preset(preset_id)

    def get_logs(self, *, after: Any = 0, limit: Any = None, job_id: str = "") -> dict[str, Any]:
        return self.jobs.get_logs(after=after, limit=limit, job_id=job_id)

    def clear_logs(self) -> dict[str, Any]:
        return self.jobs.clear_logs()

    async def test_provider(self, mode: str = "ping") -> dict[str, Any]:
        client = OpenAICompatibleClient(load_settings())
        if str(mode or "ping").lower() == "actual":
            return await client.actual()
        return await client.ping()

    async def create_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self.jobs.create(payload)

    def list_jobs(self) -> dict[str, Any]:
        return {"ok": True, "jobs": self.jobs.list()}

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self.jobs.snapshot(job_id)

    async def cancel_job(self, job_id: str) -> dict[str, Any]:
        return await self.jobs.cancel(job_id)

    def update_item(self, job_id: str, item_id: str, text: str) -> dict[str, Any]:
        return self.jobs.update_item(job_id, item_id, text)

    async def commit_job(
        self,
        job_id: str,
        *,
        all_items: bool = False,
        item_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        return await self.jobs.commit(job_id, all_items=all_items, item_ids=item_ids)

    async def shutdown(self) -> None:
        await self.jobs.shutdown()


__all__ = ["TaggingService"]
