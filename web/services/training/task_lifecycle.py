"""Background task ownership for one WebUI training run."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any


def track_run_task(
    self,
    awaitable: Awaitable[Any],
    *,
    generation: int,
    output: bool = False,
) -> asyncio.Task[Any]:
    task = asyncio.create_task(awaitable)
    run_tasks = self._job_tasks.setdefault(generation, set())
    run_tasks.add(task)
    if output:
        self._output_task = task
        self._output_task_generation = generation

    def task_done(done: asyncio.Task[Any]) -> None:
        tasks = self._job_tasks.get(generation)
        if tasks is not None:
            tasks.discard(done)
            if not tasks:
                self._job_tasks.pop(generation, None)
        if output and self._output_task is done:
            self._output_task = None
            self._output_task_generation = 0
        if done.cancelled():
            return
        try:
            exc = done.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            self._remember_log("error", f"后台任务异常: {exc}")

    task.add_done_callback(task_done)
    return task


async def cancel_run_tasks(
    self,
    generation: int | None = None,
    *,
    exclude: asyncio.Task[Any] | None = None,
) -> None:
    generations = [generation] if generation is not None else list(self._job_tasks)
    tasks: list[asyncio.Task[Any]] = []
    for run_generation in generations:
        for task in tuple(self._job_tasks.get(run_generation, ())):
            if task is exclude or task.done():
                continue
            task.cancel()
            tasks.append(task)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
