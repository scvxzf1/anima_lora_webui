"""Async subprocess wiring for GPU helpers."""

from __future__ import annotations

import asyncio
from typing import Any

from web.services.training.gpu import (
    get_gpu_stats as get_gpu_stats_impl,
    list_available_gpus as list_available_gpus_impl,
)


async def get_gpu_stats(gpu_whitelist: list[int] | None = None) -> dict[str, Any]:
    return await get_gpu_stats_impl(
        gpu_whitelist,
        create_subprocess_exec=asyncio.create_subprocess_exec,
        stdout_pipe=asyncio.subprocess.PIPE,
        stderr_devnull=asyncio.subprocess.DEVNULL,
    )


async def list_available_gpus() -> list[dict[str, Any]]:
    return await list_available_gpus_impl(
        create_subprocess_exec=asyncio.create_subprocess_exec,
        stdout_pipe=asyncio.subprocess.PIPE,
        stderr_devnull=asyncio.subprocess.DEVNULL,
    )
