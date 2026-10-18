"""Concurrency contract for the shared data-cleanup startup gate."""

from __future__ import annotations

import asyncio

import pytest

from web.services.data_cleanup_gate import (
    data_cleanup_lease,
    data_launch_lease,
    guard_data_launch,
)


def test_cleanup_and_launch_reservations_are_mutually_exclusive():
    with data_launch_lease():
        with pytest.raises(RuntimeError, match="任务正在启动"):
            with data_cleanup_lease():
                pass

    with data_cleanup_lease():
        with pytest.raises(RuntimeError, match="数据清理正在进行中"):
            with data_launch_lease():
                pass


def test_async_launch_guard_holds_reservation_across_awaits():
    entered = asyncio.Event()
    release = asyncio.Event()

    @guard_data_launch
    async def start_task():
        entered.set()
        await release.wait()
        return "started"

    async def run():
        task = asyncio.create_task(start_task())
        await entered.wait()
        with pytest.raises(RuntimeError, match="任务正在启动"):
            with data_cleanup_lease():
                pass
        release.set()
        assert await task == "started"
        with data_cleanup_lease():
            pass

    asyncio.run(run())
