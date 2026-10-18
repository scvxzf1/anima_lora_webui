"""Process-local gate shared by destructive data cleanup and job launchers."""

from __future__ import annotations

from contextlib import contextmanager
import threading
from collections.abc import Iterator
from functools import wraps

_state_lock = threading.Lock()
_active_cleanups = 0
_active_launches = 0


def is_data_cleanup_active() -> bool:
    with _state_lock:
        return _active_cleanups > 0


@contextmanager
def data_cleanup_lease() -> Iterator[None]:
    global _active_cleanups
    with _state_lock:
        if _active_launches:
            raise RuntimeError("任务正在启动，请等待启动完成后再清理")
        if _active_cleanups:
            raise RuntimeError("已有数据清理正在进行中")
        _active_cleanups += 1
    try:
        yield
    finally:
        with _state_lock:
            _active_cleanups = max(0, _active_cleanups - 1)

@contextmanager
def data_launch_lease() -> Iterator[None]:
    """Reserve startup atomically against cleanup until task state is published."""
    global _active_launches
    with _state_lock:
        if _active_cleanups:
            raise RuntimeError("数据清理正在进行中")
        _active_launches += 1
    try:
        yield
    finally:
        with _state_lock:
            _active_launches -= 1


def guard_data_launch(function):
    """Hold a startup reservation across awaits without blocking the event loop."""
    @wraps(function)
    async def guarded(*args, **kwargs):
        with data_launch_lease():
            return await function(*args, **kwargs)
    return guarded

