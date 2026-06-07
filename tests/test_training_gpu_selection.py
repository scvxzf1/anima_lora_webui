from __future__ import annotations

import asyncio

from web.services import training_service
from web.services.training import gpu as gpu_helpers
from web.services.training_service import _apply_gpu_whitelist, _normalize_gpu_whitelist


def test_normalize_gpu_whitelist_filters_invalid_and_duplicates():
    assert _normalize_gpu_whitelist(["1", 0, "bad", 1, "-2", 2]) == [1, 0, 2]
    assert gpu_helpers.normalize_gpu_whitelist(["1", 0, "bad", 1, "-2", 2]) == [1, 0, 2]


def test_apply_gpu_whitelist_sets_cuda_visible_devices():
    env = {}

    _apply_gpu_whitelist(env, [2, 0])

    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["CUDA_VISIBLE_DEVICES"] == "2,0"


def test_apply_gpu_whitelist_keeps_default_when_empty():
    env = {"CUDA_VISIBLE_DEVICES": "3"}

    _apply_gpu_whitelist(env, [])

    assert env["CUDA_VISIBLE_DEVICES"] == "3"


def test_get_gpu_stats_uses_selected_physical_gpu(monkeypatch):
    class FakeProcess:
        async def communicate(self):
            return (
                b"0, 1024, 24576, 0, 44\n"
                b"1, 5120, 24576, 88, 71\n",
                b"",
            )

    async def fake_create_subprocess_exec(*args, **kwargs):
        return FakeProcess()

    monkeypatch.setattr(training_service.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    stats = asyncio.run(training_service._get_gpu_stats([1]))

    assert stats["gpu_index"] == 1
    assert stats["gpu_indices"] == [1]
    assert stats["vram_used_gb"] == 5
    assert stats["vram_total_gb"] == 24
    assert stats["gpu_util"] == 88
    assert stats["gpu_temp"] == 71


def test_get_gpu_stats_without_selection_keeps_first_gpu(monkeypatch):
    class FakeProcess:
        async def communicate(self):
            return (
                b"0, 1024, 24576, 11, 44\n"
                b"1, 5120, 24576, 88, 71\n",
                b"",
            )

    async def fake_create_subprocess_exec(*args, **kwargs):
        return FakeProcess()

    monkeypatch.setattr(training_service.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    stats = asyncio.run(training_service._get_gpu_stats([]))

    assert stats["gpu_index"] == 0
    assert stats["gpu_indices"] == [0]
    assert stats["gpu_util"] == 11


def test_training_gpu_helper_lists_available_gpus_with_injected_runner():
    class FakeProcess:
        async def communicate(self):
            return (
                b"0, NVIDIA RTX 5060 Ti, 16384\n"
                b"bad,row\n"
                b"1, NVIDIA RTX 4090, 24576\n",
                b"",
            )

    async def fake_create_subprocess_exec(*args, **kwargs):
        return FakeProcess()

    gpus = asyncio.run(
        gpu_helpers.list_available_gpus(
            create_subprocess_exec=fake_create_subprocess_exec,
            stdout_pipe=object(),
            stderr_devnull=object(),
        )
    )

    assert gpus == [
        {
            "index": 0,
            "name": "NVIDIA RTX 5060 Ti",
            "label": "GPU 0 · NVIDIA RTX 5060 Ti",
            "memory_total_mb": 16384,
            "memory_total_gb": 16.0,
        },
        {
            "index": 1,
            "name": "NVIDIA RTX 4090",
            "label": "GPU 1 · NVIDIA RTX 4090",
            "memory_total_mb": 24576,
            "memory_total_gb": 24.0,
        },
    ]
