from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from web.services.training import live_monitor
from web.services.training.gpu_snapshot import GpuSnapshot, GpuSnapshotCache, parse_gpu_snapshot, sample_gpus
from web.services.training_service import TrainingService
from web.routes.training import handle_gpus


GPU_OUTPUT = (
    '0, GPU-aaa, "GPU Alpha", 1024, 24576, 12, 44\n'
    '1, GPU-bbb, "GPU Beta", 5120, 32768, 88, 71\n'
)


def test_gpu_snapshot_preserves_all_cards_and_legacy_aggregate():
    snapshot = GpuSnapshot(parse_gpu_snapshot(GPU_OUTPUT), 100.0)

    assert [row["uuid"] for row in snapshot.inventory()] == ["GPU-aaa", "GPU-bbb"]
    assert snapshot.inventory()[1]["memory_used_gb"] == 5.0
    assert snapshot.history_rows() == [
        {"index": 0, "uuid": "GPU-aaa", "name": "GPU Alpha", "gpu_util": 12, "gpu_temp": 44,
         "vram_used_gb": 1.0, "vram_total_gb": 24.0},
        {"index": 1, "uuid": "GPU-bbb", "name": "GPU Beta", "gpu_util": 88, "gpu_temp": 71,
         "vram_used_gb": 5.0, "vram_total_gb": 32.0},
    ]
    assert snapshot.stats([1, 0]) == {
        "gpu_index": 0,
        "gpu_indices": [0, 1],
        "vram_used_gb": 6.0,
        "vram_total_gb": 56.0,
        "gpu_util": 88,
        "gpu_temp": 71,
    }
    assert snapshot.stats([])["gpu_indices"] == [0]
    assert snapshot.stats([4]) == {}


def test_gpu_snapshot_keeps_device_when_a_metric_is_unavailable():
    snapshot = GpuSnapshot(parse_gpu_snapshot("2, GPU-ccc, GPU Gamma, N/A, 8192, N/A, 40"), 100.0)

    assert snapshot.inventory()[0]["memory_total_gb"] == 8.0
    assert "memory_used_gb" not in snapshot.inventory()[0]
    assert "vram_used_gb" not in snapshot.history_rows()[0]
    assert snapshot.stats([2]) == {"gpu_index": 2, "gpu_indices": [2], "vram_total_gb": 8.0, "gpu_temp": 40}


def test_gpu_snapshot_keeps_device_when_total_memory_is_unavailable():
    snapshot = GpuSnapshot(parse_gpu_snapshot("2, GPU-ccc, GPU Gamma, 1024, N/A, 20, 40"), 100.0)

    assert snapshot.inventory()[0]["name"] == "GPU Gamma"
    assert "memory_total_gb" not in snapshot.inventory()[0]
    assert snapshot.stats([2]) == {"gpu_index": 2, "gpu_indices": [2], "vram_used_gb": 1.0,
                                    "gpu_util": 20, "gpu_temp": 40}


def test_sampler_uses_one_bounded_query_for_all_gpus():
    commands = []

    class Process:
        returncode = 0

        async def communicate(self):
            return GPU_OUTPUT.encode(), b""

    async def runner(*args, **kwargs):
        commands.append(args)
        return Process()

    snapshot = asyncio.run(sample_gpus(create_subprocess_exec=runner))

    assert snapshot is not None
    assert len(snapshot.rows) == 2
    assert len(commands) == 1
    assert commands[0][0] == "nvidia-smi"
    assert "uuid,name,memory.used" in commands[0][1]


def test_snapshot_cache_single_flight_force_refresh_and_shorter_running_ttl(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("web.services.training.gpu_snapshot.time.monotonic", lambda: clock[0])
    samples = []

    async def sampler():
        samples.append(clock[0])
        await asyncio.sleep(0)
        return GpuSnapshot(parse_gpu_snapshot(GPU_OUTPUT), clock[0])

    async def run():
        cache = GpuSnapshotCache(sampler)
        first, second = await asyncio.gather(cache.read(ttl=10), cache.read(ttl=10))
        assert first is second
        assert len(samples) == 1
        clock[0] = 3.0
        assert await cache.read(ttl=10) is first
        assert len(samples) == 1
        assert await cache.read(ttl=2) is not first
        assert len(samples) == 2
        assert await cache.read(ttl=2, force=True) is not first
        assert len(samples) == 3

    asyncio.run(run())


def test_failed_sampling_is_backed_off_and_not_returned_as_live_data(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("web.services.training.gpu_snapshot.time.monotonic", lambda: clock[0])
    results = [GpuSnapshot(parse_gpu_snapshot(GPU_OUTPUT), 100.0), None, GpuSnapshot((), 110.0)]
    calls = []

    async def sampler():
        calls.append(clock[0])
        return results.pop(0)

    async def run():
        cache = GpuSnapshotCache(sampler)
        assert await cache.read(ttl=2) is not None
        clock[0] = 2.1
        assert await cache.read(ttl=2) is None
        assert await cache.read(ttl=2) is None
        assert len(calls) == 2
        assert (await cache.read(ttl=2, force=True)).rows == ()
        assert len(calls) == 3

    asyncio.run(run())


def test_service_inventory_fails_closed_and_monitor_writes_only_fresh_samples(monkeypatch):
    class Cache:
        def __init__(self):
            self.values = [GpuSnapshot(parse_gpu_snapshot(GPU_OUTPUT), 100.0), None]

        async def read(self, **kwargs):
            return self.values.pop(0)

    service = SimpleNamespace(status="running", _gpu_snapshot_cache=Cache(), current_gpu_whitelist=[1],
                              _run_generation=1, _last_output_at=None, _latest_system_stats=None)
    records = []
    messages = []
    service._append_history_jsonl = lambda name, stats: records.append((name, dict(stats)))

    async def broadcast(message):
        messages.append(message)
        service.status = "idle"

    service._broadcast = broadcast
    monkeypatch.setattr(live_monitor, "SYSTEM_MONITOR_INTERVAL_SECONDS", 0)

    async def run():
        await live_monitor._monitor_system(service)
        inventory = await TrainingService.gpu_inventory(service)
        assert inventory == {"gpus": [], "sampled_at": None, "stale": True}

    asyncio.run(run())
    assert len(records) == len(messages) == 1
    assert records[0][0] == "system.jsonl"
    assert records[0][1]["gpu_indices"] == [1]
    assert records[0][1]["vram_used_gb"] == 5.0
    assert [row["uuid"] for row in records[0][1]["per_gpu"]] == ["GPU-aaa", "GPU-bbb"]
    assert messages[0]["per_gpu"] == records[0][1]["per_gpu"]
    assert json.loads(json.dumps(records[0][1]))["per_gpu"][1]["vram_total_gb"] == 32.0


def test_monitor_does_not_write_failed_sample(monkeypatch):
    async def failed_read(**kwargs):
        return None

    service = SimpleNamespace(status="running", _gpu_snapshot_cache=SimpleNamespace(read=failed_read),
                              current_gpu_whitelist=[0], _run_generation=1, _last_output_at=None,
                              _latest_system_stats=None)
    records = []
    messages = []
    service._append_history_jsonl = lambda *args: records.append(args)

    async def broadcast(message):
        messages.append(message)

    async def stop_after_sleep(_seconds):
        service.status = "idle"

    service._broadcast = broadcast
    monkeypatch.setattr(live_monitor.asyncio, "sleep", stop_after_sleep)
    asyncio.run(live_monitor._monitor_system(service))

    assert service._latest_system_stats is None
    assert records == messages == []


def test_gpu_route_preserves_envelope_and_forwards_explicit_refresh():
    calls = []

    async def inventory(*, force=False):
        calls.append(force)
        return {"gpus": [{"index": 1, "name": "GPU Beta"}], "sampled_at": 100.0, "stale": False}

    request = SimpleNamespace(app={"training_service": SimpleNamespace(gpu_inventory=inventory)},
                              query={"refresh": "1"})
    response = asyncio.run(handle_gpus(request))

    assert calls == [True]
    assert json.loads(response.text) == {"ok": True, "gpus": [{"index": 1, "name": "GPU Beta"}],
                                         "sampled_at": 100.0, "stale": False}
