"""Shared, bounded-cost GPU sampling for the training service."""

from __future__ import annotations

import asyncio
import csv
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from web.services.training.gpu import normalize_gpu_whitelist


GPU_QUERY = "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu"
GPU_FORMAT = "--format=csv,noheader,nounits"


@dataclass(frozen=True)
class GpuSnapshot:
    rows: tuple[dict[str, Any], ...]
    sampled_at: float

    def inventory(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.rows]

    def history_rows(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for row in self.rows:
            item = {key: row[key] for key in ("index", "uuid", "name", "gpu_util", "gpu_temp") if key in row}
            for source, target in (("memory_used_mb", "vram_used_gb"), ("memory_total_mb", "vram_total_gb")):
                if source in row:
                    item[target] = round(row[source] / 1024, 2)
            result.append(item)
        return result

    def stats(self, whitelist: list[int] | None = None) -> dict[str, Any]:
        selected = normalize_gpu_whitelist(whitelist)
        rows = [row for row in self.rows if row["index"] in selected] if selected else list(self.rows[:1])
        if not rows:
            return {}
        stats: dict[str, Any] = {
            "gpu_index": rows[0]["index"],
            "gpu_indices": [row["index"] for row in rows],
        }
        if all("memory_total_mb" in row for row in rows):
            stats["vram_total_gb"] = round(sum(row["memory_total_mb"] for row in rows) / 1024, 2)
        if all("memory_used_mb" in row for row in rows):
            stats["vram_used_gb"] = round(sum(row["memory_used_mb"] for row in rows) / 1024, 2)
        for key in ("gpu_util", "gpu_temp"):
            values = [row[key] for row in rows if key in row]
            if values:
                stats[key] = max(values)
        return stats


def _integer(value: str) -> int | None:
    try:
        return int(value.strip())
    except ValueError:
        return None


def parse_gpu_snapshot(text: str) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for parts in csv.reader(text.splitlines(), skipinitialspace=True):
        parts = [part.strip() for part in parts]
        if len(parts) != 7:
            continue
        index = _integer(parts[0])
        total = _integer(parts[4])
        if index is None:
            continue
        used = _integer(parts[3])
        utilization = _integer(parts[5])
        temperature = _integer(parts[6])
        row: dict[str, Any] = {
            "index": index,
            "uuid": parts[1],
            "name": parts[2],
            "label": f"GPU {index} · {parts[2]}",
        }
        if total is not None:
            row["memory_total_mb"] = total
            row["memory_total_gb"] = round(total / 1024, 1)
        if used is not None:
            row["memory_used_mb"] = used
            row["memory_used_gb"] = round(used / 1024, 1)
        if utilization is not None:
            row["gpu_util"] = utilization
        if temperature is not None:
            row["gpu_temp"] = temperature
        rows.append(row)
    return tuple(rows)


async def sample_gpus(
    *,
    create_subprocess_exec: Callable[..., Awaitable[Any]] | None = None,
    timeout: float = 3.0,
) -> GpuSnapshot | None:
    runner = create_subprocess_exec or asyncio.create_subprocess_exec
    try:
        proc = await asyncio.wait_for(runner(
            "nvidia-smi", GPU_QUERY, GPU_FORMAT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        ), timeout=timeout)
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return None
        if getattr(proc, "returncode", 0) != 0:
            return None
        rows = parse_gpu_snapshot(stdout.decode(errors="replace"))
        if stdout.strip() and not rows:
            return None
        return GpuSnapshot(rows, time.time())
    except (OSError, asyncio.TimeoutError):
        return None


class GpuSnapshotCache:
    def __init__(self, sampler: Callable[[], Awaitable[GpuSnapshot | None]] = sample_gpus):
        self._sampler = sampler
        self._lock = asyncio.Lock()
        self._snapshot: GpuSnapshot | None = None
        self._sampled_monotonic = 0.0
        self._retry_at = 0.0

    async def read(self, *, ttl: float, force: bool = False) -> GpuSnapshot | None:
        now = time.monotonic()
        if not force and self._snapshot is not None and now - self._sampled_monotonic < ttl:
            return self._snapshot
        async with self._lock:
            now = time.monotonic()
            if not force and self._snapshot is not None and now - self._sampled_monotonic < ttl:
                return self._snapshot
            if not force and now < self._retry_at:
                return None
            snapshot = await self._sampler()
            now = time.monotonic()
            if snapshot is None:
                self._retry_at = now + 3.0
                self._snapshot = None
                return None
            self._snapshot = snapshot
            self._sampled_monotonic = now
            self._retry_at = 0.0
            return snapshot
