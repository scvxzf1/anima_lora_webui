"""GPU selection and nvidia-smi parsing helpers for WebUI training."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable


CreateSubprocessExec = Callable[..., Awaitable[Any]]


async def get_gpu_stats(
    gpu_whitelist: list[int] | None = None,
    *,
    create_subprocess_exec: CreateSubprocessExec | None = None,
    stdout_pipe: Any | None = None,
    stderr_devnull: Any | None = None,
) -> dict[str, Any]:
    try:
        runner = create_subprocess_exec or asyncio.create_subprocess_exec
        proc = await runner(
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
            stdout=stdout_pipe if stdout_pipe is not None else asyncio.subprocess.PIPE,
            stderr=stderr_devnull if stderr_devnull is not None else asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        rows = parse_gpu_stats_rows(stdout.decode(errors="replace"))
        selected = normalize_gpu_whitelist(gpu_whitelist)
        if selected:
            selected_set = set(selected)
            rows = [row for row in rows if row["index"] in selected_set]
        else:
            rows = rows[:1]
        return aggregate_gpu_stats_rows(rows)
    except Exception:
        pass
    return {}


def parse_gpu_stats_rows(text: str) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            rows.append({
                "index": int(parts[0]),
                "memory_used_mb": int(parts[1]),
                "memory_total_mb": int(parts[2]),
                "gpu_util": int(parts[3]),
                "gpu_temp": int(parts[4]),
            })
        except ValueError:
            continue
    return rows


def aggregate_gpu_stats_rows(rows: list[dict[str, int]]) -> dict[str, Any]:
    if not rows:
        return {}
    indices = [row["index"] for row in rows]
    used_mb = sum(row["memory_used_mb"] for row in rows)
    total_mb = sum(row["memory_total_mb"] for row in rows)
    return {
        "gpu_index": indices[0],
        "gpu_indices": indices,
        "vram_used_gb": round(used_mb / 1024, 2),
        "vram_total_gb": round(total_mb / 1024, 2),
        "gpu_util": max(row["gpu_util"] for row in rows),
        "gpu_temp": max(row["gpu_temp"] for row in rows),
    }


async def list_available_gpus(
    *,
    create_subprocess_exec: CreateSubprocessExec | None = None,
    stdout_pipe: Any | None = None,
    stderr_devnull: Any | None = None,
) -> list[dict[str, Any]]:
    try:
        runner = create_subprocess_exec or asyncio.create_subprocess_exec
        proc = await runner(
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
            stdout=stdout_pipe if stdout_pipe is not None else asyncio.subprocess.PIPE,
            stderr=stderr_devnull if stderr_devnull is not None else asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
    except Exception:
        return []

    gpus: list[dict[str, Any]] = []
    for line in stdout.decode(errors="replace").splitlines():
        item = parse_available_gpu_row(line)
        if item is not None:
            gpus.append(item)
    return gpus


def parse_available_gpu_row(line: str) -> dict[str, Any] | None:
    parts = [part.strip() for part in line.split(",", 2)]
    if len(parts) < 2:
        return None
    try:
        index = int(parts[0])
    except ValueError:
        return None
    memory_total_mb = int_or_none(parts[2]) if len(parts) >= 3 else None
    item: dict[str, Any] = {
        "index": index,
        "name": parts[1],
        "label": f"GPU {index} · {parts[1]}",
    }
    if memory_total_mb is not None:
        item["memory_total_mb"] = memory_total_mb
        item["memory_total_gb"] = round(memory_total_mb / 1024, 1)
    return item


def normalize_gpu_whitelist(value: Any) -> list[int]:
    if value is None or value == "":
        return []
    raw_items = value if isinstance(value, list) else [value]
    out: list[int] = []
    for item in raw_items:
        try:
            index = int(str(item).strip())
        except (TypeError, ValueError):
            continue
        if index < 0 or index in out:
            continue
        out.append(index)
    return out


def apply_gpu_whitelist(env: dict[str, str], whitelist: list[int]) -> None:
    if whitelist:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in whitelist)


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
