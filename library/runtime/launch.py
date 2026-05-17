"""Shared launch argument helpers for training subprocesses."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

ACCELERATE_NUM_PROCESSES_ENV = "ANIMA_ACCELERATE_NUM_PROCESSES"


def resolve_accelerate_num_processes(env: Mapping[str, str] | None = None) -> str:
    values = os.environ if env is None else env
    raw = values.get(ACCELERATE_NUM_PROCESSES_ENV, "").strip()
    if not raw:
        return "1"
    try:
        count = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{ACCELERATE_NUM_PROCESSES_ENV} must be a positive integer, got {raw!r}"
        ) from exc
    if count < 1:
        raise ValueError(
            f"{ACCELERATE_NUM_PROCESSES_ENV} must be a positive integer, got {raw!r}"
        )
    return str(count)


def accelerate_training_command_prefix(
    python_exe: str,
    train_script: str | Path,
    env: Mapping[str, str] | None = None,
) -> list[str]:
    return [
        python_exe,
        "-m",
        "accelerate.commands.accelerate_cli",
        "launch",
        "--num_processes",
        resolve_accelerate_num_processes(env),
        "--num_machines",
        "1",
        "--dynamo_backend",
        "no",
        "--num_cpu_threads_per_process",
        "3",
        "--mixed_precision",
        "bf16",
        str(train_script),
    ]
