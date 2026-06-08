"""Shared launch argument helpers for training subprocesses."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

ACCELERATE_NUM_PROCESSES_ENV = "ANIMA_ACCELERATE_NUM_PROCESSES"
ACCELERATE_LAUNCH_ENV = "ANIMA_ACCELERATE_LAUNCH"
ACCELERATE_MIXED_PRECISION_ENV = "ANIMA_ACCELERATE_MIXED_PRECISION"
_ACCELERATE_MIXED_PRECISION_CHOICES = {"no", "fp16", "bf16"}


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


def resolve_accelerate_mixed_precision(env: Mapping[str, str] | None = None) -> str:
    values = os.environ if env is None else env
    raw = values.get(ACCELERATE_MIXED_PRECISION_ENV, "").strip().lower()
    if not raw:
        return "bf16"
    if raw not in _ACCELERATE_MIXED_PRECISION_CHOICES:
        raise ValueError(
            f"{ACCELERATE_MIXED_PRECISION_ENV} must be one of "
            f"{sorted(_ACCELERATE_MIXED_PRECISION_CHOICES)}, got {raw!r}"
        )
    return raw


def accelerate_training_command_prefix(
    python_exe: str,
    train_script: str | Path,
    env: Mapping[str, str] | None = None,
) -> list[str]:
    values = os.environ if env is None else env
    if not values.get(ACCELERATE_LAUNCH_ENV):
        return [python_exe, str(train_script)]
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
        resolve_accelerate_mixed_precision(env),
        str(train_script),
    ]
