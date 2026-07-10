"""RNG snapshot helpers and nsys profile-step parsing for training loops."""

from __future__ import annotations

import random
from typing import Optional

import torch


def parse_profile_steps(args) -> tuple[int, int] | None:
    """Parse --profile_steps 'start-end' into (start, end) or None.

    When set, the loop calls ``torch.cuda.profiler.start()`` at ``start``
    and ``stop()`` after ``end``, so pair this with::

        nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \\
            accelerate launch ... train.py --profile_steps 3-5
    """
    raw = getattr(args, "profile_steps", None)
    if not raw:
        return None
    if "-" in raw:
        a, b = raw.split("-", 1)
        return int(a), int(b)
    n = int(raw)
    return n, n + 2


def switch_rng_state(
    seed: int,
) -> tuple[torch.ByteTensor, Optional[torch.ByteTensor], tuple]:
    cpu_rng_state = torch.get_rng_state()
    gpu_rng_state = torch.cuda.get_rng_state()
    python_rng_state = random.getstate()

    torch.manual_seed(seed)
    random.seed(seed)

    return (cpu_rng_state, gpu_rng_state, python_rng_state)


def restore_rng_state(
    rng_states: tuple[torch.ByteTensor, Optional[torch.ByteTensor], tuple],
) -> None:
    cpu_rng_state, gpu_rng_state, python_rng_state = rng_states
    torch.set_rng_state(cpu_rng_state)
    torch.cuda.set_rng_state(gpu_rng_state)
    random.setstate(python_rng_state)
