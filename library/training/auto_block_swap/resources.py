"""Read-only host telemetry. Disk swap is reported, never counted as RAM."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import psutil

GIB = 1024**3


@dataclass(frozen=True)
class HostMemory:
    total: int
    available: int
    swap_total: int
    swap_used: int
    swap_in: int
    swap_out: int

    @property
    def reserve(self) -> int:
        return max(2 * GIB, self.total // 10)

    def to_dict(self):
        return asdict(self)


def _cgroup_available() -> tuple[int, int] | None:
    """Respect cgroup v2 limits when the process is container constrained."""
    try:
        rows = Path("/proc/self/cgroup").read_text().splitlines()
        group = next(row.split(":", 2)[2] for row in rows if row.startswith("0::"))
        root = Path("/sys/fs/cgroup")
        path = root / group.lstrip("/")
        limits = []
        while path == root or root in path.parents:
            maximum = (path / "memory.max").read_text().strip()
            if maximum != "max":
                total = int(maximum)
                used = int((path / "memory.current").read_text().strip())
                limits.append((total, max(0, total - used)))
            if path == root:
                break
            path = path.parent
        if limits:
            return min(total for total, _ in limits), min(free for _, free in limits)
    except (OSError, ValueError, StopIteration):
        pass
    return None


def host_memory() -> HostMemory:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    limit = _cgroup_available()
    total = min(memory.total, limit[0]) if limit else memory.total
    available = min(memory.available, limit[1]) if limit else memory.available
    return HostMemory(total, available, swap.total, swap.used, swap.sin, swap.sout)
