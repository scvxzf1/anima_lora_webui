"""Pure descending search; only measured, resource-safe candidates can win."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Measurement:
    blocks: int
    safe: bool
    seconds: float = 0.0
    headroom: int = 0


class SwapSearch:
    def __init__(self, maximum: int, *, max_trials: int = 6, preference="balanced"):
        if maximum < 0 or max_trials < 1:
            raise ValueError("Invalid AUTO block-swap search bounds")
        self.maximum = maximum
        self.max_trials = max_trials
        self.measurements: list[Measurement] = []
        self.failed_below = -1
        self.preference = preference

    def next_candidate(self) -> int | None:
        if len(self.measurements) >= self.max_trials:
            return None
        if not self.measurements:
            return self.maximum
        successes = [m.blocks for m in self.measurements if m.safe]
        if not successes:
            return None
        lowest = min(successes)
        if lowest <= self.failed_below + 1:
            return None
        if self.failed_below >= 0:
            return (lowest + self.failed_below) // 2
        # First establish a slope with a small change, then grow the stride.
        stride = min(8, 2 ** len(successes))
        return max(0, lowest - stride)

    def observe(self, measurement: Measurement) -> None:
        if measurement.blocks != self.next_candidate():
            raise ValueError("Unexpected or repeated block-swap candidate")
        if measurement.safe and (
            not math.isfinite(measurement.seconds) or measurement.seconds <= 0
        ):
            raise ValueError("Successful candidates require finite positive timing")
        self.measurements.append(measurement)
        if not measurement.safe:
            self.failed_below = max(self.failed_below, measurement.blocks)

    def select(self) -> int:
        successes = [m for m in self.measurements if m.safe]
        if not successes:
            raise RuntimeError("AUTO block swap found no resource-safe candidate")
        fastest = min(m.seconds for m in successes)
        # Prefer more VRAM headroom when the timing difference is just noise.
        tolerance = 1.03 if self.preference == "balanced" else 1.10
        choose = min if self.preference == "ram" else max
        return choose(m.blocks for m in successes if m.seconds <= fastest * tolerance)


def host_swap_limit(block_bytes: list[int], available: int, reserve: int, *, sparse=False) -> int:
    """Conservative incremental budget after the CPU model has been loaded.

    Default masters cover ALL blocks; sparse mode covers actual exchange pairs.
    Allow two largest blocks for packing/transfer scratch and a tail copy.
    No swap/pagefile bytes are added to available physical memory.
    """
    if len(block_bytes) < 3 or any(n <= 0 for n in block_bytes):
        raise ValueError("AUTO requires at least three nonempty swappable blocks")
    from library.runtime.block_swap_sparse_masters import participating_blocks

    budget = available - reserve - 2 * max(block_bytes)
    tail = 0
    maximum = 0
    for count, size in enumerate(reversed(block_bytes), 1):
        if count > len(block_bytes) - 2:
            break
        tail += size
        masters = sum(block_bytes[i] for i in participating_blocks(len(block_bytes), count)) if sparse else sum(block_bytes)
        if masters + tail > budget:
            break
        maximum = count
    return maximum
