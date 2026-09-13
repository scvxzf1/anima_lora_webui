"""Device-independent online batch search; no model, IO or CUDA dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True)
class BatchMeasurement:
    batch: int
    seconds: float
    base_bytes: int
    peak_bytes: int
    budget_bytes: int


class AutoBatchPolicy:
    """Warm up at 1, explore 2/4, predict growth, then bracket or settle.

    A policy belongs to one model/device/dtype/encode shape for one run.
    Tail batches never promote the search. Prediction and throughput switches
    are experimental ablation controls, not user configuration.
    """

    def __init__(self, max_batch: int = 32, *, predict=True, throughput=True):
        self.max_batch = max(1, max_batch)
        self.current = 1
        self.ceiling = self.max_batch
        self.predict = predict
        self.throughput = throughput
        self.base = 0
        self.per_item = 0.0
        self.cold = True
        self.samples: list[float] = []
        self.rates: dict[int, float] = {}
        self.settled = False
        self.settled_budget = 0
        self.stable_steps = 0
        self.recovering = False

    def capacity(self, budget: int) -> int:
        if not self.predict or not self.per_item:
            return self.ceiling
        estimate = int((budget - self.base) / (self.per_item * 1.1))
        return max(1, min(self.ceiling, estimate))

    def prepare(self, budget: int) -> int:
        limit = self.capacity(budget)
        if limit < self.current:
            self._select(limit)
        if self.settled:
            # Do not keep hitting an unchanged limit. Revisit after a material
            # increase in available memory and at least 16 successful batches.
            if (
                self.stable_steps >= 16
                and budget > self.settled_budget * 1.25
                and limit > self.current
            ):
                self._select(min(limit, self.current * 2))
        return self.current

    def _select(self, batch: int) -> None:
        self.current = max(1, min(batch, self.ceiling))
        self.samples = []
        self.settled = False
        self.stable_steps = 0

    def _settle(self, budget: int) -> None:
        eligible = {
            b: rate
            for b, rate in self.rates.items()
            if b <= min(self.current, self.capacity(budget))
        }
        if self.throughput and eligible:
            best = max(eligible.values())
            self.current = min(b for b, rate in eligible.items() if rate >= best * 0.95)
        self.samples = []
        self.settled = True
        self.settled_budget = budget
        self.stable_steps = 0

    def success(self, measured: BatchMeasurement) -> None:
        # Tails may tighten the memory envelope, but never count toward growth
        # or throughput decisions. A sudden larger peak must remain a safety signal.
        self.base = max(self.base, measured.base_bytes)
        self.per_item = max(
            self.per_item,
            max(0, measured.peak_bytes - measured.base_bytes) / measured.batch,
        )
        if self.cold:
            self.cold = False
            return
        if self.settled:
            self.stable_steps += 1
            return
        if measured.batch != self.current:
            return
        self.samples.append(measured.batch / max(measured.seconds, 1e-9))
        if len(self.samples) < 2:
            return
        rate = median(self.samples)
        previous_best = max(self.rates.values(), default=0)
        self.rates[self.current] = rate
        limit = self.capacity(measured.budget_bytes)
        plateau = (
            self.throughput
            and self.current > 4
            and previous_best
            and rate < previous_best * 1.05
            and not self.recovering
        )
        if plateau or self.current >= limit:
            self._settle(measured.budget_bytes)
            return
        if self.recovering:
            candidate = (self.current + limit + 1) // 2
        elif self.current < 4 or not self.predict:
            candidate = self.current * 2
        else:
            candidate = self.current * 4
        self._select(min(limit, candidate))

    def failure(self, batch: int) -> None:
        if batch <= 1:
            raise ValueError("batch=1 failure cannot be recovered by batching")
        self.ceiling = min(self.ceiling, batch - 1)
        self.rates = {b: rate for b, rate in self.rates.items() if b < batch}
        self.recovering = True
        self._select(max(1, batch // 2))
