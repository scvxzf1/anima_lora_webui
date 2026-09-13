"""Resource-first online search with paired, shape-matched timing windows."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import statistics
from .preferences import prefer_candidate


@dataclass(frozen=True)
class Decision:
    blocks: int
    reason: str


class OnlineSwapPolicy:
    def __init__(self, maximum: int, *, interval: int = 8, warmup: int = 2, preference="balanced"):
        self.maximum = maximum
        self.current = maximum
        self.interval = interval
        self.warmup = warmup
        self.stride = 2
        self.direction = -1
        self.phase = "baseline"
        self.reference = maximum
        self.candidate = maximum
        self.before = {}
        self.candidate_times = {}
        self.samples = defaultdict(list)
        self.cold_samples = defaultdict(list)
        self.trial_warmup_cost = 0.0
        self.trial_switch_cost = 0.0
        self.seen = defaultdict(int)
        self.known_shapes = set()
        self.known_inputs = set()
        self.cooldown_until = 0
        self.rejected_below = -1
        self.retry_after = 0
        self.last_verdict = None
        self.preference = preference

    def reset_window(self):
        self.samples.clear()
        self.cold_samples.clear()
        self.seen.clear()

    def changed(self, blocks, *, seconds=0):
        self.current = blocks
        self.trial_switch_cost += seconds
        self.reset_window()

    def abandon(self, step, *, cooldown=32):
        self.phase = "baseline"
        self.reference = self.current
        self.cooldown_until = step + cooldown
        self.reset_window()

    def guard(self, shape, *, step, safe_count):
        if safe_count > self.maximum:
            raise RuntimeError(
                "Dynamic AUTO cannot retain GPU reserve even at maximum swap"
            )
        unknown = shape not in self.known_inputs
        self.known_inputs.add(shape)
        if unknown:
            self.abandon(step, cooldown=0)
            if self.current < self.maximum:
                return Decision(self.maximum, "new_shape")
        if safe_count > self.current:
            self.abandon(step)
            return Decision(safe_count, "gpu_pressure")
        return None

    def observe(
        self, shape, seconds, *, step, remaining, promotion_floor, switch_seconds=0
    ):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Dynamic AUTO requires finite positive timing")
        if step < self.cooldown_until:
            return None
        if step >= self.retry_after:
            self.rejected_below = -1
        if shape not in self.known_shapes:
            self.known_shapes.add(shape)
            self.abandon(step, cooldown=0)
        self.seen[shape] += 1
        if self.seen[shape] <= self.warmup:
            self.cold_samples[shape].append(seconds)
            return None
        self.samples[shape].append(seconds)
        if sum(map(len, self.samples.values())) < self.interval:
            return None
        # Keep long, imbalanced bucket streams bounded.
        self.samples[shape] = self.samples[shape][-self.interval :]
        if set(self.samples) != self.known_shapes or any(
            len(v) < 2 for v in self.samples.values()
        ):
            return None
        window = {
            key: statistics.median(values) for key, values in self.samples.items()
        }
        if self.phase == "baseline":
            # Avoid spending the remaining run on an unfinished experiment.
            if remaining < 3 * (self.interval + self.warmup * len(window)):
                return None
            floor = max(promotion_floor, self.rejected_below + 1)
            target = min(
                self.maximum, max(floor, self.current + self.direction * self.stride)
            )
            if target == self.current or target > self.maximum:
                self.direction = -self.direction
                self.reset_window()
                return None
            self.reference, self.candidate = self.current, target
            self.before = window
            self.trial_warmup_cost = 0.0
            self.trial_switch_cost = 0.0
            self.phase = "candidate"
            return Decision(target, "explore")
        self.trial_warmup_cost += sum(
            max(0, cold - window[key])
            for key, values in self.cold_samples.items()
            for cold in values
        )
        if self.phase == "candidate":
            self.candidate_times = window
            self.phase = "recheck"
            return Decision(self.reference, "paired_reference")
        keys = set(window) & set(self.before) & set(self.candidate_times)
        drift = (
            max(abs(window[k] / self.before[k] - 1) for k in keys) if keys else math.inf
        )
        reference = (
            statistics.mean(math.sqrt(window[k] * self.before[k]) for k in keys)
            if keys
            else 0
        )
        candidate = (
            statistics.mean(self.candidate_times[k] for k in keys) if keys else math.inf
        )
        saving = reference - candidate
        stable = keys == self.known_shapes and drift <= 0.10
        resource_preferred = prefer_candidate(
            self.preference, self.reference, self.candidate,
            reference_seconds=reference, candidate_seconds=candidate,
        )
        cost = self.trial_warmup_cost + self.trial_switch_cost + switch_seconds
        speed_worthwhile = (
            saving > 0.03 * reference and saving * remaining > cost
            and (self.preference == "balanced" or resource_preferred or reference > candidate * 1.10)
        )
        # A resource preference may trade up to 10% total projected time, including migration.
        resource_worthwhile = resource_preferred and (
            candidate * remaining + cost <= reference * remaining * 1.10
        )
        worthwhile = (
            stable
            and self.candidate >= promotion_floor
            and (speed_worthwhile or resource_worthwhile)
        )
        self.last_verdict = {
            "stable": stable,
            "drift": drift if math.isfinite(drift) else None,
            "reference_seconds": reference,
            "candidate_seconds": candidate if keys else None,
            "accepted": worthwhile,
            "preference": self.preference,
            "resource_preferred": resource_preferred,
            "warmup_cost_seconds": self.trial_warmup_cost,
            "switch_cost_seconds": self.trial_switch_cost + switch_seconds,
        }
        self.phase = "baseline"
        self.reset_window()
        if worthwhile:
            self.stride = min(8, self.stride * 2)
            return Decision(self.candidate, "accepted")
        self.cooldown_until = step + max(32, self.interval * 4)
        if stable and candidate <= reference * 1.03 and self.stride < 8:
            self.stride *= 2
        else:
            self.stride = 2
            self.direction = -self.direction
        return None
