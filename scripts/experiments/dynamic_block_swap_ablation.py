"""Deterministic policy ablations, not a substitute for hardware measurements."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics

from library.training.auto_block_swap.online_policy import Decision, OnlineSwapPolicy
from library.training.auto_block_swap.process import write_result


class AblatedPolicy(OnlineSwapPolicy):
    """Unsafe counterfactuals live only in this synthetic experiment."""

    def __init__(self, variant, **kwargs):
        super().__init__(26, **kwargs)
        self.variant = variant

    def guard(self, shape, *, step, safe_count):
        if self.variant == "without_pressure_guard":
            self.known_inputs.add(shape)
            return None
        return super().guard(shape, step=step, safe_count=safe_count)

    def observe(self, *args, **kwargs):
        if self.variant == "fixed_stride":
            self.stride = 2
        decision = super().observe(*args, **kwargs)
        if self.variant == "without_warmup_cost":
            self.trial_warmup_cost = 0
        if decision and decision.reason == "paired_reference":
            if self.variant == "without_reference_recheck":
                before = statistics.mean(self.before.values())
                candidate = statistics.mean(self.candidate_times.values())
                accepted = candidate < before * 0.97
                self.last_verdict = {"accepted": accepted}
                self.phase = "baseline"
                return Decision(
                    self.candidate if accepted else self.reference,
                    "accepted" if accepted else "rejected",
                )
        return decision

    def changed(self, blocks, *, seconds=0):
        super().changed(blocks, seconds=seconds)
        if self.variant == "fixed_stride":
            self.stride = 2


def replay(variant, timings, *, warmup=0, pressure=None):
    policy = AblatedPolicy(variant, interval=4, warmup=warmup)
    switches = []
    unsafe_updates = 0
    accepted = 0
    for step, seconds in enumerate(timings, 1):
        floor = (pressure or {}).get(step, 0)
        guard = policy.guard("shape", step=step, safe_count=floor)
        if guard:
            policy.changed(guard.blocks)
            switches.append({"step": step, "blocks": guard.blocks, "reason": guard.reason})
        unsafe_updates += policy.current < floor
        timing = seconds(policy.current) if callable(seconds) else seconds
        decision = policy.observe(
            "shape", timing, step=step, remaining=1000, promotion_floor=0
        )
        if decision:
            policy.changed(decision.blocks)
            switches.append({"step": step, "blocks": decision.blocks, "reason": decision.reason})
            accepted += decision.reason == "accepted"
    return {
        "variant": variant,
        "final_blocks": policy.current,
        "accepted": accepted,
        "unsafe_updates": unsafe_updates,
        "switches": switches,
        "last_verdict": policy.last_verdict,
    }


def run_ablations():
    drift = [10] * 4 + [8] * 4 + [7] * 4
    cold = [10] * 5 + [10000] + [8] * 4 + [10] * 5
    pressure = {step: 25 for step in range(5, 9)}

    def plateau(blocks):
        return 10 if blocks >= 24 else 9.95 if blocks >= 20 else 8

    return {
        "evidence_kind": "synthetic_policy_ablation_not_gpu_performance",
        "reference_drift": [
            replay(variant, drift)
            for variant in ("full", "without_reference_recheck")
        ],
        "compile_cost": [
            replay(variant, cold, warmup=1)
            for variant in ("full", "without_warmup_cost")
        ],
        "pressure": [
            replay(variant, [10] * 8, pressure=pressure)
            for variant in ("full", "without_pressure_guard")
        ],
        "plateau": [
            replay(variant, [plateau] * 120)
            for variant in ("full", "fixed_stride")
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = run_ablations()
    write_result(args.output, results)
    for scenario, variants in results.items():
        if isinstance(variants, list):
            for row in variants:
                print(scenario, row["variant"], row["final_blocks"], row["accepted"], row["unsafe_updates"])


if __name__ == "__main__":
    main()
