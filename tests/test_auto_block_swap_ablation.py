"""Deterministic algorithm ablations, NOT hardware performance evidence."""

from library.training.auto_block_swap.policy import Measurement, SwapSearch


def simulate(*, fixed_stride=False, backoff=True, reserve=True):
    search = SwapSearch(26, max_trials=16)
    candidate = search.next_candidate()
    results = []
    while candidate is not None:
        # Synthetic workload: raw OOM below16, safety reserve needs >=18.
        safe = candidate >= (18 if reserve else 16)
        sample = Measurement(candidate, safe, 1 + candidate / 10, candidate - 16)
        results.append(sample)
        if fixed_stride:
            if not safe or candidate == 0:
                break
            candidate -= 1
            continue
        search.observe(sample)
        if not safe and not backoff:
            break
        candidate = search.next_candidate()
    successes = [sample for sample in results if sample.safe]
    best = min(sample.seconds for sample in successes)
    selected = max(
        sample.blocks for sample in successes if sample.seconds <= best * 1.03
    )
    return selected, len(results)


def test_growth_ablation_reduces_candidate_count_without_changing_result():
    full = simulate()
    linear = simulate(fixed_stride=True)
    assert full[0] == linear[0] == 18
    assert full[1] < linear[1]


def test_failure_bisection_recovers_better_tested_candidate():
    full = simulate()
    no_backoff = simulate(backoff=False)
    assert full[0] == 18
    assert no_backoff[0] == 20


def test_reserve_ablation_exposes_unsafe_margin():
    assert simulate(reserve=True)[0] == 18
    assert simulate(reserve=False)[0] == 16
