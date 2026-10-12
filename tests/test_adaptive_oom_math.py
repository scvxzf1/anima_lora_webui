"""Finite-domain evidence for the bounded OOM policy, not a GPU safety proof."""

from argparse import Namespace
from itertools import product
from types import SimpleNamespace

import pytest
import torch

from library.training.adaptive_runtime import contract
from library.training.adaptive_runtime.retry import MemoryPlan, RetryLimits, run_recovery
from library.training.auto_block_swap.policy import Measurement, SwapSearch, host_swap_limit
from library.training.auto_block_swap.probe import model_block_bytes


def _oom_runner(plans, *, host_limits=None, capacity=None):
    def runner(plan, *, attempt, resume):
        assert resume is None
        plans.append(plan)
        result = {"status": "cuda_oom", "stage": "model_load",
                  "optimizer_started": False}
        if capacity is not None:
            host_limit = host_limits[attempt]
            result.update(model_limit=capacity["model_limit"],
                          requested_cap=capacity["requested_cap"],
                          effective_cap=min(capacity["effective_cap"], host_limit),
                          host_limit=host_limit)
        return result

    return runner


def _dense_feasible(sizes, available, reserve, count):
    # An independent accounting equation: full masters, two-block scratch,
    # and the exact swapped suffix; Python integers have no rounding error.
    return (reserve + 2 * max(sizes) + sum(sizes) + sum(sizes[-count:])
            <= available)


def test_fixed_cap_exponential_closed_form_and_attempt_bound():
    cases = 0
    for cap in range(9):
        for start, attempts, increment in product(range(cap + 1), range(1, 6), range(1, 5)):
            plans = []
            report = run_recovery(
                MemoryPlan(start, True, 4, 2),
                RetryLimits(cap, attempts, increment, swap_strategy="exponential"),
                _oom_runner(plans), record=lambda value: None,
            )
            actual = [plan.blocks_to_swap for plan in plans]
            expected = []
            for index in range(attempts):
                if expected and expected[-1] == cap:
                    break
                # Before the last permitted transition, the geometric series
                # is d(1 + 2 + ... + 2^(k-1)) = d(2^k - 1).
                value = min(cap, start + increment * (2**index - 1))
                if index == attempts - 1 and index > 0:
                    value = cap
                expected.append(value)
            assert actual == expected, (cap, start, attempts, increment)
            assert len(actual) <= min(attempts, cap - start + 1)
            geometric_steps = 0
            while increment * (2**geometric_steps - 1) < cap - start:
                geometric_steps += 1
            assert len(actual) <= min(attempts, geometric_steps + 1)
            assert all(left < right for left, right in zip(actual, actual[1:]))
            assert all(start <= value <= cap for value in actual)
            assert len(set(plans)) == len(plans)
            assert all(plan.gradient_checkpointing and
                       (plan.micro_batch, plan.accumulation) == (4, 2) for plan in plans)
            assert report["status"] == "failed"
            if attempts > 1 and actual[-1] < cap:
                pytest.fail("Final permitted transition did not reach the cap")
            cases += 1
    assert cases == 900


def test_dynamic_host_cap_never_authorizes_a_future_overshoot():
    cases = 0
    for cap in range(1, 5):
        capacity = {"model_limit": cap, "requested_cap": 0, "effective_cap": cap}
        for start, attempts, increment in product(range(cap + 1), range(2, 5), range(1, 3)):
            for limits in product(range(cap + 1), repeat=attempts):
                plans = []
                report = run_recovery(
                    MemoryPlan(start, True),
                    RetryLimits(cap, attempts, increment, swap_strategy="exponential"),
                    _oom_runner(plans, host_limits=limits, capacity=capacity),
                    record=lambda value: None, capacity=capacity,
                )
                actual = [plan.blocks_to_swap for plan in plans]
                observed = limits[:len(actual)]
                assert report["effective_cap"] == min(cap, *observed)
                assert len(actual) <= min(attempts, cap - start + 1)
                assert all(a < b for a, b in zip(actual, actual[1:]))
                assert all(plan.gradient_checkpointing for plan in plans)
                for index, value in enumerate(actual):
                    prior_cap = min(cap, *observed[:index]) if index else cap
                    assert value <= prior_cap
                    if value > min(cap, *observed[:index + 1]):
                        assert index == len(actual) - 1
                        assert report["attempts"][-1]["stop_reason"] == "host_limit"
                cases += 1
    assert cases == 11_252


def test_decreasing_host_cap_can_be_below_an_already_started_plan():
    capacity = {"model_limit": 8, "requested_cap": 0, "effective_cap": 8}
    plans = []
    report = run_recovery(
        MemoryPlan(3, True), RetryLimits(8, 4, 1, swap_strategy="exponential"),
        _oom_runner(plans, host_limits=(2, 8, 8, 8), capacity=capacity),
        record=lambda value: None, capacity=capacity,
    )
    assert [plan.blocks_to_swap for plan in plans] == [3]
    assert report["effective_cap"] == 2
    assert report["attempts"][0]["stop_reason"] == "host_limit"


def test_dense_host_limit_matches_exhaustive_integer_budget():
    cases = 0
    for count in range(3, 6):
        for sizes in product(range(1, 4), repeat=count):
            for reserve in (0, 2):
                base = reserve + 2 * max(sizes) + sum(sizes)
                for available in range(base - 2, base + sum(sizes) + 2):
                    feasible = [swap for swap in range(1, count - 1)
                                if _dense_feasible(sizes, available, reserve, swap)]
                    expected = max(feasible, default=0)
                    assert host_swap_limit(list(sizes), available, reserve) == expected
                    if expected:
                        assert all(_dense_feasible(sizes, available, reserve, swap)
                                   for swap in range(1, expected + 1))
                    cases += 1
    assert cases == 9288


def test_dense_host_threshold_and_resource_monotonicity():
    cases = 0
    for count in range(3, 6):
        for sizes in product(range(1, 4), repeat=count):
            threshold = 2 * max(sizes) + sum(sizes) + sizes[-1]
            assert host_swap_limit(list(sizes), threshold - 1, 0) == 0
            assert host_swap_limit(list(sizes), threshold, 0) >= 1
            values = [host_swap_limit(list(sizes), budget, 0)
                      for budget in range(threshold - 1, threshold + sum(sizes) + 1)]
            assert values == sorted(values)
            for index in range(count):
                larger = list(sizes)
                larger[index] += 1
                for budget in range(threshold - 1, threshold + sum(sizes) + 1):
                    assert host_swap_limit(larger, budget, 0) <= host_swap_limit(
                        list(sizes), budget, 0)
            cases += 1
    assert cases == 351


def test_zero_host_limit_is_only_a_no_positive_swap_sentinel():
    sizes = [2, 3, 4]
    assert host_swap_limit(sizes, available=0, reserve=0) == 0
    assert sum(sizes) + 2 * max(sizes) > 0
    assert not _dense_feasible(sizes, available=0, reserve=0, count=1)


def test_cpu_linear_inventory_uses_actual_mixed_dtype_weight_bytes():
    blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(4)])
    blocks[0].half()
    blocks[2].bfloat16()
    model = SimpleNamespace(blocks=blocks)
    expected = [block.weight.numel() * block.weight.element_size() for block in blocks]
    assert expected == [8, 16, 8, 16]
    assert model_block_bytes(model, "anima") == expected
    threshold = 2 * max(expected) + sum(expected) + expected[-1]
    assert host_swap_limit(expected, threshold, 0) == 1


def test_precision_hash_excludes_memory_controls_but_detects_precision_drift(monkeypatch):
    monkeypatch.setattr(contract, "precision_environment", lambda: {"device": "fixed-cpu"})
    args = Namespace(adaptive_precision="fp16_fp32", adaptive_fp32_modules=["blocks.0.*"],
                     adaptive_loss_scale=1024, model_family="anima", attn_mode="torch")
    frozen = contract.precision_contract_id(args)
    args.blocks_to_swap = 7
    args.gradient_checkpointing = True
    args.train_batch_size = 1
    args.gradient_accumulation_steps = 8
    args.adaptive_oom_retry_max_swap = 26
    args._adaptive_swap_capacity = {"effective_cap": 7}
    assert contract.precision_contract_id(args) == frozen
    args.adaptive_fp32_modules = ["blocks.1.*"]
    assert contract.precision_contract_id(args) != frozen


@pytest.mark.parametrize("stage,host_limit,expected_reason", [
    ("model_load", None, "attempts_exhausted"),
    ("forward", None, "missing_host_budget"),
])
def test_unknown_host_budget_only_allows_model_load_oom(
    stage, host_limit, expected_reason
):
    capacity = {"model_limit": 4, "requested_cap": 0, "effective_cap": 4}

    def runner(plan, *, attempt, resume):
        return {"status": "cuda_oom", "stage": stage, "optimizer_started": False,
                **capacity, "host_limit": host_limit}

    report = run_recovery(MemoryPlan(1, True), RetryLimits(4, 1), runner,
                          record=lambda value: None, capacity=capacity)
    assert report["attempts"][0]["stop_reason"] == expected_reason


def test_optimizer_started_without_committed_checkpoint_cannot_retry():
    plans = []

    def runner(plan, *, attempt, resume):
        plans.append(plan)
        return {"status": "cuda_oom", "stage": "backward",
                "optimizer_started": True, "committed_checkpoint": None}

    report = run_recovery(MemoryPlan(0, True), RetryLimits(4, 4), runner,
                          record=lambda value: None)
    assert len(plans) == 1
    assert report["attempts"][0]["stop_reason"] == "no_committed_checkpoint"


def test_failed_candidates_cannot_certify_unmeasured_counts():
    # Deliberately non-monotone fake outcomes: the successful count is skipped
    # by the exponential retry schedule. This is no claim about CUDA behavior.
    success = {4}
    observed = []

    def runner(plan, *, attempt, resume):
        observed.append(plan.blocks_to_swap)
        return {"status": "ok" if plan.blocks_to_swap in success else "cuda_oom",
                "stage": "model_load", "optimizer_started": False}

    report = run_recovery(MemoryPlan(), RetryLimits(8, 4, 2, swap_strategy="exponential"),
                          runner, record=lambda value: None)
    assert observed == [0, 2, 6, 8]
    assert report["status"] == "failed" and 4 not in observed and 4 in success
    search = SwapSearch(8)
    search.observe(Measurement(8, safe=False))
    assert search.next_candidate() is None
    assert 4 in success  # Failure at maximum cannot prove failure below it.


def test_safe_runtime_is_not_monotone_in_swap_count():
    # Resource safety and throughput are separate assertions.
    seconds = {2: 12.0, 4: 9.0, 6: 11.0}
    assert seconds[4] < seconds[2] and seconds[4] < seconds[6]
    assert not (seconds[2] <= seconds[4] <= seconds[6])
    assert not (seconds[2] >= seconds[4] >= seconds[6])
