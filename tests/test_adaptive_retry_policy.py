import pytest

from library.training.adaptive_runtime.retry import MemoryPlan, RetryLimits, next_plan, run_recovery


def test_exponential_retry_reaches_final_cap_in_four_attempts():
    plans = []

    def runner(plan, **kwargs):
        plans.append(plan.blocks_to_swap)
        return {"status": "cuda_oom", "stage": "model_load", "optimizer_started": False}

    result = run_recovery(MemoryPlan(), RetryLimits(38, 4, 2, swap_strategy="exponential"),
                          runner, record=lambda value: None)
    assert plans == [0, 2, 6, 38]
    assert result["attempts"][-1]["stop_reason"] == "attempts_exhausted"
    assert "next_reason" not in result["attempts"][-1]


def test_linear_default_keeps_bench_behavior():
    plan, reason = next_plan(MemoryPlan(0, True), RetryLimits(10, 4, 2), stage="forward")
    assert plan.blocks_to_swap == 2 and reason == "increase_block_swap"
    plan, _ = next_plan(plan, RetryLimits(10, 4, 2), stage="forward")
    assert plan.blocks_to_swap == 4


def test_one_attempt_never_advertises_next_plan():
    result = run_recovery(MemoryPlan(), RetryLimits(26, 1, swap_strategy="exponential"),
                          lambda *a, **k: {"status": "cuda_oom", "stage": "backward",
                                          "optimizer_started": False},
                          record=lambda value: None)
    assert len(result["attempts"]) == 1
    assert result["attempts"][0]["stop_reason"] == "attempts_exhausted"


def test_host_cap_only_decreases_and_prevents_overshoot():
    capacity = {"model_limit": 38, "requested_cap": 0, "effective_cap": 38}
    plans = []

    def runner(plan, **kwargs):
        plans.append(plan.blocks_to_swap)
        return {"status": "cuda_oom", "stage": "forward", "optimizer_started": False,
                "model_limit": 38, "requested_cap": 0,
                "effective_cap": 7 if len(plans) == 1 else 12,
                "host_limit": 7 if len(plans) == 1 else 12}

    result = run_recovery(MemoryPlan(), RetryLimits(38, 4, 2, swap_strategy="exponential"),
                          runner, record=lambda value: None, capacity=capacity)
    assert plans == [0, 2, 6, 7]
    assert result["effective_cap"] == 7
    assert result["cap_reason"] == "measured_host_budget"
    assert result["attempts"][0]["cap_adjustment"] == "measured_host_budget"


def test_unknown_or_drifted_capacity_fails_closed():
    capacity = {"model_limit": 26, "requested_cap": 0, "effective_cap": 26}
    for fields in ({}, {"model_limit": 28, "requested_cap": 0, "effective_cap": 26}):
        report = run_recovery(MemoryPlan(), RetryLimits(26),
                              lambda *a, **k: {"status": "cuda_oom", "stage": "model_load",
                                                  "optimizer_started": False, **fields},
                              record=lambda value: None, capacity=capacity)
        assert len(report["attempts"]) == 1
        assert report["attempts"][0]["stop_reason"] == "model_capacity_mismatch"


def test_preload_capacity_allows_model_load_oom_without_host_estimate():
    capacity = {"model_limit": 26, "requested_cap": 0, "effective_cap": 26}
    plans = []

    def runner(plan, **kwargs):
        plans.append(plan.blocks_to_swap)
        return {"status": "cuda_oom", "stage": "model_load", "optimizer_started": False,
                **capacity, "host_limit": None, "host_budget_reason": "not_measured"}

    report = run_recovery(MemoryPlan(), RetryLimits(26, 2, swap_strategy="exponential"),
                          runner, record=lambda value: None, capacity=capacity)
    assert plans == [0, 26]
    assert report["effective_cap"] == 26


def test_host_limit_below_current_request_stops():
    capacity = {"model_limit": 26, "requested_cap": 0, "effective_cap": 26}
    report = run_recovery(MemoryPlan(10), RetryLimits(26),
                          lambda *a, **k: {"status": "host_limit", **capacity,
                                              "effective_cap": 8, "host_limit": 8},
                          record=lambda value: None, capacity=capacity)
    assert report["effective_cap"] == 8
    assert report["attempts"][0]["stop_reason"] == "host_limit"


@pytest.mark.parametrize("status", ["ok", "cuda_oom"])
def test_positive_swap_requires_measured_host_budget_after_loading(status):
    capacity = {"model_limit": 26, "requested_cap": 0, "effective_cap": 26}
    report = run_recovery(
        MemoryPlan(2, True), RetryLimits(26),
        lambda *a, **k: {"status": status, "stage": "forward",
                        "optimizer_started": False, **capacity, "host_limit": None},
        record=lambda value: None, capacity=capacity,
    )
    assert report["status"] == "failed"
    assert len(report["attempts"]) == 1
    assert report["attempts"][0]["stop_reason"] == "missing_host_budget"


def test_positive_swap_with_measured_host_budget_can_succeed():
    capacity = {"model_limit": 26, "requested_cap": 0, "effective_cap": 26}
    report = run_recovery(
        MemoryPlan(2, True), RetryLimits(26),
        lambda *a, **k: {"status": "ok", **capacity, "host_limit": 26},
        record=lambda value: None, capacity=capacity,
    )
    assert report["status"] == "ok"
