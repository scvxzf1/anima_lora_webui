import pytest

from library.training.auto_block_swap.online_policy import OnlineSwapPolicy


def run(policy, timings, *, start=0, floor=0):
    decisions = []
    for step, seconds in enumerate(timings, start + 1):
        policy.guard("shape", step=step, safe_count=floor)
        decision = policy.observe(
            "shape", seconds, step=step, remaining=1000, promotion_floor=floor
        )
        if decision:
            decisions.append(decision)
            policy.changed(decision.blocks)
    return decisions


def test_paired_search_accepts_real_gain_and_accelerates_stride():
    policy = OnlineSwapPolicy(26, interval=4, warmup=0)
    decisions = run(policy, [10] * 4 + [8] * 4 + [10] * 4)
    assert [(d.blocks, d.reason) for d in decisions] == [
        (24, "explore"),
        (26, "paired_reference"),
        (24, "accepted"),
    ]
    assert policy.stride == 4
    assert policy.last_verdict["accepted"]
    assert run(policy, [8] * 4, start=12)[0].blocks == 20


@pytest.mark.parametrize("candidate,after", [(9.8, 10), (12, 10), (8, 14)])
def test_noise_slowdown_or_reference_drift_rejects(candidate, after):
    policy = OnlineSwapPolicy(26, interval=4, warmup=0)
    decisions = run(policy, [10] * 4 + [candidate] * 4 + [after] * 4)
    assert [d.blocks for d in decisions] == [24, 26]
    assert not policy.last_verdict["accepted"]
    assert not run(policy, [10] * 20, start=12)


def test_pressure_overrides_exploration_and_release_allows_retry():
    policy = OnlineSwapPolicy(26, interval=4, warmup=0)
    run(policy, [10] * 4)
    assert policy.current == 24
    decision = policy.guard("shape", step=5, safe_count=25)
    assert decision.blocks == 25 and decision.reason == "gpu_pressure"
    assert policy.phase == "baseline" and policy.rejected_below == -1
    policy.changed(decision.blocks)
    assert run(policy, [10] * 4, start=40)[0].blocks == 23
    with pytest.raises(RuntimeError, match="maximum swap"):
        policy.guard("new", step=6, safe_count=27)


def test_unseen_resolution_rewarms_at_maximum():
    policy = OnlineSwapPolicy(26, interval=4, warmup=0)
    run(policy, [10] * 4)
    decision = policy.guard("larger", step=5, safe_count=0)
    assert decision.blocks == 26 and decision.reason == "new_shape"
    policy.changed(26)
    assert policy.phase == "baseline" and policy.seen == {}


def test_warmup_and_short_remaining_run_do_not_trigger_exploration():
    policy = OnlineSwapPolicy(26, interval=4, warmup=2)
    policy.guard("shape", step=0, safe_count=0)
    for step in range(1, 7):
        assert (
            policy.observe("shape", 10, step=step, remaining=4, promotion_floor=0)
            is None
        )
    assert len(policy.samples["shape"]) == 4


def test_windows_match_shapes_not_their_frequency():
    policy = OnlineSwapPolicy(26, interval=4, warmup=0)
    policy.known_shapes = {"small", "large"}
    for step, shape in enumerate(["small", "small", "small", "large"]):
        assert (
            policy.observe(shape, 1, step=step, remaining=1000, promotion_floor=0)
            is None
        )
    assert (
        policy.observe("large", 1, step=5, remaining=1000, promotion_floor=0).blocks
        == 24
    )


def test_compile_warmup_cost_prevents_unamortized_acceptance():
    policy = OnlineSwapPolicy(26, interval=4, warmup=1)
    decisions = run(policy, [10] * 5 + [10000] + [8] * 4 + [10] * 5)
    assert [decision.blocks for decision in decisions] == [24, 26]
    assert not policy.last_verdict["accepted"]
    assert policy.last_verdict["warmup_cost_seconds"] == 9992


def test_expensive_migrations_prevent_acceptance():
    policy = OnlineSwapPolicy(26, interval=4, warmup=0)
    for step, seconds in enumerate([10] * 4 + [8] * 4 + [10] * 4, 1):
        decision = policy.observe(
            "shape",
            seconds,
            step=step,
            remaining=100,
            promotion_floor=0,
            switch_seconds=500,
        )
        if decision:
            policy.changed(decision.blocks, seconds=500)
    assert not policy.last_verdict["accepted"]


def test_small_plateau_does_not_prevent_wider_exploration():
    policy = OnlineSwapPolicy(26, interval=4, warmup=0)
    run(policy, [10] * 4 + [9.9] * 4 + [10] * 4)
    assert policy.current == 26 and policy.stride == 4
    assert run(policy, [10] * 4, start=44)[0].blocks == 22


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_invalid_timing_is_rejected(seconds):
    policy = OnlineSwapPolicy(26)
    with pytest.raises(ValueError, match="finite positive"):
        policy.observe("shape", seconds, step=1, remaining=1000, promotion_floor=0)


def test_migration_refusal_expires_and_is_not_a_permanent_lock():
    policy = OnlineSwapPolicy(26, interval=4, warmup=0)
    policy.rejected_below, policy.retry_after = 24, 64
    assert run(policy, [10] * 4)[0].blocks == 25
    policy.changed(26)
    policy.abandon(64, cooldown=0)
    assert run(policy, [10] * 4, start=64)[0].blocks == 24
    assert policy.rejected_below == -1


def test_new_shape_discards_inflight_trial_before_acceptance():
    policy = OnlineSwapPolicy(26, interval=4, warmup=0)
    run(policy, [10] * 4 + [8] * 4)
    assert policy.phase == "recheck"
    assert policy.guard("new", step=9, safe_count=0) is None
    assert policy.phase == "baseline"
    assert policy.last_verdict is None
    assert policy.observe("new", 5, step=9, remaining=1000, promotion_floor=0) is None
    assert policy.samples == {"new": [5]}


def test_acceptance_verdict_respects_changed_resource_floor():
    policy = OnlineSwapPolicy(26, interval=4, warmup=0)
    run(policy, [10] * 4 + [8] * 4)
    assert not run(policy, [10] * 4, start=8, floor=25)
    assert not policy.last_verdict["accepted"]
