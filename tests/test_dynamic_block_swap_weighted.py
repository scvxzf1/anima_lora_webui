import json

import pytest

from library.training.auto_block_swap.online_policy import OnlineSwapPolicy


def _run(baseline, candidate, after):
    policy = OnlineSwapPolicy(26, interval=100, warmup=0)
    policy.known_shapes = {"common", "rare"}
    decisions = []
    step = 0
    for counts, seconds in ((baseline, 10), (candidate, None), (after, 10)):
        for shape in ["common"] * counts[0] + ["rare"] * counts[1]:
            step += 1
            value = seconds if seconds is not None else (12 if shape == "common" else 1)
            decision = policy.observe(shape, value, step=step, remaining=10000,
                                      promotion_floor=0)
            if decision:
                decisions.append(decision.reason)
                policy.changed(decision.blocks)
    return policy, decisions


def test_imbalanced_load_rejects_equal_weight_false_gain():
    policy, decisions = _run((90, 10), (10, 90), (10, 90))
    assert decisions == ["explore", "paired_reference"]
    assert policy.last_verdict["candidate_seconds"] == pytest.approx(10.9)
    assert policy.last_verdict["shape_weights"] == {"common": 0.9, "rare": 0.1}
    assert policy.last_verdict["baseline_shape_counts"] == {"common": 90, "rare": 10}
    assert not policy.last_verdict["accepted"]
    json.dumps(policy.last_verdict, allow_nan=False)


def test_imbalanced_load_accepts_real_gain_despite_equal_weight_slowdown():
    policy = OnlineSwapPolicy(26, interval=100, warmup=0)
    policy.known_shapes = {"common", "rare"}
    decisions = []
    step = 0
    for common, rare, timings in ((90, 10, (10, 10)), (10, 90, (8, 20)), (10, 90, (10, 10))):
        for shape in ["common"] * common + ["rare"] * rare:
            step += 1
            decision = policy.observe(shape, timings[shape == "rare"], step=step,
                                      remaining=10000, promotion_floor=0)
            if decision:
                decisions.append(decision.reason)
                policy.changed(decision.blocks)
    assert decisions == ["explore", "paired_reference", "accepted"]
    assert policy.last_verdict["candidate_seconds"] == pytest.approx(9.2)
    assert policy.last_verdict["shape_weights"] == {"common": 0.9, "rare": 0.1}


@pytest.mark.parametrize("reset", ["reset_window", "abandon"])
def test_new_baseline_does_not_reuse_old_counts(reset):
    policy = OnlineSwapPolicy(26, interval=4, warmup=0)
    policy.known_shapes = {"common", "rare"}
    policy.baseline_counts.update(common=90, rare=10)
    if reset == "abandon":
        policy.trial_weights = {"common": 0.9, "rare": 0.1}
        policy.trial_counts = {"common": 90, "rare": 10}
        policy.abandon(0, cooldown=0)
        assert not policy.trial_weights and not policy.trial_counts
    else:
        policy.reset_window()
    for step, shape in enumerate(["common", "common", "rare", "rare"], 1):
        policy.observe(shape, 10, step=step, remaining=1000, promotion_floor=0)
    assert policy.trial_weights == {"common": 0.5, "rare": 0.5}
    assert policy.trial_counts == {"common": 2, "rare": 2}


def test_consecutive_trials_freeze_only_their_own_baseline_counts():
    policy, _ = _run((90, 10), (10, 90), (10, 90))
    assert not policy.baseline_counts
    assert not policy.trial_counts and not policy.trial_weights
    first_verdict = policy.last_verdict.copy()
    policy.changed(24)
    step = policy.cooldown_until
    for counts, timings in (((50, 50), (10, 10)), ((90, 10), (8, 8)), ((10, 90), (10, 10))):
        for shape in ["common"] * counts[0] + ["rare"] * counts[1]:
            step += 1
            decision = policy.observe(shape, timings[shape == "rare"], step=step,
                                      remaining=10000, promotion_floor=0)
            if decision:
                policy.changed(decision.blocks)
    assert policy.last_verdict["shape_weights"] == {"common": 0.5, "rare": 0.5}
    assert policy.last_verdict["baseline_shape_counts"] == {"common": 50, "rare": 50}
    assert policy.last_verdict["accepted"]
    assert first_verdict["shape_weights"] == {"common": 0.9, "rare": 0.1}
