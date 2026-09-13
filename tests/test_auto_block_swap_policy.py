import pytest

from library.training.auto_block_swap.policy import (
    Measurement,
    SwapSearch,
    host_swap_limit,
)


def test_descending_growth_and_failure_bisection():
    search = SwapSearch(26, max_trials=10)
    for blocks, safe in [
        (26, True),
        (24, True),
        (20, True),
        (12, False),
        (16, False),
        (18, True),
        (17, False),
    ]:
        assert search.next_candidate() == blocks
        search.observe(Measurement(blocks, safe, 10 + blocks / 10))
    assert search.next_candidate() is None
    assert search.select() == 20  # Within 3% of swap18, with more headroom.


def test_initial_failure_never_selects_untested_fallback():
    search = SwapSearch(26)
    search.observe(Measurement(26, False))
    assert search.next_candidate() is None
    with pytest.raises(RuntimeError, match="no resource-safe"):
        search.select()


def test_zero_and_trial_budget():
    search = SwapSearch(0)
    search.observe(Measurement(0, True, 1))
    assert search.next_candidate() is None
    assert search.select() == 0
    limited = SwapSearch(26, max_trials=1)
    limited.observe(Measurement(26, True, 1))
    assert limited.next_candidate() is None
    assert limited.select() == 26


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_timing(seconds):
    with pytest.raises(ValueError, match="timing"):
        SwapSearch(3).observe(Measurement(3, True, seconds))


def test_rejects_wrong_candidate():
    with pytest.raises(ValueError, match="candidate"):
        SwapSearch(3).observe(Measurement(2, True, 1))


def test_host_budget_includes_all_masters_and_scratch():
    # 6 blocks, 60 fixed masters + 20 scratch + 20 reserve.
    assert host_swap_limit([10] * 6, 140, 20) == 4
    assert host_swap_limit([10] * 6, 120, 20) == 2
    assert host_swap_limit([10] * 6, 99, 20) == 0
    assert host_swap_limit([1, 2, 3, 4], 28, 2) == 2


@pytest.mark.parametrize("sizes", [[], [10, 10], [10, 0, 10], [10, -1, 10]])
def test_invalid_inventory(sizes):
    with pytest.raises(ValueError):
        host_swap_limit(sizes, 100, 10)
