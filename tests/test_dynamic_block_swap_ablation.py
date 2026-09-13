from scripts.experiments.dynamic_block_swap_ablation import run_ablations


def test_reference_recheck_prevents_drift_false_positive():
    full, ablated = run_ablations()["reference_drift"]
    assert full["accepted"] == 0
    assert ablated["accepted"] == 1


def test_compile_cost_prevents_unamortized_acceptance():
    full, ablated = run_ablations()["compile_cost"]
    assert full["accepted"] == 0
    assert ablated["accepted"] == 1


def test_pressure_guard_prevents_unsafe_updates():
    full, ablated = run_ablations()["pressure"]
    assert full["unsafe_updates"] == 0
    assert ablated["unsafe_updates"] == 4


def test_stride_growth_crosses_small_plateau():
    full, ablated = run_ablations()["plateau"]
    assert full["final_blocks"] <= 18
    assert full["accepted"] >= 1
    assert ablated["final_blocks"] == 26
    assert ablated["accepted"] == 0
    assert {row["blocks"] for row in ablated["switches"]} == {24, 26}
