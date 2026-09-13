import json

import pytest

from scripts.experiments.auto_block_swap_report import (
    compare_losses,
    summarize_hardware,
    summarize_probe,
    summarize_search,
    summarize_updates,
)


def test_inflight_worker_result_is_not_finalized_evidence(tmp_path):
    path = tmp_path / "result.json"
    path.write_text('{"status": "ok", "updates": []}')
    assert summarize_probe(path, tmp_path, []) is None


def test_reused_candidates_and_confirmation_drift_stay_distinct(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "selected": 26,
                "trials": [
                    {
                        "blocks": 26,
                        "status": "ok",
                        "safe": True,
                        "seconds": 10,
                        "elapsed_seconds": 100,
                        "reused_from": "full",
                    }
                ],
                "confirmation": {
                    "status": "ok",
                    "safe": True,
                    "seconds": 15,
                    "elapsed_seconds": 150,
                },
            }
        )
    )
    report = summarize_search(path, tmp_path)
    assert report["confirmation_timing_change_percent"] == 50
    assert report["candidate_count"] == 1
    assert report["fresh_candidates"] == 0
    assert report["logical_candidate_seconds"] == 100
    assert report["fresh_candidate_seconds"] == 0
    assert report["confirmation_seconds"] == 150


def test_timings_weight_cases_equally_and_exclude_cold_updates():
    rows = [
        {
            "case": case,
            "update": i,
            "seconds": seconds,
            "loss": 0.5,
            "peak_allocated": 1024**3,
            "peak_reserved": 2 * 1024**3,
        }
        for case, values in enumerate(([100, 4, 6, 8], [200, 8, 10, 12]))
        for i, seconds in enumerate(values)
    ]
    result = summarize_updates(rows)
    assert result["completed_updates"] == 8
    assert result["warm_seconds"] == 8
    assert result["hot_seconds"] == 9
    assert result["peak_allocated_gib"] == 1
    assert result["cases"][0]["first_seconds"] == 100
    assert summarize_updates([]) == {"completed_updates": 0}


def test_hardware_window_filters_idle_and_unavailable_sensors():
    rows = [
        {
            "time": time,
            "devices": [
                {
                    "utilization.gpu": load,
                    "temperature.gpu": temp,
                    "clocks.sm": "1900",
                    "power.draw": "[N/A]",
                }
            ],
        }
        for time, load, temp in ((1, "100", "60"), (2, "50", "65"), (3, "99", "70"))
    ]
    result = summarize_hardware(rows, 2, 4)
    assert result["active_samples"] == 1
    assert result["temperature.gpu"]["median"] == 70
    assert "power.draw" not in result


def test_loss_comparison_requires_matching_cases_and_updates():
    first = {
        "source": "a",
        "shapes": [{"width": 896, "height": 1200, "batch_size": 1}],
        "cases": [{"updates": 2, "losses": [0.5, 0.6]}],
    }
    second = {**first, "source": "b", "cases": [{"updates": 2, "losses": [0.5, 0.61]}]}
    assert compare_losses(first, second)[
        "max_absolute_loss_difference"
    ] == pytest.approx(0.01)
    with pytest.raises(ValueError, match="shapes"):
        compare_losses(first, {**second, "shapes": []})
    with pytest.raises(ValueError, match="update counts"):
        compare_losses(first, {**second, "cases": [{"updates": 1, "losses": [0.5]}]})
