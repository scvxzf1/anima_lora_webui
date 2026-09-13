import json

import pytest

from scripts.experiments.dynamic_block_swap_report import summarize


def write_rows(path, values):
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def test_report_requires_completion_and_distinguishes_pressure_recovery(tmp_path):
    report = tmp_path / "auto-block-swap/calibration-test"
    report.mkdir(parents=True)
    write_rows(tmp_path / "updates.jsonl", [{
        "step": step, "blocks": 26 if step < 3 else 24,
        "seconds": 10, "peak_allocated": 1024**3, "dynamo_unique_graphs": 2,
    } for step in range(1, 5)])
    write_rows(tmp_path / "pressure.jsonl", [
        {"step": 1, "held_bytes": 1024**3}, {"step": 2, "held_bytes": 0},
    ])
    write_rows(report / "runtime.jsonl", [
        {"event": "switch", "step": 3, "blocks": 24, "reason": "explore"},
        {"event": "end", "step": 4, "completed": False},
    ])
    result = summarize(tmp_path)
    assert result["pressure_recovery_explored"]
    assert result["completion"] is None
    assert result["observed_blocks"] == [24, 26]
    assert result["peak_gib"] == 1
    assert not result["runtime_end"]["completed"]
    assert not result["acceptance"]["passed"]
    assert not result["acceptance"]["pressure_cycle_completed"]


def test_hot_segments_exclude_two_warmups_per_shape(tmp_path):
    report = tmp_path / "auto-block-swap/calibration-test"
    report.mkdir(parents=True)
    write_rows(report / "runtime.jsonl", [
        {"event": "update", "step": step, "blocks": 26,
         "seconds": seconds, "shapes": [[1, 16, 64, 64]], "verdict": None}
        for step, seconds in enumerate([100, 50, 10, 12], 1)
    ])
    result = summarize(tmp_path)
    assert list(result["segments"][0]["hot_seconds_by_shape"].values()) == [11]


def completed_run(root, *, pressure=False):
    (root / "request.json").write_text(json.dumps({
        "mode": "dynamic", "steps": 4, "pressure": pressure,
    }))
    (root / "completion.json").write_text(json.dumps({
        "status": "completed", "finite": True, "steps": 4,
    }))
    write_rows(root / "updates.jsonl", [{
        "step": step, "blocks": 26, "seconds": 1,
        "peak_allocated": 1, "dynamo_unique_graphs": 1,
    } for step in range(1, 5)])
    report = root / "auto-block-swap/calibration-test"
    report.mkdir(parents=True)
    events = [{
        "event": "update", "step": step, "blocks": 26, "seconds": 1,
        "shapes": [[1, 16, 64, 64]],
    } for step in range(1, 5)]
    events.append({"event": "end", "step": 4, "completed": True})
    write_rows(report / "runtime.jsonl", events)
    return report, events


@pytest.mark.parametrize("failure", ["status", "finite", "steps", "runtime", "end_step", "ambiguous"])
def test_report_rejects_invalid_completion_evidence(tmp_path, failure):
    report, events = completed_run(tmp_path)
    if failure in ("status", "finite", "steps"):
        path = tmp_path / "completion.json"
        result = json.loads(path.read_text())
        result[failure] = {"status": "failed", "finite": False, "steps": 3}[failure]
        path.write_text(json.dumps(result))
    elif failure == "runtime":
        events[-1]["completed"] = False
        write_rows(report / "runtime.jsonl", events)
    elif failure == "end_step":
        events[-1]["step"] = 1
        write_rows(report / "runtime.jsonl", events)
    else:
        other = report.with_name("calibration-other")
        other.mkdir()
        write_rows(other / "runtime.jsonl", events)
    assert not summarize(tmp_path)["acceptance"]["passed"]


def test_complete_run_is_not_proof_of_speedup(tmp_path):
    completed_run(tmp_path)
    result = summarize(tmp_path)
    assert result["acceptance"]["passed"]
    assert not result["acceptance"]["causal_speedup_proven"]
    assert not result["decisions"]


def test_pressure_acceptance_requires_adjustment_release_and_reexploration(tmp_path):
    report, events = completed_run(tmp_path, pressure=True)
    write_rows(tmp_path / "pressure.jsonl", [
        {"step": 1, "held_bytes": 1024}, {"step": 2, "held_bytes": 0},
    ])
    assert not summarize(tmp_path)["acceptance"]["passed"]
    events.extend([
        {"event": "switch", "step": 1, "blocks": 26, "reason": "gpu_pressure"},
        {"event": "switch", "step": 3, "blocks": 24, "reason": "explore"},
    ])
    write_rows(report / "runtime.jsonl", events)
    assert summarize(tmp_path)["acceptance"]["passed"]
