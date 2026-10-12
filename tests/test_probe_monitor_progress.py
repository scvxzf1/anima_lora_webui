"""Progress compatibility across probe histories and production reports."""

import json

from bench.adaptive_runtime.probe_monitor import _progress


def test_production_progress_counts_effective_updates(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"updates": {"attempted": 4, "effective": 3,
                                           "skipped": 1, "total_global_step": 23,
                                           "extra": True}}))
    assert _progress(path)["completed_steps"] == 3


def test_legacy_probe_progress_counts_history(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"updates": [{"step": 1}, {"step": 2}]}))
    assert _progress(path)["completed_steps"] == 2
