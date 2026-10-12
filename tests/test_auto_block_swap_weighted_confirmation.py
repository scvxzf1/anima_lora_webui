import json
from types import SimpleNamespace

import pytest

from library.training.auto_block_swap import coordinator, probe


def test_representative_frequency_keeps_datasets_separate():
    batch = lambda bucket, index: SimpleNamespace(
        bucket_index=bucket, bucket_batch_size=2, batch_index=index
    )
    class Dataset:
        def __init__(self, batches):
            self.buckets_indices = batches
            self.bucket_manager = SimpleNamespace(
                resos=[(512, 512), (1024, 1024)],
                buckets=[list(range(20)), list(range(20))],
            )

        def __len__(self):
            return len(self.buckets_indices)

    group = SimpleNamespace(datasets=[
        Dataset([batch(0, i) for i in range(9)] + [batch(1, 0)]),
        Dataset([batch(0, 0), batch(0, 1)]),
    ])
    cases = probe.representative_batches(group)
    assert {(c["dataset"], c["width"]): c["weight"] for c in cases} == {
        (0, 512): 9, (0, 1024): 1, (1, 512): 2,
    }
    assert probe.weighted_case_seconds(cases, [10, 20, 30]) == pytest.approx(
        sum(c["weight"] * t for c, t in zip(cases, [10, 20, 30])) / 12
    )


@pytest.mark.parametrize("weights,times", [([], []), ([0], [1]), ([float('nan')], [1]), ([1], [0]), ([1], [float('inf')])])
def test_invalid_case_weights_or_timings_rejected(weights, times):
    with pytest.raises(ValueError):
        probe.weighted_case_seconds([{"weight": w} for w in weights], times)


def _run_confirmation(tmp_path, confirmed):
    tmp_path.mkdir(exist_ok=True)
    calls = []

    def runner(args, directory, *, timeout):
        calls.append(args)
        if args._auto_swap_probe["inventory"]:
            return {"status": "inventory", "model_swap_limit": 0,
                    "host_swap_limit": 0, "block_count": 2}
        return {"status": "ok", "safe": True,
                "seconds": 10 if len(calls) == 2 else confirmed,
                "headroom": 100, "host_min_available": 10,
                "host_reserve": 1, "swap_io_bytes": 0}

    args = SimpleNamespace(auto_block_swap_timeout=30, auto_block_swap_max_trials=1,
                           auto_block_swap_preference="balanced", seed=42)
    calibrate = coordinator.run_calibration
    return args, lambda: calibrate(args, tmp_path, runner=runner)


@pytest.mark.parametrize("seconds,accepted", [(9, True), (11, True), (8.9, False), (11.1, False), (0, False), (float('nan'), False), (float('inf'), False)])
def test_confirmation_timing_drift_fails_closed(tmp_path, seconds, accepted):
    args, run = _run_confirmation(tmp_path, seconds)
    if accepted:
        assert run() == 0
    else:
        with pytest.raises(RuntimeError, match="confirmation"):
            run()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["confirmation_timing"]["stable"] is accepted
    assert summary["confirmation_timing"]["limit"] == 0.10
    assert summary["confirmation_timing"]["candidate_seconds"] == 10
    assert (summary.get("status") == "selected") is accepted
    assert not hasattr(args, "_auto_swap_resolved")


def test_confirmation_failure_does_not_resolve_parent_configuration(tmp_path, monkeypatch):
    args, run = _run_confirmation(tmp_path / "probe", 15)
    args.auto_block_swap = True
    args.output_dir = str(tmp_path)
    args.blocks_to_swap = 8
    monkeypatch.setattr(coordinator, "_validate_launch", lambda _: None)
    monkeypatch.setattr(coordinator, "run_calibration", lambda *_: run())
    with pytest.raises(RuntimeError, match="timing drift"):
        coordinator.calibrate_if_requested(args)
    assert args.blocks_to_swap == 8
    assert not getattr(args, "_auto_swap_resolved", False)
    failure_path, = (tmp_path / "auto-block-swap").glob("calibration-*/failure.json")
    assert json.loads(failure_path.read_text())["status"] == "failed"
