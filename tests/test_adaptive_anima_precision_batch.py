"""Candidate budgets, measured refinement, and batch worker failure boundaries."""

import json
from types import SimpleNamespace

import pytest

from bench.adaptive_runtime import anima_precision_batch as batch
from bench.adaptive_runtime import anima_precision_batch_policy as policy


def comparison(error=0.01, passes=True):
    metric = {"relative_l2": error}
    return {
        "status": "compared",
        "scope": "fixed_pre_step_state_and_inputs_not_quality_certificate",
        "within_experimental_tolerances": passes,
        "cases": [
            {
                "prediction": metric,
                "input_gradient": metric,
                "adapter_gradient": metric,
                "within_experimental_tolerances": passes,
            }
        ],
    }


def history(choices, passing=()):
    return [
        {"candidate": c, "comparison": comparison(0.001 * (i + 1), i in passing)}
        for i, c in enumerate(choices)
    ]


def test_measured_refinement_combination_and_budget():
    initial = policy.next_stage([])
    assert [c["blocks"] for c in initial] == [list(q) for q in policy.QUARTERS]
    rejected = history(initial)
    refined = policy.next_stage(rejected)
    assert refined[0]["blocks"] == [0, 1, 2]
    assert refined[1]["blocks"] == [3, 4, 5, 6]
    combined = policy.next_stage(history(initial, passing=(0, 1)))
    assert combined[0]["blocks"] == list(range(14))
    h = rejected + history(refined)
    confirmed = policy.next_stage(h)
    assert len(confirmed) <= 4
    h += history(confirmed)
    assert len({tuple(item["candidate"]["blocks"]) for item in h}) == 12
    assert policy.next_stage(h) == []
    assert (
        policy.storage_estimate([0], {"blocks.0.mlp.layer2.weight": 100, "other": 100})[
            "weight_payload_reduction_fraction"
        ]
        == 0.25
    )


def args(tmp_path, **changes):
    weights, inputs = tmp_path / "weights", tmp_path / "inputs"
    weights.write_bytes(b"w")
    inputs.write_bytes(b"i")
    values = dict(
        weights=weights,
        inputs=inputs,
        output_dir=tmp_path / "batch",
        gpu_uuid="GPU-test",
        swap=20,
        steps=1,
        single_sigma=0.5,
        loss_scale=1024,
        worker_timeout=150,
        batch_timeout=480,
        max_new_candidates=4,
        continue_run=False,
    )
    return SimpleNamespace(**(values | changes))


def test_commands_capture_same_initial_update_and_exact_targets(tmp_path):
    cfg = args(tmp_path)
    cmd = batch.command(
        cfg,
        tmp_path / "worker",
        "fp16-islands",
        reference=tmp_path / "reference",
        blocks=[7, 8],
    )
    assert "--capture-update" in cmd
    assert cmd[cmd.index("--single-sigma") + 1] == "0.5"
    assert [
        cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--fp16-only-pattern"
    ] == ["blocks.7.mlp.layer2", "blocks.8.mlp.layer2"]
    assert "--fp32-pattern" not in cmd


def test_failed_worker_stops_segment_and_preserves_summary(tmp_path, monkeypatch):
    cfg = args(tmp_path)
    monkeypatch.setattr(batch.policy, "weight_inventory", lambda _: {})
    calls = []

    def monitor(argv, directory, uuid, timeout):
        calls.append(directory.name)
        return {"status": "timeout", "returncode": -15}

    result = batch.run(cfg, gpu_check=lambda _: {"compute_cap": "8.6"}, monitor=monitor)
    assert result["status"] == "failed"
    assert calls == ["reference"] and not result["candidates"]
    assert (
        json.loads((cfg.output_dir / "summary.json").read_text())["status"] == "failed"
    )


def test_continuation_rejects_changed_evidence_before_worker(tmp_path, monkeypatch):
    cfg = args(tmp_path)
    cfg.output_dir.mkdir()
    monkeypatch.setattr(batch.policy, "weight_inventory", lambda _: {})
    cfg.continue_run = True
    (cfg.output_dir / "environment.json").write_text(
        json.dumps({"identity": batch.identities(cfg)})
    )
    (cfg.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "scope": batch.SCOPE,
                "status": "screening",
                "production_ready": False,
                "controls_complete": True,
                "controls": [{"directory": "missing", "evidence_sha256": {}}],
                "candidates": [],
            }
        )
    )
    with pytest.raises(FileNotFoundError):
        batch.run(
            cfg,
            gpu_check=lambda _: {},
            monitor=lambda *args: pytest.fail("must not launch"),
        )
