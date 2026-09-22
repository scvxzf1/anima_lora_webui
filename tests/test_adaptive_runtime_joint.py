import json
import sys
from types import SimpleNamespace

import pytest

from bench.adaptive_runtime.search_dit import probe_command
from library.training.adaptive_runtime.joint import discover_preflight_plan
from library.training.adaptive_runtime.retry import MemoryPlan, RetryLimits


def nonfinite(name="projection", **overrides):
    return {"status": "error", "failure_kind": "nonfinite", "disposable_probe": True,
            "first_nonfinite": {"module": name, "type": "Linear",
                                "inputs": [{"finite": True}], "outputs": [{"finite": False}]},
            **overrides}


def search(results, **kwargs):
    calls = []

    def runner(plan, promoted, *, attempt):
        calls.append((plan.blocks_to_swap, promoted, attempt))
        return results[attempt]

    report = discover_preflight_plan(
        MemoryPlan(blocks_to_swap=20, gradient_checkpointing=True),
        RetryLimits(max_blocks=28, max_attempts=kwargs.pop("max_attempts", 6), swap_increment=4),
        runner, record=lambda r: None, **kwargs,
    )
    return report, calls


def test_promotion_then_oom_preserves_promoted_module():
    report, calls = search([nonfinite(),
                            {"status": "cuda_oom", "stage": "forward", "disposable_probe": True},
                            {"status": "ok", "disposable_probe": True,
                             "fp32_modules": ["seeded", "projection"]}])
    assert calls == [(20, (), 0), (20, ("projection",), 1), (24, ("projection",), 2)]
    assert report["status"] == "finite_only"
    assert report["selected"]["blocks_to_swap"] == 24
    assert report["precision_calibrated"] is False
    assert report["production_ready"] is False
    assert report["resolved_fp32_modules"] == ["seeded", "projection"]


def test_oom_then_promotion_keeps_memory_plan_and_discards_only_probe_updates():
    report, calls = search([
        {"status": "cuda_oom", "stage": "backward", "disposable_probe": True,
         "optimizer_started": True, "updates": [{"step": 1}]},
        nonfinite(optimizer_started=True, updates=[{"step": 1}]),
        {"status": "ok", "disposable_probe": True, "fp32_modules": ["projection"]},
    ])
    assert calls == [(20, (), 0), (24, (), 1), (24, ("projection",), 2)]
    assert report["calibration_only"] is True


@pytest.mark.parametrize("result", [None, {"status": "ok"}, nonfinite(disposable_probe=False),
                                    {"status": "host_limit"}, {"status": "timeout"}])
def test_never_restarts_unmarked_training_or_infrastructure_failure(result):
    report, calls = search([result])
    assert report["status"] == "failed" and len(calls) == 1


@pytest.mark.parametrize("extra", [{"fp32_modules": ["projection"]},
                                   {"failure_kind": "runtime"}, {"fp32_modules": None}])
def test_no_promotion_for_already_fp32_or_invalid_failure(extra):
    report, calls = search([nonfinite(**extra)])
    assert len(calls) == 1 and report["fp32_modules"] == []


@pytest.mark.parametrize("limits", [{"max_attempts": 1}, {"max_promotions": 0},
                                    {"allow_precision_promotions": False}])
def test_bounds_do_not_publish_untested_promotion(limits):
    report, calls = search([nonfinite()], **limits)
    assert len(calls) == 1 and report["fp32_modules"] == []


def test_no_repeated_numerical_candidate():
    report, calls = search([nonfinite(), nonfinite()])
    assert len(calls) == 2 and report["status"] == "failed"


def test_worker_cannot_silently_ignore_precision_promotion():
    report, _ = search([nonfinite(),
                        {"status": "ok", "disposable_probe": True, "fp32_modules": []}])
    assert report["status"] == "failed"
    assert "selected" not in report
    assert report["attempts"][-1]["stop_reason"] == "precision_profile_not_applied"


def test_oom_never_promotes_even_with_stale_trace():
    report, calls = search([nonfinite(status="cuda_oom", stage="optimizer")])
    assert len(calls) == 1 and report["fp32_modules"] == []
    assert report["attempts"][0]["stop_reason"] == "no_authorized_memory_adjustment"


@pytest.mark.parametrize("status", ["host_limit", "timeout", "cancelled"])
def test_marked_infrastructure_failure_with_stale_trace_cannot_promote(status):
    report, calls = search([nonfinite(status=status)])
    assert len(calls) == 1 and report["fp32_modules"] == []


def test_joint_command_is_disposable_and_has_no_checkpoint(tmp_path):
    request = tmp_path / "request.json"
    payload = {"plan": {"blocks_to_swap": 24, "gradient_checkpointing": True,
                        "micro_batch": 1, "accumulation": 1}, "resume": None}
    request.write_text(json.dumps(payload))
    args = SimpleNamespace(model_family="z_image", weights="weights", inputs="inputs",
                           fp32_pattern=["*.w2"], memory_limit_gib=6.3)
    command = probe_command(args, "fp16-islands", ("projection",), request)
    assert "--disposable-probe" in command and "--trace-numerics" in command
    assert "--checkpoint-every-step" not in command and "--resume" not in command
    assert command[command.index("--swap") + 1] == "24"
    assert command.count("--fp32-pattern") == 2
    payload["resume"] = "checkpoint.pt"
    request.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="must not resume"):
        probe_command(args, "fp16-islands", (), request)


@pytest.mark.parametrize("family", ["krea", "z_image"])
def test_disposable_checkpoint_rejected_before_creating_output(tmp_path, monkeypatch, family):
    from bench.adaptive_runtime import probe_krea_train, probe_z_image_train

    module = probe_krea_train if family == "krea" else probe_z_image_train
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["probe", "--weights", "unused", "--inputs", "unused",
                                     "--output", str(output), "--precision", "fp16-islands",
                                     "--disposable-probe", "--checkpoint-every-step"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    assert not output.exists()
