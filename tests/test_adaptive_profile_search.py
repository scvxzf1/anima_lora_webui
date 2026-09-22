import copy
import json
from types import SimpleNamespace

import pytest

from bench.adaptive_runtime.search_z_image_precision import command, profiles_for
from bench.adaptive_runtime import search_z_image_precision as cli
from library.training.adaptive_runtime.profile_search import PrecisionProfile, search_profiles
from library.training.adaptive_runtime.retry import MemoryPlan, RetryLimits


def comparison(passed):
    return {"status": "compared", "scope": "fixed_pre_step_state_and_inputs_not_quality_certificate",
            "cases": [{"within_experimental_tolerances": passed}],
            "within_experimental_tolerances": passed}


def success(profile, plan):
    return {"status": "ok", "disposable_probe": True, "precision_profile_resolved": True,
            "precision": profile.precision, "loss_scale": profile.loss_scale,
            "swap": plan.blocks_to_swap, "fp32_modules": ["layer.q"]}


def test_numerical_rejection_then_oom_retry_then_validated_selection():
    profiles = [PrecisionProfile("a", "fp16-islands", ("layer.*",), 128),
                PrecisionProfile("b", "fp16-islands", ("layer.*",), 1024)]
    calls, recorded = [], []

    def runner(profile, plan, *, attempt):
        calls.append((profile.name, plan.blocks_to_swap, attempt))
        if attempt == 1:
            return {"status": "cuda_oom", "disposable_probe": True,
                    "stage": "backward", "optimizer_started": True}
        return success(profile, plan)

    report = search_profiles(MemoryPlan(20, True), RetryLimits(24, 4, 4), profiles, runner,
                             lambda attempt: comparison(attempt == 2),
                             record=lambda r: recorded.append(copy.deepcopy(r)))
    assert calls == [("a", 20, 0), ("b", 20, 1), ("b", 24, 2)]
    assert report["status"] == "probe_validated"
    assert report["selected_profile"]["name"] == "b"
    assert report["production_ready"] is False and report["precision_calibrated"] is False
    assert report["full_model_calibrated"] is False
    assert recorded[-1] == report


@pytest.mark.parametrize("result", [None, {"status": "ok"}, {"status": "host_limit"},
                                    {"status": "timeout"}, {"status": "cancelled"},
                                    {"status": "error", "disposable_probe": True},
                                    {"status": "ok", "disposable_probe": True}])
def test_fail_closed_does_not_try_next_profile(result):
    calls = []

    def runner(profile, plan, *, attempt):
        calls.append(attempt)
        return result

    report = search_profiles(MemoryPlan(0, True), RetryLimits(4),
                             [PrecisionProfile("a", "bf16"), PrecisionProfile("b", "bf16")],
                             runner, lambda _: pytest.fail("must not compare"), record=lambda _: None)
    assert report["status"] == "failed" and calls == [0]


@pytest.mark.parametrize("kind", ["nonfinite", "numerical", "oom"])
def test_exhausted_profiles_are_not_reported_as_calibrated(kind):
    profile = PrecisionProfile("a", "bf16")
    plan = MemoryPlan(4, True)
    result = {"status": "error", "failure_kind": "nonfinite", "disposable_probe": True}
    if kind == "oom":
        result.update(status="cuda_oom", stage="backward")
    elif kind == "numerical":
        result = success(profile, plan)
    report = search_profiles(plan, RetryLimits(4), [profile], lambda *a, **k: result,
                             lambda _: comparison(False), record=lambda _: None)
    assert report["status"] == "failed"
    assert report["attempts"][-1]["stop_reason"] == "profiles_exhausted"


@pytest.mark.parametrize("where", ["runner", "comparator"])
def test_exception_records_terminal_failure(where):
    def fail(*args, **kwargs):
        raise ValueError("corrupt capture")

    profile, plan = PrecisionProfile("a", "bf16"), MemoryPlan(0, True)
    records = []
    report = search_profiles(plan, RetryLimits(4), [profile],
                             fail if where == "runner" else lambda *a, **k: success(profile, plan),
                             fail, record=lambda r: records.append(copy.deepcopy(r)))
    assert report["status"] == "failed"
    assert records[-1]["status"] == "failed"
    assert "corrupt capture" in report["attempts"][-1]["error"]


def test_global_attempt_limit_includes_oom_attempts():
    report = search_profiles(MemoryPlan(0, True), RetryLimits(8, 2, 2),
                             [PrecisionProfile("a", "bf16")],
                             lambda *a, **k: {"status": "cuda_oom", "stage": "forward", "disposable_probe": True},
                             lambda _: pytest.fail("must not compare"), record=lambda _: None)
    assert len(report["attempts"]) == 2
    assert report["status"] == "failed"
    assert report["attempts"][-1]["stop_reason"] == "attempt_limit"


@pytest.mark.parametrize("bad", [None, {}, {**comparison(True), "scope": "finite_only"},
                                 {**comparison(True), "cases": []},
                                 {**comparison(True), "cases": [{"within_experimental_tolerances": False}]}])
def test_no_selection_from_invalid_comparison(bad):
    profile, plan = PrecisionProfile("a", "bf16"), MemoryPlan(0, True)
    report = search_profiles(plan, RetryLimits(4), [profile], lambda *a, **k: success(profile, plan),
                             lambda _: bad, record=lambda _: None)
    assert report["status"] == "failed"


def test_profile_order_hardware_and_fp32_control_is_opt_in():
    assert [p.name for p in profiles_for(None, candidate="fp16", loss_scale=1024)] == [
        "seed", "conditioning", "mlp", "attention"]
    assert profiles_for(None, candidate="bf16", loss_scale=1024)[0].name == "bf16"
    assert profiles_for(["fp32-control"], candidate="fp16", loss_scale=1024)[0].fp32_patterns == ("*",)
    with pytest.raises(ValueError, match="native"):
        profiles_for(["bf16"], candidate="fp16", loss_scale=1024)


@pytest.mark.parametrize("name,extra", [("conditioning-mlp", "*.feed_forward.*"),
                                       ("conditioning-attention", "*.attention.*")])
def test_combination_profiles_are_explicit_supersets_not_automatic_defaults(name, extra):
    base = profiles_for(["conditioning"], candidate="fp16", loss_scale=128)[0]
    combined = profiles_for([name], candidate="fp16", loss_scale=128)[0]
    assert set(combined.fp32_patterns) == {*base.fp32_patterns, extra}
    assert combined.loss_scale == 128
    assert combined.precision == "fp16-islands"
    assert name not in [p.name for p in profiles_for(None, candidate="fp16", loss_scale=128)]


def test_worker_command_uses_requested_profile_and_reference_without_resume(tmp_path):
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"plan": {"blocks_to_swap": 26, "gradient_checkpointing": True}, "resume": None}))
    args = SimpleNamespace(weights="weights", inputs="inputs", reference="reference",
                           steps=3, resolution=256, memory_limit_gib=6.3)
    profile = profiles_for(["seed"], candidate="fp16", loss_scale=1024)[0]
    argv = command(args, profile, request)
    assert argv[argv.index("--swap") + 1] == "26"
    assert argv[argv.index("--replay-reference") + 1] == "reference"
    assert "--checkpoint-every-step" not in argv and "--resume" not in argv


@pytest.mark.parametrize("plan,limits", [(MemoryPlan(0, False), RetryLimits(4)),
                                        (MemoryPlan(0, True, 2), RetryLimits(4)),
                                        (MemoryPlan(0, True), RetryLimits(4, allow_batch_change=True))])
def test_search_only_authorizes_swap_changes(plan, limits):
    with pytest.raises(ValueError, match="fixed checkpointed"):
        search_profiles(plan, limits, [PrecisionProfile("a", "bf16")],
                        lambda *a, **k: pytest.fail("must not start"),
                        lambda _: None, record=lambda _: None)


def test_recording_failure_aborts_before_worker():
    def fail(_):
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        search_profiles(MemoryPlan(0, True), RetryLimits(4), [PrecisionProfile("a", "bf16")],
                        lambda *a, **k: pytest.fail("must not start"), lambda _: None, record=fail)


@pytest.mark.parametrize("capability,expected", [((8, 0), "bf16"), ((7, 5), "fp16-islands"),
                                                ((6, 0), "fp16-islands"), ((6, 1), "fp32-reference")])
def test_cli_creates_output_before_initial_record(tmp_path, monkeypatch, capability, expected):
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "manifest.json").write_text(json.dumps({
        "schema": "adaptive_training_replay_v1", "status": "captured", "uuid": "test-device", "steps": 3,
        "signature": {"comparison_mode": "fixed_pre_step_state_and_inputs", "model_family": "z_image",
                      "resolution": 256},
    }))
    output = tmp_path / "nested" / "search"
    monkeypatch.setattr("sys.argv", ["search", "--weights", "weights", "--inputs", "inputs",
                                    "--reference", str(reference), "--output", str(output)])
    monkeypatch.setattr(cli, "read_capture", lambda _: None)
    monkeypatch.setattr(cli.torch.cuda, "get_device_properties", lambda _: SimpleNamespace(uuid="test-device"))
    monkeypatch.setattr(cli.torch.cuda, "get_device_capability", lambda: capability)

    def search(*args, record, **kwargs):
        assert output.is_dir()
        assert args[2][0].precision == expected
        if expected == "fp32-reference":
            assert len(args[2]) == 1
        record({"status": "failed", "attempts": []})
        return {"status": "failed"}

    monkeypatch.setattr(cli, "search_profiles", search)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert json.loads((output / "summary.json").read_text())["status"] == "failed"
