"""Hardware default boundaries and their CPU-only CLI wiring checks."""

from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch

from bench.adaptive_runtime import calibrate_capture, recovery, search_dit, search_z_image_precision
from library.training.adaptive_runtime import precision
from library.training.adaptive_runtime.profile_search import PrecisionProfile, search_profiles
from library.training.adaptive_runtime.retry import MemoryPlan, RetryLimits


CAPABILITIES = [
    ((9, 0), "bf16", "bf16"), ((8, 0), "bf16", "bf16"), ((8, 6), "bf16", "bf16"),
    ((7, 9), "fp16", "fp16-islands"), ((7, 5), "fp16", "fp16-islands"),
    ((7, 0), "fp16", "fp16-islands"), ((6, 2), "fp32", "fp32-reference"),
    ((6, 1), "fp32", "fp32-reference"), ((6, 0), "fp16", "fp16-islands"),
    ((5, 2), "fp32", "fp32-reference"), ((3, 5), "fp32", "fp32-reference"),
]


@pytest.mark.parametrize("capability,candidate,worker_mode", CAPABILITIES)
def test_exact_hardware_boundaries(capability, candidate, worker_mode):
    assert precision.preferred_candidate(capability) == candidate
    assert precision.preferred_candidate(list(capability)) == candidate
    assert precision.preferred_probe_precision(capability) == worker_mode


@pytest.mark.parametrize("invalid", [None, (), (7,), (7, 0, 0), (0, 0), (-1, 0),
                                     (7, -1), (7, 10), (True, 0), (7.0, 0), "75"])
def test_invalid_capabilities_fail_closed(invalid):
    with pytest.raises(ValueError, match="compute capability"):
        precision.preferred_candidate(invalid)


def test_fp32_local_default_checks_reference_without_half_trial(monkeypatch):
    calls = []
    original = precision._evaluate

    def evaluate(*args, **kwargs):
        calls.append(kwargs["dtype"])
        return original(*args, **kwargs)

    monkeypatch.setattr(precision, "_evaluate", evaluate)
    module = torch.nn.Linear(2, 2, bias=False).requires_grad_(False)
    module.weight.fill_(40000)
    result = precision.calibrate_unit(module, [((torch.ones(1, 2),), {})],
                                      device="cpu", candidate="fp32")
    assert calls == [torch.float32]
    assert result["selected"] == "fp32" and not result["low_precision_tested"]
    assert result["validation_scope"] == "fp32_reference_finiteness_only"
    assert result["cases"][0]["safe"]
    with pytest.raises(ValueError, match="Nonfinite FP32"):
        precision.calibrate_unit(module, [((torch.full((1, 2), float("nan")),), {})],
                                 device="cpu", candidate="fp32")
    with pytest.raises(ValueError, match="At least one"):
        precision.calibrate_unit(module, [], device="cpu", candidate="fp32")


def fake_runner(monkeypatch, module, commands):
    class Runner:
        def __init__(self, directory, command, **kwargs):
            self.directory, self.command = directory, command

        def __call__(self, plan, *, attempt, resume):
            path = self.directory / f"attempt-{attempt:03d}"
            path.mkdir(parents=True)
            request = path / "request.json"
            request.write_text(json.dumps({"plan": asdict(plan), "resume": resume}))
            argv = self.command(request)
            commands.append(argv)
            return {"status": "ok", "disposable_probe": True, "fp32_modules": []}

    monkeypatch.setattr(module, "IsolatedRunner", Runner)


@pytest.mark.parametrize("capability,candidate,worker_mode", CAPABILITIES)
@pytest.mark.parametrize("family", ["krea2", "z_image"])
@pytest.mark.parametrize("entry", ["search", "recovery"])
def test_auto_cli_reaches_correct_worker(tmp_path, monkeypatch, capability, candidate, worker_mode,
                                         family, entry):
    module = search_dit if entry == "search" else recovery
    output = tmp_path / "output"
    argv = ["probe", "--weights", "unused", "--inputs", "unused", "--output", str(output)]
    if entry == "search":
        argv += ["--model-family", family]
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(module.torch.cuda, "get_device_capability", lambda: capability)
    commands = []
    fake_runner(monkeypatch, module, commands)
    if entry == "search":
        module.main()
    else:
        module.main(model_family=family)
    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("--precision") + 1] == worker_mode
    assert "--fp32-pattern" not in command and "--scaled-checkpoint" not in command
    assert ("--checkpoint-every-step" in command) == (entry == "recovery")
    assert json.loads((output / "summary.json").read_text())["precision"] == worker_mode


@pytest.mark.parametrize("entry", ["search", "recovery"])
def test_manual_mode_is_not_replaced_by_hardware_default(tmp_path, monkeypatch, entry):
    module = search_dit if entry == "search" else recovery
    argv = ["probe", "--weights", "unused", "--inputs", "unused", "--output", str(tmp_path / "out"),
            "--precision", "fp32-reference"]
    if entry == "search":
        argv += ["--model-family", "krea2"]
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(module.torch.cuda, "get_device_capability", lambda: pytest.fail("not auto"))
    commands = []
    fake_runner(monkeypatch, module, commands)
    module.main() if entry == "search" else module.main(model_family="krea2")
    assert commands[0][commands[0].index("--precision") + 1] == "fp32-reference"


@pytest.mark.parametrize("entry", ["search", "recovery"])
def test_auto_fp32_rejects_island_patterns_before_launch(tmp_path, monkeypatch, entry):
    module = search_dit if entry == "search" else recovery
    output = tmp_path / "out"
    argv = ["probe", "--weights", "unused", "--inputs", "unused", "--output", str(output),
            "--fp32-pattern", "*"]
    if entry == "search":
        argv += ["--model-family", "krea2"]
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(module.torch.cuda, "get_device_capability", lambda: (6, 1))
    with pytest.raises(SystemExit) as exc:
        module.main() if entry == "search" else module.main(model_family="krea2")
    assert exc.value.code == 2 and not output.exists()


@pytest.mark.parametrize("capability,candidate,worker_mode", CAPABILITIES)
def test_capture_cli_propagates_hardware_policy(tmp_path, monkeypatch, capability, candidate, worker_mode):
    output = tmp_path / "calibration.json"
    monkeypatch.setattr("sys.argv", ["calibrate", "--capture", "unused", "--output", str(output)])
    monkeypatch.setattr(calibrate_capture, "read_manifest", lambda _: {"source": {}, "units": [{}]})
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(
        name="test-device", uuid="test-uuid", major=capability[0]))

    def calibrate(directory, unit, actual):
        assert actual == candidate
        return {"name": "layer", "selected": actual}

    monkeypatch.setattr(calibrate_capture, "calibrate", calibrate)
    calibrate_capture.main()
    report = json.loads(output.read_text())
    assert report["local_plan"] == {"layer": candidate}
    assert report["low_precision_tested"] == (candidate != "fp32")


@pytest.mark.parametrize("capability,candidate,worker_mode", CAPABILITIES)
def test_profile_search_respects_hardware_default(capability, candidate, worker_mode):
    profiles = search_z_image_precision.profiles_for(
        None, candidate=precision.preferred_candidate(capability), loss_scale=1024)
    assert profiles[0].precision == worker_mode
    if candidate == "fp32":
        assert len(profiles) == 1
        assert profiles[0].loss_scale == 1 and profiles[0].fp32_patterns == ()


@pytest.mark.parametrize("kwargs", [{"loss_scale": 1024}, {"fp32_patterns": ("*",)}])
def test_fp32_profile_rejects_scaler_and_islands(kwargs):
    with pytest.raises(ValueError, match="Invalid"):
        PrecisionProfile("fp32", "fp32-reference", **kwargs)


def test_fp32_profile_command_and_numerical_gate(tmp_path):
    profile = PrecisionProfile("fp32", "fp32-reference")
    plan = MemoryPlan(24, True)
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"plan": asdict(plan), "resume": None}))
    args = SimpleNamespace(weights="weights", inputs="inputs", reference="reference",
                           steps=3, resolution=256, memory_limit_gib=None)
    argv = search_z_image_precision.command(args, profile, request)
    assert argv[argv.index("--precision") + 1] == "fp32-reference"
    assert float(argv[argv.index("--loss-scale") + 1]) == 1
    assert "--fp32-pattern" not in argv and "--replay-reference" in argv
    result = {"status": "ok", "disposable_probe": True, "precision_profile_resolved": True,
              "precision": "fp32-reference", "loss_scale": 1, "swap": 24, "fp32_modules": []}
    report = search_profiles(plan, RetryLimits(26), [profile], lambda *a, **k: result,
                             lambda _: {"status": "compared",
                                        "scope": "fixed_pre_step_state_and_inputs_not_quality_certificate",
                                        "cases": [{"within_experimental_tolerances": True}],
                                        "within_experimental_tolerances": True}, record=lambda _: None)
    assert report["status"] == "probe_validated"
    assert report["selected_profile"]["precision"] == "fp32-reference"
    assert not report["full_model_calibrated"] and not report["production_ready"]


def test_auto_fp32_recovery_rejects_loss_scaling(tmp_path, monkeypatch):
    output = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["recovery", "--weights", "unused", "--inputs", "unused",
                                    "--output", str(output), "--scaled-checkpoint", "--loss-scale", "1024"])
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (6, 1))
    with pytest.raises(SystemExit) as exc:
        recovery.main(model_family="krea2")
    assert exc.value.code == 2 and not output.exists()


@pytest.mark.parametrize("family", ["krea2", "z_image"])
def test_fp32_recovery_preserves_checkpoint_and_precision(tmp_path, family):
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"plan": {"blocks_to_swap": 24}, "resume": "/saved/step-1.pt"}))
    args = SimpleNamespace(weights="weights", inputs="inputs", memory_limit_gib=None, fp32_pattern=[])
    argv = recovery.worker_command(args, "fp32-reference", request, model_family=family)
    assert argv[argv.index("--precision") + 1] == "fp32-reference"
    assert argv[argv.index("--resume") + 1] == "/saved/step-1.pt"
    assert "--loss-scale" not in argv and "--scaled-checkpoint" not in argv
