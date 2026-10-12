"""CPU-only Anima precision diagnostic contracts."""

import json
from types import SimpleNamespace

import pytest
from safetensors.torch import save_file
import torch

from bench.adaptive_runtime import anima_gradient_diagnostics as gradients
from bench.adaptive_runtime import diagnose_anima_precision as matrix
from bench.adaptive_runtime import probe_anima_train as probe


def arguments(tmp_path, **changes):
    weights, inputs = tmp_path / "weights", tmp_path / "inputs"
    weights.write_bytes(b"weight")
    inputs.write_bytes(b"input")
    return SimpleNamespace(**{**dict(weights=weights, inputs=inputs, output_dir=tmp_path / "diagnostics",
                                    gpu_uuid="GPU-test", swap=20, steps=2, worker_timeout=10,
                                    case=["mlp-only-fp16", "conditioning-attention-scale64"],
                                    max_cases=8), **changes})


def test_case_selection_bounds_and_repetition():
    assert len(matrix.selected_cases(None, 8)) == 6
    with pytest.raises(ValueError, match="max-cases"):
        matrix.selected_cases(None, 9)
    with pytest.raises(ValueError, match="unique"):
        matrix.selected_cases(["mlp-only-fp16"] * 2, 8)
    with pytest.raises(ValueError, match="Unknown"):
        matrix.selected_cases(["not-a-case"], 8)
    with pytest.raises(ValueError, match="within"):
        matrix.selected_cases(None, 5)


@pytest.mark.parametrize("name,flag,scale", [
    ("mlp-only-fp16", "--fp16-only-pattern", "1024"),
    ("attention-only-fp16", "--fp16-only-pattern", "1024"),
    ("conditioning-only-fp16", "--fp16-only-pattern", "1024"),
    ("conditioning-attention-scale1", "--fp32-pattern", "1"),
    ("conditioning-attention-scale64", "--fp32-pattern", "64"),
    ("conditioning-attention-scale1024", "--fp32-pattern", "1024"),
    ("mlp-layer1-only-fp16", "--fp16-only-pattern", "1024"),
    ("mlp-layer2-only-fp16", "--fp16-only-pattern", "1024"),
    ("fp32-control", "--fp32-pattern", "1024"),
])
def test_case_worker_command(tmp_path, name, flag, scale):
    command = matrix.case_command(arguments(tmp_path), tmp_path, name, tmp_path / "replay")
    assert flag in command and ("--fp32-pattern" in command) != (flag == "--fp16-only-pattern")
    assert command[command.index("--loss-scale") + 1] == scale
    assert command[command.index("--replay-reference") + 1] == str(tmp_path / "replay")


@pytest.mark.parametrize("name,pattern", [("mlp-layer1-only-fp16", "blocks.*.mlp.layer1"),
                                          ("mlp-layer2-only-fp16", "blocks.*.mlp.layer2")])
def test_split_mlp_cases_target_one_projection_and_fixed_sigma(tmp_path, name, pattern):
    cfg = arguments(tmp_path, steps=1, single_sigma=0.5)
    command = matrix.case_command(cfg, tmp_path, name, tmp_path / "replay")
    assert command[command.index("--fp16-only-pattern") + 1] == pattern
    assert command[command.index("--single-sigma") + 1] == "0.5"


@pytest.mark.parametrize("steps,sigma", [(2, 0.5), (1, float("nan")), (1, 0), (1, 1)])
def test_matrix_rejects_invalid_fixed_sigma_before_gpu(tmp_path, steps, sigma):
    cfg = arguments(tmp_path, steps=steps, single_sigma=sigma)
    with pytest.raises(ValueError, match="single-sigma"):
        matrix.run(cfg, gpu_check=lambda _: pytest.fail("Must reject before GPU query"))
    assert not cfg.output_dir.exists()


def test_group_parser_uses_actual_parameter_names():
    assert gradients.parameter_group("gradient.lora_unet_blocks_0_mlp_layer1.lora_down.weight") == (
        "block-000", "mlp", "down")
    assert gradients.parameter_group("gradient.lora_unet_blocks_12_self_attn_q_proj.lora_up.weight") == (
        "block-012", "self_attn", "up")
    assert gradients.parameter_group("gradient.lora_unet_blocks_1_cross_attn_k_proj.lora_up.weight") == (
        "block-001", "cross_attn", "up")
    assert gradients.parameter_group("gradient.unknown.lora_down.weight") == ("other", "other", "other")


def test_finite_gradient_norm_overflow_is_numerical_rejection():
    parameter = torch.nn.Parameter(torch.ones(2))
    parameter.grad = torch.full_like(parameter, 1e38)
    assert bool(torch.isfinite(parameter.grad).all())
    with pytest.raises(probe.NonfiniteTrainingError, match="gradient norm"):
        probe.clip_finite_gradients([parameter])


def test_clip_does_not_hide_unrelated_errors(monkeypatch):
    def invalid(*args, **kwargs):
        raise RuntimeError("unrelated clipping failure")

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", invalid)
    with pytest.raises(RuntimeError, match="unrelated"):
        probe.clip_finite_gradients([])


def test_fp16_target_complement_uses_real_linear_inventory():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Module()])
            self.blocks[0].mlp = torch.nn.Module()
            self.blocks[0].mlp.layer1 = torch.nn.Linear(1, 1)
            self.blocks[0].self_attn = torch.nn.Module()
            self.blocks[0].self_attn.q_proj = torch.nn.Linear(1, 1)
            self.final_layer = torch.nn.Linear(1, 1)

    names = {n for n, m in Model().named_modules() if type(m) is torch.nn.Linear}
    complement = set(probe.fp32_complement(Model(), matrix.MLP))
    assert names - complement == {"blocks.0.mlp.layer1"}
    assert complement == {"blocks.0.self_attn.q_proj", "final_layer"}
    with pytest.raises(ValueError, match="FP16 targets"):
        probe.fp32_complement(Model(), ["blocks.*.absent.*"])


def _capture(tmp_path, name, values):
    root = tmp_path / name
    directory = root / "training-capture"
    directory.mkdir(parents=True)
    tensors = {f"gradient.{key}": torch.tensor(value, dtype=torch.float32)
               for key, value in values.items()}
    tensors["prediction"] = torch.ones(1)
    tensors["input_gradient"] = torch.ones(1)
    save_file(tensors, str(directory / "step-000001.safetensors"))
    (root / "result.json").write_text(json.dumps({
        "status": "ok", "precision": "fp32-reference" if name == "reference" else "fp16-islands",
        "inputs_sha256": "inputs", "loss_scale": 1, "updates": [{"step": 1}]}))
    signature = {"precision": "fp32-reference" if name == "reference" else "fp16-islands",
                 "inputs_sha256": "inputs", "loss_scale": 1}
    (directory / "manifest.json").write_text(json.dumps({
        "schema": "adaptive_training_capture_v1", "status": "captured", "steps": 1,
        "signature": signature, "expected_gradients": sorted(k for k in tensors if k.startswith("gradient.")),
        "cases": [{"step": 1, "sigma": 0.2, "file": "step-000001.safetensors",
                   "sha256": __import__("hashlib").sha256((directory / "step-000001.safetensors").read_bytes()).hexdigest()}]}))
    return directory


def test_fp64_group_metrics_zero_counts_and_unknown(tmp_path):
    a = "lora_unet_blocks_0_mlp_layer1.lora_down.weight"
    b = "lora_unet_blocks_1_self_attn_q_proj.lora_up.weight"
    reference = _capture(tmp_path, "reference", {a: [1e20, 0], b: [0, 0], "unknown.weight": [2]})
    trial = _capture(tmp_path, "candidate", {a: [1e20, 1], b: [0, 0], "unknown.weight": [3]})
    report = gradients.diagnose(reference, trial, {"status": "compared", "scope": "fixed_pre_step_test",
                                                     "cases": [{}]})
    assert report["overall"]["diff_square"] == 2
    assert report["overall"]["reference_l2"] >= 1e20
    assert report["groups"]["family"]["mlp"]["diff_square_fraction"] == 0.5
    assert report["per_step"]["1"]["sigma"] == 0.2
    assert report["per_step"]["1"]["groups"]["block"]["block-000"]["diff_square_fraction"] == 0.5
    assert report["groups"]["family"]["self_attn"]["reference_zero_elements"] == 2
    assert report["unknown_parameters"] == ["gradient.unknown.weight"]
    assert report["zero_counts_are_observations_not_underflow_classification"]
    assert gradients.Accumulator().result(0)["relative_l2"] == 0


@pytest.mark.parametrize("within_tolerances", [False, True])
def test_matrix_runs_all_cases_without_early_selection(tmp_path, monkeypatch, within_tolerances):
    cfg = arguments(tmp_path)
    monkeypatch.setattr(matrix, "_complete", lambda *a, **kw: True)
    launched = []

    def worker(argv, directory, uuid, timeout):
        launched.append(directory.name)
        return {"status": "ok", "returncode": 0, "precision_manifest": {
            "assignments": {"blocks.0.mlp.layer1": "fp16", "blocks.0.self_attn.q_proj": "fp32"}}}

    report = matrix.run(cfg, gpu_check=lambda _: {"uuid": "GPU-test"}, monitor=worker,
                        comparator=lambda *_: {"status": "compared", "scope": "fixed_pre_step_test",
                                               "within_experimental_tolerances": within_tolerances},
                        diagnostic=lambda *_: {"status": "diagnosed"})
    assert report["status"] == "completed" and len(report["cases"]) == 2
    expected = "within_experimental_tolerances" if within_tolerances else "numerical_rejection"
    assert len(launched) == 3 and all(c["outcome"] == expected for c in report["cases"])
    assert all((cfg.output_dir / f"case-{i:02d}-{name}" / "comparison.json").is_file()
               for i, name in enumerate(cfg.case, 1))
    assert not (cfg.output_dir / "selected-plan.json").exists()


@pytest.mark.parametrize("failure", ["gpu_unavailable", "host_limit", "telemetry_error"])
def test_monitor_failure_stops_matrix(tmp_path, monkeypatch, failure):
    cfg = arguments(tmp_path)
    monkeypatch.setattr(matrix, "_complete", lambda *a, **kw: True)
    called = []

    def worker(argv, directory, uuid, timeout):
        called.append(directory.name)
        return {"status": "ok" if directory.name == "reference" else failure}

    report = matrix.run(cfg, gpu_check=lambda _: {"uuid": "GPU-test"}, monitor=worker)
    assert report["status"] == "stopped" and len(called) == 2
    assert report["cases"][0]["outcome"] == "infrastructure_stop"
    assert (cfg.output_dir / "summary.json").is_file()


def test_numerical_worker_failure_is_preserved_and_continues(tmp_path, monkeypatch):
    cfg = arguments(tmp_path)
    monkeypatch.setattr(matrix, "_complete", lambda *a, **kw: True)

    def worker(argv, directory, uuid, timeout):
        return ({"status": "ok"} if directory.name == "reference" else
                {"status": "error", "failure_kind": "nonfinite", "error": "gradient norm"})

    report = matrix.run(cfg, gpu_check=lambda _: {"uuid": "GPU-test"}, monitor=worker)
    assert report["status"] == "completed" and len(report["cases"]) == 2
    assert all(c["outcome"] == "numerical_rejection" for c in report["cases"])
    assert all(c["result"]["error"] == "gradient norm" for c in report["cases"])


@pytest.mark.parametrize("kind", ["runtime", "cleanup"])
def test_runtime_and_cleanup_failure_stop_matrix(tmp_path, monkeypatch, kind):
    cfg = arguments(tmp_path)
    monkeypatch.setattr(matrix, "_complete", lambda *a, **kw: True)
    launches = []

    def worker(argv, directory, uuid, timeout):
        launches.append(directory.name)
        return ({"status": "ok"} if directory.name == "reference" else
                {"status": "error", "failure_kind": kind})

    report = matrix.run(cfg, gpu_check=lambda _: {"uuid": "GPU-test"}, monitor=worker)
    assert report["status"] == "stopped" and len(launches) == 2
    assert report["cases"][0]["outcome"] == "worker_failed"
