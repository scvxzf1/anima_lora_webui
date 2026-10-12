import json
from types import SimpleNamespace

import pytest
from safetensors.torch import save_file

from bench.convrot_90hx import assessment, grid
from bench.convrot_90hx.manifest import write_json

import torch

from bench.convrot_90hx.assessment import GATES, vector_pass
from bench.convrot_90hx.numerics import vector_metrics


def test_same_norm_wrong_direction_fails_frozen_vector_gate():
    result = vector_metrics(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))
    assert not vector_pass(result, GATES["grad_rel_l2"], GATES["grad_cosine"])


def test_zero_reference_uses_absolute_budget():
    assert vector_pass(vector_metrics(torch.zeros(2), torch.zeros(2)), 0.05, 0.995)
    assert not vector_pass(vector_metrics(torch.zeros(2), torch.ones(2)), 0.05, 0.995)


def test_nonfinite_never_passes_gate():
    assert not vector_pass(
        vector_metrics(torch.ones(2), torch.full((2,), float("inf"))), 0.05
    )


def write_sample(folder, gradients=None, updates=None, missing=None):
    folder.mkdir(parents=True)
    values = {"large": torch.ones(2), "small": torch.full((2,), 1e-4)}
    save_file(
        values if gradients is None else gradients, str(folder / "grads.safetensors")
    )
    save_file(
        values if updates is None else updates, str(folder / "updates.safetensors")
    )
    save_file({"output": torch.ones(2)}, str(folder / "output.safetensors"))
    write_json(
        folder / "sample.json",
        {
            "loss": 1.0,
            "finite": [True],
            "inputs": {"input_hash": "same"},
            "missing_gradients": [] if missing is None else missing,
        },
    )
    return folder


@pytest.mark.parametrize("reported_missing", [[], ["small"]])
def test_both_sides_missing_gradient_cannot_pass(tmp_path, reported_missing):
    values = {"large": torch.ones(2)}
    ref = write_sample(tmp_path / "ref", gradients=values, missing=reported_missing)
    cand = write_sample(tmp_path / "cand", gradients=values, missing=reported_missing)
    result = assessment.assess_sample(ref, cand, {"large": {}, "small": {}})
    assert result["status"] == "FAIL"
    assert not result["checks"]["gradient_names"]
    if reported_missing:
        assert not result["checks"]["missing"]


def test_missing_small_update_cannot_hide_in_overall_gate(tmp_path):
    ref = write_sample(tmp_path / "ref")
    cand = write_sample(tmp_path / "cand", updates={"large": torch.ones(2)})
    result = assessment.assess_sample(ref, cand, {"large": {}, "small": {}})
    assert result["checks"]["updates"]
    assert result["status"] == "FAIL"
    assert not result["checks"]["update_names"]
    assert not result["checks"]["update_per_layer"]


def test_small_update_wrong_direction_fails_layer_gate(tmp_path):
    ref = write_sample(tmp_path / "ref")
    cand = write_sample(
        tmp_path / "cand",
        updates={
            "large": torch.ones(2),
            "small": torch.full((2,), -1e-4),
        },
    )
    result = assessment.assess_sample(ref, cand, {"large": {}, "small": {}})
    assert result["checks"]["updates"]
    assert result["failing_update_layers"] == ["small"]
    assert result["status"] == "FAIL"


@pytest.mark.parametrize("side", ["ref", "cand"])
def test_either_reported_missing_gradient_fails(tmp_path, side):
    ref = write_sample(tmp_path / "ref", missing=["small"] if side == "ref" else [])
    cand = write_sample(tmp_path / "cand", missing=["small"] if side == "cand" else [])
    result = assessment.assess_sample(ref, cand, {"large": {}, "small": {}})
    assert not result["checks"]["missing"]
    assert result["status"] == "FAIL"


def write_run(folder, names, status="PASS", declared=None):
    folder.mkdir()
    manifest = {
        "initial_hash": "same",
        "model": {"initial_parameters": {"large": {}, "small": {}}},
        "expected_samples": list(names) if declared is None else declared,
    }
    write_json(folder / "manifest.json", manifest)
    write_json(
        folder / "result.json",
        {
            "status": status,
            "count": len(names),
            "samples": [{"name": n} for n in names],
        },
    )
    for name in names:
        write_sample(folder / name)
    return folder


def run_assessment(monkeypatch, ref, cand, out):
    monkeypatch.setattr(
        "sys.argv", ["assessment", str(ref), str(cand), "--out", str(out)]
    )
    assessment.main()


def test_single_sample_cannot_formally_pass(tmp_path, monkeypatch):
    names = assessment.EXPECTED_SAMPLES[:1]
    ref, cand = (write_run(tmp_path / side, names) for side in ("ref", "cand"))
    with pytest.raises(ValueError, match="incomplete numerical sample matrix"):
        run_assessment(monkeypatch, ref, cand, tmp_path / "assessment.json")


@pytest.mark.parametrize("delta", [-1, 1])
def test_candidate_extra_or_missing_sample_rejected(tmp_path, monkeypatch, delta):
    names = assessment.EXPECTED_SAMPLES
    ref = write_run(tmp_path / "ref", names)
    candidate_names = names[:-1] if delta < 0 else names + ["cache2-seed0-sigma0.1"]
    cand = write_run(tmp_path / "cand", candidate_names, declared=names)
    with pytest.raises(ValueError, match="incomplete numerical sample matrix"):
        run_assessment(monkeypatch, ref, cand, tmp_path / "assessment.json")


@pytest.mark.parametrize("extra", [False, True])
def test_actual_candidate_folders_must_match_completed_result(
    tmp_path, monkeypatch, extra
):
    import shutil

    ref, cand = (
        write_run(tmp_path / side, assessment.EXPECTED_SAMPLES)
        for side in ("ref", "cand")
    )
    if extra:
        write_sample(cand / "cache2-seed0-sigma0.1")
    else:
        shutil.rmtree(cand / assessment.EXPECTED_SAMPLES[-1])
    with pytest.raises(ValueError, match="incomplete numerical sample matrix"):
        run_assessment(monkeypatch, ref, cand, tmp_path / "assessment.json")


@pytest.mark.parametrize("status", ["RUNNING", "FAIL", "INCOMPLETE"])
def test_requires_successful_completed_result(tmp_path, monkeypatch, status):
    ref = write_run(tmp_path / "ref", assessment.EXPECTED_SAMPLES)
    cand = write_run(tmp_path / "cand", assessment.EXPECTED_SAMPLES, status=status)
    with pytest.raises(ValueError, match="incomplete numerical sample matrix"):
        run_assessment(monkeypatch, ref, cand, tmp_path / "assessment.json")


def test_complete_matrix_passes(tmp_path, monkeypatch):
    ref, cand = (
        write_run(tmp_path / side, assessment.EXPECTED_SAMPLES)
        for side in ("ref", "cand")
    )
    out = tmp_path / "assessment.json"
    run_assessment(monkeypatch, ref, cand, out)
    assert json.loads(out.read_text())["status"] == "PASS"


def test_grid_preserves_missing_gradient_evidence(tmp_path, monkeypatch):
    network = torch.nn.Linear(2, 1)
    initial = {n: p.detach().clone() for n, p in network.named_parameters()}
    monkeypatch.setattr(grid, "drain", lambda model: None)
    monkeypatch.setattr(grid, "load_input", lambda *args: ({}, {"input_hash": "same"}))
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(
        grid,
        "step",
        lambda *args: (
            torch.ones(1),
            torch.tensor(1.0),
            torch.tensor([True]),
        ),
    )
    args = SimpleNamespace(
        out_dir=tmp_path, cache_dir=tmp_path, image_dir=tmp_path, swap=0
    )
    with pytest.raises(RuntimeError, match="missing gradients"):
        grid.run_sample(args, None, network, initial, 0, 0, 0.1)
    sample = tmp_path / "cache0-seed0-sigma0.1"
    assert json.loads((sample / "sample.json").read_text())["missing_gradients"] == [
        "bias",
        "weight",
    ]
    assert (sample / "grads.safetensors").exists()
    assert (sample / "updates.safetensors").exists()
