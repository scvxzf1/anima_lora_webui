import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from bench.adaptive_runtime.capture import LinearCapture, capture_path, read_manifest, select_rows
from bench.adaptive_runtime import calibrate_capture


def test_sampling_includes_extreme_and_is_bounded():
    value = torch.ones(2, 50, 4)
    value[0, 23, 2] = 100
    rows, indices = select_rows(value, 7)
    assert len(rows) <= 7
    assert 23 in indices
    assert rows.dtype == torch.float32
    assert rows.device.type == "cpu"
    assert select_rows(value, 1)[1] == [23]
    with pytest.raises(ValueError):
        select_rows(value, 0)


def frozen_model():
    return torch.nn.Sequential(torch.nn.Linear(4, 3)).requires_grad_(False)


def test_capture_forward_once_per_step_and_cleanup(tmp_path):
    model = frozen_model()
    report = {"stage": "forward", "updates": [], "sigma": 0.2}
    directory = tmp_path / "capture"
    capture = LinearCapture(model, ["0"], directory, report, max_cases=2, max_rows=2)
    model(torch.randn(8, 4))
    model(torch.randn(8, 4))
    assert len(capture.manifest["units"][0]["cases"]) == 1
    report.update(stage="backward", updates=[{"step": 1}])
    model(torch.randn(8, 4))
    assert len(capture.manifest["units"][0]["cases"]) == 1
    report["stage"] = "forward"
    model(torch.randn(8, 4))
    capture.close()
    manifest = read_manifest(directory)
    assert [c["step"] for c in manifest["units"][0]["cases"]] == [1, 2]
    assert not model[0]._forward_pre_hooks
    assert manifest["precision_calibrated"] is False


def test_partial_and_nonfinite_rejected(tmp_path):
    model = frozen_model()
    capture = LinearCapture(model, ["0"], tmp_path / "capture", {"stage": "forward"})
    with pytest.raises(ValueError, match="nonfinite"):
        model(torch.full((2, 4), float("nan")))
    capture.close()
    with pytest.raises(ValueError, match="completed"):
        read_manifest(capture.directory)


def test_reject_trainable_aliases_and_budget(tmp_path):
    model = frozen_model()
    with pytest.raises(ValueError, match="budget"):
        LinearCapture(model, ["0"], tmp_path / "budget", {}, max_bytes=1)
    model[0].requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        LinearCapture(model, ["0"], tmp_path / "trainable", {})
    model[0].requires_grad_(False)
    model.add_module("alias", model[0])
    with pytest.raises(ValueError, match="Aliased"):
        LinearCapture(model, ["0", "alias"], tmp_path / "alias", {})


@pytest.mark.parametrize("relative", [None, "", ".", "../outside", "/tmp/outside"])
def test_reject_invalid_path(tmp_path, relative):
    with pytest.raises(ValueError):
        capture_path(tmp_path, relative)


def test_reject_symlink_escape(tmp_path):
    (tmp_path / "escape").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        capture_path(tmp_path, "escape/weights")


@pytest.mark.parametrize("mutation", ["empty", "cases", "duplicate", "path"])
def test_malformed_manifest(tmp_path, mutation):
    model = frozen_model()
    capture = LinearCapture(model, ["0"], tmp_path / "capture", {"stage": "forward"}, max_cases=1)
    model(torch.randn(2, 4))
    capture.close()
    manifest = read_manifest(capture.directory)
    if mutation == "empty":
        manifest["units"] = []
    elif mutation == "cases":
        manifest["units"][0]["cases"] = []
    elif mutation == "duplicate":
        manifest["units"].append(manifest["units"][0])
    else:
        manifest["units"][0]["weights"] = "../outside"
    (capture.directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        read_manifest(capture.directory)


def test_calibration_uses_plain_frozen_snapshot(tmp_path, monkeypatch):
    model = frozen_model()
    capture = LinearCapture(model, ["0"], tmp_path / "capture", {"stage": "forward"}, max_cases=1)
    model(torch.randn(2, 4))
    capture.close()

    def check(layer, cases, **kwargs):
        assert type(layer) is torch.nn.Linear
        assert not layer.weight.requires_grad
        assert len(cases) == 1
        assert kwargs == {"device": "cuda", "candidate": "fp16"}
        return {"selected": "fp16"}

    monkeypatch.setattr(calibrate_capture, "calibrate_unit", check)
    unit = read_manifest(capture.directory)["units"][0]
    result = calibrate_capture.calibrate(capture.directory, unit, "fp16")
    assert result["scope"] == "frozen_linear_output_and_input_vjp_only"
    assert len(result["weights_sha256"]) == 64
    assert len(result["input_sha256"]) == 1
    case_path = capture.directory / unit["cases"][0]["file"]
    tensors = load_file(str(case_path))
    save_file({"wrong_key": tensors["input"]}, str(case_path))
    with pytest.raises(ValueError, match="Invalid captured Linear input"):
        calibrate_capture.calibrate(capture.directory, unit, "fp16")
