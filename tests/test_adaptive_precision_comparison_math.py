"""CPU comparison-gate boundaries; no model training or optimizer updates."""

from fractions import Fraction
import hashlib
import math

import pytest
from safetensors.torch import save_file
import torch

from bench.adaptive_runtime.compare_training import aggregate_metrics, compare_case, load_case
from library.training.adaptive_runtime.precision import _measure, calibrate_unit


@pytest.fixture(autouse=True)
def forbid_cuda_initialization(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Comparison proofs must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    with torch.random.fork_rng(devices=[]):
        yield


def tensor(value):
    return torch.tensor([value], dtype=torch.float32, device="cpu")


def case(*, gradient=1.0, input_gradient=1.0):
    return {"prediction": tensor(1), "input_gradient": tensor(input_gradient),
            "gradient.weight": tensor(gradient)}


@pytest.mark.parametrize("value", [0.0, 2**-149, 2**-126, torch.finfo(torch.float32).max])
def test_double_reductions_stay_finite_across_fp32_capture_range(value):
    vector = tensor(value)
    local = _measure([vector], [vector.clone()])
    aggregate = aggregate_metrics({"x": vector}, {"x": vector.clone()}, ["x"])
    assert local["finite"] is True
    for metrics in (local, aggregate):
        assert metrics["relative_l2"] == 0
        assert math.isfinite(metrics["cosine"])
    assert aggregate["absolute_l2"] == 0
    assert aggregate["reference_l2"] == value


def test_opposite_maximum_fp32_values_do_not_overflow_metric_arithmetic():
    largest = torch.finfo(torch.float32).max
    reference, candidate = tensor(largest), tensor(-largest)
    local = _measure([reference], [candidate])
    aggregate = aggregate_metrics({"x": reference}, {"x": candidate}, ["x"])
    for metrics in (local, aggregate):
        assert metrics["relative_l2"] == pytest.approx(2)
        assert metrics["cosine"] == pytest.approx(-1)
    assert aggregate["absolute_l2"] == 2 * largest


@pytest.mark.parametrize("candidate_gradient,passes", [(0.0, True), (1.0, False)])
def test_zero_gradient_convention_and_nonzero_candidate(candidate_gradient, passes):
    report = compare_case(case(gradient=0, input_gradient=0),
                          case(gradient=candidate_gradient, input_gradient=0))
    assert report["within_experimental_tolerances"] is passes
    assert report["input_gradient"]["cosine"] == 1
    assert report["adapter_gradient"]["cosine"] == (1 if passes else 0)


def test_aggregate_gradient_gate_does_not_bound_first_adam_update():
    small = Fraction(1, 2**26)
    reference = {**case(), "gradient.small": tensor(float(small))}
    candidate = {**case(), "gradient.small": tensor(0)}
    report = compare_case(reference, candidate)
    # Characterize the current aggregate contract, not a quality acceptance claim.
    assert report["within_experimental_tolerances"] is True
    assert report["adapter_gradient"]["relative_l2"] < 2e-8
    assert report["adapter_gradient"]["cosine"] > 0.999
    worst = report["worst_adapter_parameters"][0]
    assert worst["name"] == "gradient.small"
    assert worst["relative_l2"] == 1 and worst["cosine"] == 0
    assert worst["absolute_l2"] == float(small)
    # First bias-corrected Adam step, zero moments, no decay, same starting weights.
    learning_rate, epsilon = Fraction(1, 1000), Fraction(1, 10**8)
    reference_update = learning_rate * small / (small + epsilon)
    candidate_update = Fraction(0)
    assert abs(reference_update - candidate_update) > learning_rate / 2


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_local_candidate_nonfinite_metrics_fail_closed(value):
    assert _measure([tensor(1)], [tensor(value)]) == {
        "finite": False, "relative_l2": None, "cosine": None,
    }


@pytest.mark.parametrize("invalid", ["nan", "inf", "negative_inf", "float64", "empty"])
def test_capture_loader_rejects_invalid_tensors_before_comparison(tmp_path, invalid):
    tensors = case()
    replacements = {"nan": tensor(float("nan")), "inf": tensor(float("inf")),
                    "negative_inf": tensor(-float("inf")), "float64": tensor(1).double(),
                    "empty": torch.empty(0, dtype=torch.float32, device="cpu")}
    tensors["gradient.weight"] = replacements[invalid]
    path = tmp_path / "case.safetensors"
    save_file(tensors, str(path))
    record = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match="Invalid training capture tensors"):
        load_case(tmp_path, record, ["gradient.weight"])


@pytest.mark.parametrize("change", ["keys", "shape"])
def test_comparison_rejects_changed_parameter_contract(change):
    candidate = case()
    if change == "keys":
        candidate["gradient.other"] = candidate.pop("gradient.weight")
    else:
        candidate["gradient.weight"] = torch.ones(2, dtype=torch.float32, device="cpu")
    with pytest.raises(ValueError, match="key or shape contract changed"):
        compare_case(case(), candidate)


@pytest.mark.xfail(
    strict=True, raises=pytest.fail.Exception,
    reason="Aggregate cosine denominator floor rejects identical tiny gradients",
)
def test_identical_tiny_capture_should_pass_comparison():
    reference = case(gradient=2**-60, input_gradient=2**-60)
    report = compare_case(reference, {name: value.clone() for name, value in reference.items()})
    for group in ("prediction", "input_gradient", "adapter_gradient"):
        assert report[group]["relative_l2"] == 0
        assert math.isfinite(report[group]["cosine"])
        if report[group]["cosine"] != pytest.approx(1):
            pytest.fail("Identical nonzero capture tensors must have cosine 1")
    assert report["within_experimental_tolerances"] is True


@pytest.mark.xfail(
    strict=True, raises=pytest.fail.Exception,
    reason="Local cosine denominator floor labels a finite FP32 self-reference unsafe",
)
def test_fp32_reference_with_small_gradient_should_have_safe_case():
    class TinySlope(torch.nn.Module):
        def forward(self, value):
            return value * 2**-60

    report = calibrate_unit(TinySlope(), [((tensor(1),), {})], device="cpu", candidate="fp32")
    assert report["selected"] == "fp32" and report["low_precision_tested"] is False
    assert report["validation_scope"] == "fp32_reference_finiteness_only"
    evidence = report["cases"][0]
    for group in ("output", "gradient"):
        assert evidence[group]["finite"] is True
        assert evidence[group]["relative_l2"] == 0
        assert math.isfinite(evidence[group]["cosine"])
        if evidence[group]["cosine"] != pytest.approx(1):
            pytest.fail("A nonzero FP32 self-reference must have cosine 1")
    assert evidence["safe"] is True
