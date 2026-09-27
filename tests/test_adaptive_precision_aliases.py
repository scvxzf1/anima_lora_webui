"""CPU-only precision ownership contracts and explicit known installer gaps."""

from types import SimpleNamespace

import pytest
import torch

from library.training.adaptive_runtime.islands import install_precision_islands
from library.training.adaptive_runtime.training_precision import install_training_precision


@pytest.fixture(autouse=True)
def forbid_cuda_initialization(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Static precision ownership tests must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    with torch.random.fork_rng(devices=[]):
        yield


def base(*, shared=None, frozen=False, dtype=torch.float32):
    model = torch.nn.Sequential(*(
        torch.nn.Linear(2, 2, device="cpu", dtype=dtype) for _ in range(2)
    ))
    with torch.no_grad():
        for module in model:
            module.weight.copy_(torch.tensor([[1.0, 0.5], [-0.25, 2.0]]))
            module.bias.copy_(torch.tensor([0.5, -0.5]))
    if shared is not None:
        setattr(model[1], shared, getattr(model[0], shared))
    return model.requires_grad_(not frozen)


def configuration(patterns):
    return SimpleNamespace(adaptive_fp32_modules=patterns, adaptive_loss_scale=1024.0,
                           model_family="krea2_raw")


def snapshot(model):
    parameters = {
        name: (parameter, parameter.detach().clone(), parameter.requires_grad)
        for name, parameter in model.named_parameters(remove_duplicate=False)
    }
    forwards = {name: module.forward for name, module in model.named_modules()}
    return parameters, forwards, list(model.state_dict())


def assert_unchanged(model, before):
    parameters, forwards, keys = before
    current = dict(model.named_parameters(remove_duplicate=False))
    assert current.keys() == parameters.keys()
    assert list(model.state_dict()) == keys
    for name, (parameter, value, requires_grad) in parameters.items():
        assert current[name] is parameter
        assert parameter.requires_grad == requires_grad
        torch.testing.assert_close(parameter, value, rtol=0, atol=0, equal_nan=True)
    for name, module in model.named_modules():
        assert module.forward == forwards[name]
        for attribute in ("_adaptive_dtype", "_adaptive_original_forward",
                          "_adaptive_training_precision", "_adaptive_precision_manifest_id",
                          "_adaptive_precision_manifest_names"):
            assert not hasattr(module, attribute)


@pytest.mark.parametrize("shared", ["weight", "bias"])
@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("patterns", [[], ["1"], ["*"]], ids=["all-fp16", "mixed", "all-fp32"])
def test_training_rejects_distinct_modules_sharing_parameters_before_mutation(shared, frozen, patterns):
    model = base(shared=shared, frozen=frozen, dtype=torch.bfloat16)
    assert model[0] is not model[1]
    assert getattr(model[0], shared) is getattr(model[1], shared)
    value = torch.tensor([[2.0, -2.0]], dtype=torch.bfloat16)
    expected = model(value).detach().clone()
    before = snapshot(model)
    with pytest.raises(ValueError, match="Aliased base parameters"):
        install_training_precision(model, configuration(patterns))
    assert_unchanged(model, before)
    torch.testing.assert_close(model(value), expected, rtol=0, atol=0)


def test_equal_values_in_independent_parameters_allow_mixed_precision():
    model = base(dtype=torch.bfloat16)
    assert model[0].weight is not model[1].weight
    assert model[0].weight.data_ptr() != model[1].weight.data_ptr()
    torch.testing.assert_close(model[0].weight, model[1].weight, rtol=0, atol=0)
    keys = list(model.state_dict())
    value = torch.tensor([[2.0, -2.0]], requires_grad=True)
    expected = value
    for module in model:
        expected = torch.nn.functional.linear(expected, module.weight.detach().float(),
                                              module.bias.detach().float())
    expected_gradient, = torch.autograd.grad(expected.sum(), value)
    manifest = install_training_precision(model, configuration(["1"]))
    result = model(value)
    result.sum().backward()
    assert manifest["assignments"] == {"0": "fp16", "1": "fp32"}
    assert model[0].weight.dtype == model[0].bias.dtype == torch.float16
    assert model[1].weight.dtype == model[1].bias.dtype == torch.float32
    assert result.dtype == value.grad.dtype == torch.float32
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert list(model.state_dict()) == keys
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(value.grad, expected_gradient, rtol=0, atol=0)


@pytest.mark.parametrize("invalid,match", [
    ("dtype", "fp16/bf16/fp32"),
    ("trainable", "frozen"),
    ("nonfinite", "Nonfinite"),
    ("overflow", "overflow"),
    ("module_alias", "Aliased precision"),
])
def test_direct_installer_validation_does_not_partially_wrap_first_unit(invalid, match):
    model = base(frozen=True)
    assignments = {"0": "fp16", "1": "fp16"}
    if invalid == "dtype":
        assignments["1"] = "fp64"
    elif invalid == "trainable":
        model[1].weight.requires_grad_(True)
    elif invalid == "nonfinite":
        model[1].weight.fill_(float("nan"))
    elif invalid == "overflow":
        model[1].weight.fill_(2**17)
    elif invalid == "module_alias":
        model[1] = model[0]
    before = snapshot(model)
    with pytest.raises(ValueError, match=match):
        install_precision_islands(model, assignments)
    assert_unchanged(model, before)


@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="Direct island installer lacks shared-Parameter preflight; training entry rejects it",
)
@pytest.mark.parametrize("assignments", [
    {"0": "fp16", "1": "fp32"},
    {"1": "fp32", "0": "fp16"},
    {"0": "fp16"},
], ids=["fp16-first", "fp32-first", "unassigned-consumer"])
def test_direct_installer_must_reject_shared_weight_before_mutation(assignments):
    model = base(shared="weight", frozen=True)
    before = snapshot(model)
    with pytest.raises(ValueError, match="(?i)(alias|shared)"):
        install_precision_islands(model, assignments)
    assert_unchanged(model, before)
