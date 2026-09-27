"""CPU witnesses for scale feasibility and targeted FP32 promotion, not DiT certification."""

from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from bench.adaptive_runtime.scaling import backward_unscaled, make_scaler
from library.training.adaptive_runtime.training_precision import install_training_precision
from networks.lora_modules.lora import LoRAModule


SMALL = Fraction(1, 2**30)
ZERO_TIE = Fraction(1, 2**25)
OVERFLOW_TIE = Fraction(65520)
BRANCHES = ("small", "large")


@pytest.fixture(autouse=True)
def forbid_cuda_initialization(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Scale feasibility fixtures must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    with torch.random.fork_rng(devices=[]):
        yield


def run_case(*, scale, promoted=(), large=2**20, checkpointed=False):
    model = torch.nn.ModuleDict({
        name: torch.nn.Sequential(*(
            torch.nn.Linear(1, 1, bias=False, device="cpu", dtype=torch.float32)
            for _ in range(2)
        )) for name in BRANCHES
    })
    model.requires_grad_(False)
    for parameter in model.parameters():
        parameter.fill_(1)
    # Keep the upstream base branch FP32 as well as its LoRA arithmetic.
    config = SimpleNamespace(adaptive_loss_scale=1024.0, model_family="anima",
                             adaptive_fp32_modules=["*.0", *(f"{name}.1" for name in promoted)])
    manifest = install_training_precision(model, config)
    adapters = []
    for name in BRANCHES:
        adapter = LoRAModule(name, model[name][0], lora_dim=1, alpha=1)
        adapter.apply_to()
        adapter.train()
        with torch.no_grad():
            adapter.lora_down.weight.fill_(1)
            adapter.lora_up.weight.zero_()
        adapters.append(adapter)
    optimizer = torch.optim.SGD([p for adapter in adapters for p in adapter.parameters()], lr=0.1)
    scaler = make_scaler(scale, device="cpu")
    noisy = torch.ones(1, 2, device="cpu", requires_grad=True)
    outputs = []
    for i, name in enumerate(BRANCHES):
        value = noisy[:, i:i + 1]
        outputs.append(checkpoint(model[name], value, use_reentrant=False)
                       if checkpointed else model[name](value))
    prediction = torch.cat(outputs, dim=1)
    loss = (prediction * torch.tensor([[float(SMALL), float(large)]])).sum()
    backward_unscaled(loss, noisy, optimizer, scaler)
    assert all(p.dtype == p.grad.dtype == torch.float32
               for adapter in adapters for p in adapter.parameters())
    return SimpleNamespace(
        prediction=prediction.detach(), loss=loss.detach(), input_gradient=noisy.grad.detach(),
        up=torch.stack([adapter.lora_up.weight.grad.reshape(()) for adapter in adapters]),
        down=torch.stack([adapter.lora_down.weight.grad.reshape(()) for adapter in adapters]),
        assignments=manifest["assignments"], optimizer=optimizer, scaler=scaler,
    )


def test_necessary_cast_intervals_can_be_disjoint_for_every_positive_scale():
    lower = ZERO_TIE / SMALL
    upper = OVERFLOW_TIE / 2**20
    assert lower == 32
    assert 0 < upper < 1 < lower


def test_nonempty_cast_interval_can_exclude_every_power_of_two():
    lower, upper = ZERO_TIE / SMALL, OVERFLOW_TIE / 2**10
    assert lower == 32 < 48 < upper < 64
    # 48 * 2^-30 is three quarters of one subnormal step.
    assert 48 * SMALL == Fraction(3, 4) * Fraction(1, 2**24)
    assert Fraction(1, 2**24) / 48 / SMALL == Fraction(4, 3)


@pytest.mark.parametrize("value,expected", [
    (2**-25, 0.0),
    (3 * 2**-26, 2**-24),
    (65504.0, 65504.0),
    (65519.0, 65504.0),
    (65520.0, float("inf")),
])
def test_cpu_fp16_rounding_thresholds(value, expected):
    assert torch.tensor(value, dtype=torch.float32, device="cpu").half().item() == expected


@pytest.mark.parametrize("scale,small,large", [
    (2**-5, 0.0, float(2**20)),
    (1, 0.0, float("inf")),
    (32, 0.0, float("inf")),
    (64, float(SMALL), float("inf")),
    (1024, float(SMALL), float("inf")),
])
def test_two_low_precision_branches_cannot_both_recover_gradients(scale, small, large):
    case = run_case(scale=scale)
    torch.testing.assert_close(case.prediction, torch.ones(1, 2), rtol=0, atol=0)
    assert torch.isfinite(case.loss)
    assert case.assignments["small.1"] == case.assignments["large.1"] == "fp16"
    assert case.up[0].item() == small
    assert case.up[1].item() == large


@pytest.mark.parametrize("scale", [64, 1024])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_promoting_large_branch_recovers_lora_and_input_gradients(scale, checkpointed):
    reference = run_case(scale=scale, promoted=BRANCHES, checkpointed=checkpointed)
    mixed = run_case(scale=scale, promoted=("large",), checkpointed=checkpointed)
    expected = torch.tensor([float(SMALL), float(2**20)])
    assert mixed.assignments["small.1"] == "fp16"
    assert mixed.assignments["large.1"] == "fp32"
    for field in ("prediction", "loss", "input_gradient", "up", "down"):
        torch.testing.assert_close(getattr(mixed, field), getattr(reference, field), rtol=0, atol=0)
    torch.testing.assert_close(mixed.up, expected, rtol=0, atol=0)
    torch.testing.assert_close(mixed.input_gradient, expected.unsqueeze(0), rtol=0, atol=0)
    torch.testing.assert_close(mixed.down, torch.zeros(2), rtol=0, atol=0)


def test_promoting_small_branch_alone_does_not_repair_large_overflow():
    case = run_case(scale=1024, promoted=("small",))
    assert case.up[0].item() == float(SMALL)
    assert torch.isinf(case.up[1])


def test_finite_gradients_can_pass_scaler_step_gate_despite_underflow(monkeypatch):
    case = run_case(scale=2, large=2**10)
    parameters = case.optimizer.param_groups[0]["params"]
    originals = [p.detach().clone() for p in parameters]
    assert all(torch.isfinite(p.grad).all() for p in parameters)
    assert case.up[0].item() == 0
    assert case.up[1].item() == 2**10
    calls = []
    # Observe the scaler's step decision without performing any optimizer update.
    monkeypatch.setattr(case.optimizer, "step", lambda: calls.append("would_update"))
    case.scaler.step(case.optimizer)
    case.scaler.update()
    assert calls == ["would_update"]
    assert case.scaler.get_scale() == 2
    for parameter, original in zip(parameters, originals, strict=True):
        torch.testing.assert_close(parameter, original, rtol=0, atol=0)


def test_feasible_non_power_of_two_scale_can_still_have_large_relative_error():
    case = run_case(scale=48, large=2**10)
    assert torch.isfinite(case.up).all()
    assert case.up[1].item() == 2**10
    ratio = case.up[0].item() / float(SMALL)
    assert ratio == pytest.approx(4 / 3, rel=1e-6)
    assert abs(ratio - 1) > 0.3
