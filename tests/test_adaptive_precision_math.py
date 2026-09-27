"""CPU counterexamples and algebra for islands; not GPU/DiT certification."""

from fractions import Fraction

import pytest
import torch

from bench.adaptive_runtime.scaling import backward_unscaled, make_scaler
from library.training.adaptive_runtime.islands import install_precision_islands


DELTA = 2**-12


@pytest.fixture(autouse=True)
def forbid_cuda_initialization(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Static precision proofs must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    with torch.random.fork_rng(devices=[]):
        yield


def frozen_scalar(weight, precision="fp16"):
    model = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False, device="cpu"))
    model.requires_grad_(False)
    model[0].weight.fill_(weight)
    install_precision_islands(model, {"0": precision})
    return model


def test_fp32_output_cannot_restore_rounded_weight():
    model = frozen_scalar(1 + DELTA)
    output = model(torch.ones(1, 1, device="cpu"))
    assert model[0].weight.dtype == torch.float16
    assert output.dtype == torch.float32
    assert output.item() == 1
    assert abs(output.item() - (1 + DELTA)) == DELTA


def test_equal_predictions_do_not_imply_equal_input_gradients():
    model = torch.nn.Sequential(torch.nn.Linear(2, 1, bias=False, device="cpu"))
    model.requires_grad_(False)
    model[0].weight.copy_(torch.tensor([[1, 1 + DELTA]]))
    install_precision_islands(model, {"0": "fp16"})
    x = torch.tensor([[1.0, 0.0]], requires_grad=True)
    prediction = model(x)
    prediction.sum().backward()
    assert prediction.item() == 1  # Also exactly 1 for the FP32 reference.
    assert torch.equal(x.grad, torch.tensor([[1.0, 1.0]]))
    assert abs(x.grad[0, 1].item() - (1 + DELTA)) == DELTA


@pytest.mark.parametrize("scale", [1, 1024])
def test_loss_scaling_cannot_repair_forward_overflow(scale):
    output = frozen_scalar(2)(torch.tensor([[32768.0]]))
    assert output.dtype == torch.float32
    assert torch.isinf(output).all()
    assert torch.isinf(output.sum() * scale)


@pytest.mark.parametrize("scale,expected", [(1, 0.0), (1024, 2**-26), (2**44, float("inf"))])
def test_backward_scaling_has_both_underflow_and_overflow_limits(scale, expected):
    model = frozen_scalar(1)
    parameter = torch.nn.Parameter(torch.ones(1, 1, device="cpu"))
    noisy = torch.ones(1, 1, device="cpu", requires_grad=True)
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler = make_scaler(scale, device="cpu")
    loss = model(noisy * parameter).sum() * 2**-26
    backward_unscaled(loss, noisy, optimizer, scaler)
    assert parameter.grad.item() == expected
    assert noisy.grad.item() == expected


@pytest.mark.parametrize("scale", [1, 1024])
def test_loss_scaling_does_not_recover_an_activation_rounded_to_zero(scale):
    model = frozen_scalar(1)
    parameter = torch.nn.Parameter(torch.ones(1, 1, device="cpu"))
    noisy = torch.tensor([[2**-26]], requires_grad=True)
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    loss = (model(noisy) * parameter).sum()
    backward_unscaled(loss, noisy, optimizer, make_scaler(scale, device="cpu"))
    assert loss.item() == 0
    assert parameter.grad.item() == 0  # FP32 reference parameter gradient is 2**-26.


def test_promotion_can_break_cancellation_and_increase_aggregate_error():
    x = torch.ones(1, 1, device="cpu")
    low_a, low_b = frozen_scalar(1 + DELTA), frozen_scalar(1 - DELTA)
    high_a, high_b = frozen_scalar(1 + DELTA, "fp32"), frozen_scalar(1 - DELTA, "fp32")
    # 1 - DELTA is a tie below 1; round-to-nearest-even selects 1.
    assert low_a[0].weight.item() == low_b[0].weight.item() == 1
    reference = (high_a(x) + high_b(x)).item()
    assert reference == (low_a(x) + low_b(x)).item() == 2
    assert abs((high_a(x) + low_b(x)).item() - reference) == DELTA
    assert abs((low_a(x) + high_b(x)).item() - reference) == DELTA


def test_downstream_gain_amplifies_local_absolute_error():
    x = torch.ones(1, 1, device="cpu")
    low = frozen_scalar(1 + DELTA)
    high = frozen_scalar(1 + DELTA, "fp32")
    downstream = frozen_scalar(32, "fp32")
    local_error = abs((high(x) - low(x)).item())
    final_error = abs((downstream(high(x)) - downstream(low(x))).item())
    assert final_error == 32 * local_error == 2**-7


def test_one_vjp_direction_cannot_bound_operator_error():
    difference = torch.diag(torch.tensor([0.0, 1000.0], dtype=torch.float64))
    cotangent = torch.tensor([1.0, 0.0], dtype=torch.float64)
    assert torch.linalg.vector_norm(difference.T @ cotangent).item() == 0
    assert torch.linalg.matrix_norm(difference, ord=2).item() == 1000


@pytest.mark.parametrize("zero_up", [True, False])
def test_live_lora_matches_chain_rule_with_fp32_parameters(zero_up):
    from networks.lora_modules.lora import LoRAModule

    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False, device="cpu"))
    model.requires_grad_(False)
    weight = torch.tensor([[1.0, 0.5], [-0.25, 2.0]])
    model[0].weight.copy_(weight)
    install_precision_islands(model, {"0": "fp16"})
    adapter = LoRAModule("math", model[0], lora_dim=1, alpha=1)
    adapter.apply_to()
    adapter.train()
    a = torch.tensor([[0.5, -0.25]])
    b = torch.zeros(2, 1) if zero_up else torch.tensor([[0.25], [-0.5]])
    with torch.no_grad():
        adapter.lora_down.weight.copy_(a)
        adapter.lora_up.weight.copy_(b)
    x = torch.tensor([[2.0, 2.0]], requires_grad=True)
    g = torch.tensor([0.5, -1.0])
    output = model(x)
    (g * output).sum().backward()
    torch.testing.assert_close(output, x @ weight.T + x @ a.T @ b.T, rtol=0, atol=0)
    torch.testing.assert_close(adapter.lora_up.weight.grad, torch.outer(g, a @ x[0]), rtol=0, atol=0)
    torch.testing.assert_close(adapter.lora_down.weight.grad, torch.outer(b.T @ g, x[0]), rtol=0, atol=0)
    torch.testing.assert_close(x.grad[0], weight.T @ g + a.T @ b.T @ g, rtol=0, atol=0)
    assert all(p.dtype == torch.float32 for p in adapter.parameters())
    if zero_up:
        assert adapter.lora_down.weight.grad.count_nonzero().item() == 0
        assert adapter.lora_up.weight.grad.count_nonzero().item() > 0


def test_fp32_lora_receives_downstream_low_precision_gradient_error():
    from networks.lora_modules.lora import LoRAModule

    gradients = {}
    for precision in ("fp16", "fp32"):
        model = frozen_scalar(1, "fp32")
        adapter = LoRAModule("upstream", model[0], lora_dim=1, alpha=1)
        adapter.apply_to()
        adapter.train()
        with torch.no_grad():
            adapter.lora_down.weight.fill_(1)
            adapter.lora_up.weight.zero_()
        downstream = frozen_scalar(1 + DELTA, precision)
        downstream(model(torch.ones(1, 1))).sum().backward()
        gradients[precision] = adapter.lora_up.weight.grad.item()
        assert adapter.lora_down.weight.grad.item() == 0
        assert adapter.lora_up.weight.grad.dtype == torch.float32
    assert gradients["fp32"] - gradients["fp16"] == DELTA


def test_forward_and_backward_bounds_with_exact_rational_arithmetic():
    # Ideal F0(x)=x^2 and F1(x)=3x. Perturb each output, and each VJP.
    f = Fraction
    x = [f(1), f(1), f(3)]
    trial = [f(9, 8)]
    eps = [f(1, 16), f(1, 8)]
    lipschitz, jacobian_lipschitz = [f(4), f(3)], [f(2), f(0)]
    trial.append(trial[0] ** 2 + eps[0])
    trial.append(3 * trial[1] + eps[1])
    assert all(abs(v) <= 2 for v in (x[0], trial[0]))
    error_bound = [abs(trial[0] - x[0])]
    for i in range(2):
        error_bound.append(lipschitz[i] * error_bound[i] + eps[i])
        assert abs(trial[i + 1] - x[i + 1]) <= error_bound[-1]
    assert error_bound[-1] == f(29, 16)
    # loss(z)=z^2/2: terminal gradient error is bounded by forward error.
    grad, trial_grad, bound = x[-1], trial[-1], error_bound[-1]
    eta = [f(1, 32), f(1, 64)]
    for i in (1, 0):
        reference_jacobian = 3 if i == 1 else 2 * x[i]
        trial_jacobian = 3 if i == 1 else 2 * trial[i]
        new_trial_grad = trial_jacobian * trial_grad + eta[i]
        bound = (lipschitz[i] * bound
                 + jacobian_lipschitz[i] * error_bound[i] * abs(grad) + eta[i])
        grad = reference_jacobian * grad
        trial_grad = new_trial_grad
        assert abs(trial_grad - grad) <= bound
