from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F

from library.runtime.peak_probe import PeakProbe
from networks.plugins.lokr import autograd as lokr_autograd
from networks.plugins.lokr.autograd import (
    lokr_add_grouped_delta_,
    lokr_project,
    lokr_project_factor,
    lokr_project_factor_group,
    normalize_lokr_grouped_delta_backward_backend,
    normalize_lokr_grouped_delta_backend,
)
from networks.plugins.lokr.module import LoKrModule


def _find_lokr_triton_test_device():
    if lokr_autograd.triton is None or not torch.cuda.is_available():
        return None
    for idx in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{idx}")
        if lokr_autograd._device_supports_lokr_triton(device):
            return device
    return None


_LOKR_TRITON_TEST_DEVICE = _find_lokr_triton_test_device()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "triton"),
        ("", "triton"),
        ("EAGER", "eager"),
        ("triton", "triton"),
    ],
)
def test_normalize_lokr_grouped_delta_backend(value, expected):
    assert normalize_lokr_grouped_delta_backend(value) == expected


def test_normalize_lokr_grouped_delta_backend_rejects_unknown_value():
    with pytest.raises(ValueError, match="unsupported LoKr grouped-delta backend"):
        normalize_lokr_grouped_delta_backend("cuda")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "triton_grad_w1_w2_grad_x"),
        ("", "triton_grad_w1_w2_grad_x"),
        ("EAGER", "eager"),
        ("triton_grad_x", "triton_grad_x"),
        ("triton_grad_w2_partial", "triton_grad_w2_partial"),
        ("triton_grad_w2_grad_x", "triton_grad_w2_grad_x"),
    ],
)
def test_normalize_lokr_grouped_delta_backward_backend(value, expected):
    assert normalize_lokr_grouped_delta_backward_backend(value) == expected


def test_normalize_lokr_grouped_delta_backward_backend_rejects_unknown_value():
    with pytest.raises(
        ValueError, match="unsupported LoKr grouped-delta backward backend"
    ):
        normalize_lokr_grouped_delta_backward_backend("triton")


def test_lokr_eval_forward_ignores_stale_timestep_mask():
    base = torch.nn.Linear(4, 4, bias=False)
    lokr = LoKrModule(
        "lora_unet_test",
        base,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
        factor=2,
        lokr_use_einsum=False,
    )
    lokr.apply_to()

    with torch.no_grad():
        lokr.org_module_ref[0].weight.zero_()
        lokr.lokr_w1.fill_(1.0)
        lokr.lokr_w2.fill_(1.0)
        lokr._timestep_mask.zero_()

    x = torch.ones(1, 4)

    lokr.train()
    train_out = lokr.org_module_ref[0](x)

    lokr.eval()
    eval_out = lokr.org_module_ref[0](x)

    torch.testing.assert_close(train_out, torch.zeros_like(train_out))
    assert torch.count_nonzero(eval_out).item() == eval_out.numel()


def test_lokr_default_einsum_forward_matches_kron_path():
    torch.manual_seed(12)
    x = torch.randn(3, 6, requires_grad=True)
    grad = torch.randn(3, 8)

    base = torch.nn.Linear(6, 8, bias=False)
    lokr = LoKrModule(
        "lora_unet_test",
        base,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
        factor=2,
    )
    lokr.apply_to()
    lokr.train()
    with torch.no_grad():
        lokr.org_module_ref[0].weight.copy_(torch.randn_like(lokr.org_module_ref[0].weight))
        lokr.lokr_w1.copy_(torch.randn_like(lokr.lokr_w1))
        lokr.lokr_w2.copy_(torch.randn_like(lokr.lokr_w2))

    y = lokr.org_module_ref[0](x)
    y.backward(grad)
    grads = [
        x.grad.clone(),
        lokr.lokr_w1.grad.clone(),
        lokr.lokr_w2.grad.clone(),
    ]

    x_ref = x.detach().clone().requires_grad_()
    w1_ref = lokr.lokr_w1.detach().clone().requires_grad_()
    w2_ref = lokr.lokr_w2.detach().clone().requires_grad_()
    org_weight = lokr.org_module_ref[0].weight.detach()
    y_ref = F.linear(x_ref, org_weight) + F.linear(
        x_ref.to(y.dtype), torch.kron(w1_ref, w2_ref).to(y.dtype)
    ).to(y.dtype)
    y_ref.backward(grad)

    assert lokr.lokr_use_einsum is True
    assert hasattr(lokr, "lokr_w2")
    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(grads[0], x_ref.grad)
    torch.testing.assert_close(grads[1], w1_ref.grad)
    torch.testing.assert_close(grads[2], w2_ref.grad)


def test_lokr_decomposed_w2_einsum_forward_matches_kron_path():
    torch.manual_seed(12)
    x = torch.randn(3, 6, requires_grad=True)
    grad = torch.randn(3, 8)

    base = torch.nn.Linear(6, 8, bias=False)
    lokr = LoKrModule(
        "lora_unet_test",
        base,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
        factor=2,
        lokr_decompose_w2=True,
    )
    lokr.apply_to()
    lokr.train()
    with torch.no_grad():
        lokr.org_module_ref[0].weight.copy_(torch.randn_like(lokr.org_module_ref[0].weight))
        lokr.lokr_w1.copy_(torch.randn_like(lokr.lokr_w1))
        lokr.lokr_w2_a.copy_(torch.randn_like(lokr.lokr_w2_a))
        lokr.lokr_w2_b.copy_(torch.randn_like(lokr.lokr_w2_b))

    y = lokr.org_module_ref[0](x)
    y.backward(grad)
    grads = [
        x.grad.clone(),
        lokr.lokr_w1.grad.clone(),
        lokr.lokr_w2_a.grad.clone(),
        lokr.lokr_w2_b.grad.clone(),
    ]

    x_ref = x.detach().clone().requires_grad_()
    w1_ref = lokr.lokr_w1.detach().clone().requires_grad_()
    w2a_ref = lokr.lokr_w2_a.detach().clone().requires_grad_()
    w2b_ref = lokr.lokr_w2_b.detach().clone().requires_grad_()
    org_weight = lokr.org_module_ref[0].weight.detach()
    w2_ref = w2a_ref @ w2b_ref
    y_ref = F.linear(x_ref, org_weight) + F.linear(
        x_ref.to(y.dtype), torch.kron(w1_ref, w2_ref).to(y.dtype)
    ).to(y.dtype)
    y_ref.backward(grad)

    assert lokr.lokr_use_einsum is True
    assert not hasattr(lokr, "lokr_w2")
    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(grads[0], x_ref.grad)
    torch.testing.assert_close(grads[1], w1_ref.grad)
    torch.testing.assert_close(grads[2], w2a_ref.grad)
    torch.testing.assert_close(grads[3], w2b_ref.grad)


def test_lokr_explicit_triton_training_uses_default_backward_backend(monkeypatch):
    torch.manual_seed(14)
    x = torch.randn(3, 6, requires_grad=True)

    base = torch.nn.Linear(6, 8, bias=False)
    lokr = LoKrModule(
        "lora_unet_test",
        base,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
        factor=2,
        lokr_grouped_delta_backend="triton",
    )
    lokr.apply_to()
    lokr.train()

    calls = []
    original = lokr_autograd.lokr_add_grouped_delta_

    def _tracking_project(*args, **kwargs):
        calls.append(
            (
                args[5],
                args[6],
                args[7],
                args[8],
                kwargs.get("backend"),
                kwargs.get("backward_backend"),
            )
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "networks.plugins.lokr.module.lokr_add_grouped_delta_",
        _tracking_project,
    )

    y = lokr.org_module_ref[0](x)
    y.sum().backward()

    assert lokr.lokr_use_einsum is True
    assert calls == [(2, 3, 4, 2, "triton", "triton_grad_w1_w2_grad_x")]
    assert x.grad is not None
    assert lokr.lokr_w1.grad is not None
    assert lokr.lokr_w2.grad is not None


def test_lokr_default_training_uses_grouped_delta_fast_path(monkeypatch):
    torch.manual_seed(15)
    x = torch.randn(3, 6, requires_grad=True)

    base = torch.nn.Linear(6, 8, bias=False)
    lokr = LoKrModule(
        "lora_unet_test",
        base,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
        factor=2,
    )
    lokr.apply_to()
    lokr.train()

    calls = []
    original = lokr_autograd.lokr_add_grouped_delta_

    def _tracking_project(*args, **kwargs):
        calls.append((kwargs.get("backend"), kwargs.get("backward_backend")))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "networks.plugins.lokr.module.lokr_add_grouped_delta_",
        _tracking_project,
    )

    y = lokr.org_module_ref[0](x)
    y.sum().backward()

    assert lokr.lokr_grouped_delta_backend == "triton"
    assert (
        lokr.lokr_grouped_delta_backward_backend
        == "triton_grad_w1_w2_grad_x"
    )
    assert calls == [("triton", "triton_grad_w1_w2_grad_x")]
    assert x.grad is not None
    assert lokr.lokr_w1.grad is not None
    assert lokr.lokr_w2.grad is not None


def test_lokr_project_matches_kron_linear_forward_and_backward():
    torch.manual_seed(1)
    factor = 2
    in_dim = 3
    out_dim = 4
    x = torch.randn(2, 5, factor * in_dim, requires_grad=True)
    w1 = torch.randn(factor, factor, requires_grad=True)
    w2 = torch.randn(out_dim, in_dim, requires_grad=True)
    grad = torch.randn(2, 5, factor * out_dim)

    y = lokr_project(x, w1, w2, factor, in_dim, out_dim)
    y.backward(grad)
    grads = [x.grad.clone(), w1.grad.clone(), w2.grad.clone()]

    x_ref = x.detach().clone().requires_grad_()
    w1_ref = w1.detach().clone().requires_grad_()
    w2_ref = w2.detach().clone().requires_grad_()
    y_ref = F.linear(x_ref, torch.kron(w1_ref, w2_ref))
    y_ref.backward(grad)

    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(grads[0], x_ref.grad)
    torch.testing.assert_close(grads[1], w1_ref.grad)
    torch.testing.assert_close(grads[2], w2_ref.grad)


def test_lokr_project_does_not_materialize_kron(monkeypatch):
    factor = 2
    in_dim = 3
    out_dim = 4
    x = torch.randn(2, factor * in_dim, requires_grad=True)
    w1 = torch.randn(factor, factor, requires_grad=True)
    w2 = torch.randn(out_dim, in_dim, requires_grad=True)

    def _fail_kron(*_args, **_kwargs):
        raise AssertionError("lokr_project must not materialize torch.kron")

    monkeypatch.setattr(torch, "kron", _fail_kron)

    y = lokr_project(x, w1, w2, factor, in_dim, out_dim)
    y.sum().backward()

    assert y.shape == (2, factor * out_dim)
    assert x.grad is not None
    assert w1.grad is not None
    assert w2.grad is not None


def test_lokr_project_factor_chunked_matches_kron_slice(monkeypatch):
    torch.manual_seed(3)
    factor = 2
    in_dim = 3
    out_dim = 4
    out_factor = 1
    x = torch.randn(5, factor * in_dim, requires_grad=True)
    w1 = torch.randn(factor, factor, requires_grad=True)
    w2 = torch.randn(out_dim, in_dim, requires_grad=True)
    grad = torch.randn(5, out_dim)

    monkeypatch.setattr(lokr_autograd, "_LOKR_PROJECT_CHUNK_BYTES", 1)

    y = lokr_project_factor(x, w1, w2, out_factor, factor, in_dim, out_dim)
    y.backward(grad)
    grads = [x.grad.clone(), w1.grad.clone(), w2.grad.clone()]

    x_ref = x.detach().clone().requires_grad_()
    w1_ref = w1.detach().clone().requires_grad_()
    w2_ref = w2.detach().clone().requires_grad_()
    y_ref_full = F.linear(x_ref, torch.kron(w1_ref, w2_ref))
    start = out_factor * out_dim
    y_ref = y_ref_full[:, start : start + out_dim]
    y_ref.backward(grad)

    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(grads[0], x_ref.grad)
    torch.testing.assert_close(grads[1], w1_ref.grad)
    torch.testing.assert_close(grads[2], w2_ref.grad)


def test_lokr_project_factor_group_matches_kron_slice(monkeypatch):
    torch.manual_seed(5)
    factor = 4
    in_dim = 3
    out_dim = 4
    out_start = 1
    out_count = 2
    x = torch.randn(2, 5, factor * in_dim, requires_grad=True)
    w1 = torch.randn(factor, factor, requires_grad=True)
    w2 = torch.randn(out_dim, in_dim, requires_grad=True)
    grad = torch.randn(2, 5, out_count * out_dim)

    monkeypatch.setattr(lokr_autograd, "_LOKR_PROJECT_CHUNK_BYTES", 1)

    y = lokr_project_factor_group(
        x, w1, w2, out_start, out_count, factor, in_dim, out_dim
    )
    y.backward(grad)
    grads = [x.grad.clone(), w1.grad.clone(), w2.grad.clone()]

    x_ref = x.detach().clone().requires_grad_()
    w1_ref = w1.detach().clone().requires_grad_()
    w2_ref = w2.detach().clone().requires_grad_()
    y_ref_full = F.linear(x_ref, torch.kron(w1_ref, w2_ref))
    start = out_start * out_dim
    y_ref = y_ref_full[..., start : start + out_count * out_dim]
    y_ref.backward(grad)

    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(grads[0], x_ref.grad)
    torch.testing.assert_close(grads[1], w1_ref.grad)
    torch.testing.assert_close(grads[2], w2_ref.grad)


def test_lokr_add_grouped_delta_matches_kron_forward_and_backward(monkeypatch):
    torch.manual_seed(7)
    factor = 4
    in_dim = 3
    out_dim = 4
    x = torch.randn(2, 5, factor * in_dim, requires_grad=True)
    base_leaf = torch.randn(2, 5, factor * out_dim, requires_grad=True)
    w1 = torch.randn(factor, factor, requires_grad=True)
    w2 = torch.randn(out_dim, in_dim, requires_grad=True)
    gate = torch.tensor([[0.75]], dtype=torch.float32)
    grad = torch.randn(2, 5, factor * out_dim)

    monkeypatch.setattr(lokr_autograd, "_LOKR_PROJECT_CHUNK_BYTES", 1)

    y = lokr_add_grouped_delta_(
        base_leaf + 0,
        x,
        w1,
        w2,
        gate,
        factor,
        in_dim,
        out_dim,
        2,
    )
    y.backward(grad)
    grads = [base_leaf.grad.clone(), x.grad.clone(), w1.grad.clone(), w2.grad.clone()]

    base_ref = base_leaf.detach().clone().requires_grad_()
    x_ref = x.detach().clone().requires_grad_()
    w1_ref = w1.detach().clone().requires_grad_()
    w2_ref = w2.detach().clone().requires_grad_()
    y_ref = base_ref + F.linear(x_ref.float(), torch.kron(w1_ref, w2_ref)) * gate
    y_ref.backward(grad)

    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(grads[0], base_ref.grad)
    torch.testing.assert_close(grads[1], x_ref.grad)
    torch.testing.assert_close(grads[2], w1_ref.grad)
    torch.testing.assert_close(grads[3], w2_ref.grad)


def test_lokr_add_grouped_delta_backward_emits_phase_ranges_when_enabled(monkeypatch):
    torch.manual_seed(8)
    factor = 2
    in_dim = 3
    out_dim = 4
    x = torch.randn(2, 5, factor * in_dim)
    w1 = torch.randn(factor, factor)
    w2 = torch.randn(out_dim, in_dim)
    gate = torch.tensor([[0.75]], dtype=torch.float32)
    grad = torch.randn(2, 5, factor * out_dim)
    seen: list[str] = []

    class _CaptureRange:
        def __init__(self, name: str):
            self.name = name

        def __enter__(self):
            seen.append(self.name)
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(lokr_autograd, "_LOKR_ENABLE_BACKWARD_PHASE_RANGES", True)
    monkeypatch.setattr(
        lokr_autograd,
        "_lokr_record_function",
        lambda name: _CaptureRange(name),
    )

    grad_x, grad_w1, grad_w2 = lokr_autograd._lokr_add_grouped_delta_backward(
        grad,
        x,
        w1,
        w2,
        gate,
        factor,
        in_dim,
        out_dim,
        2,
        1024,
        x.shape[:-1],
    )

    assert grad_x.shape == x.shape
    assert grad_w1.shape == w1.shape
    assert grad_w2.shape == w2.shape
    assert set(lokr_autograd._LOKR_BACKWARD_PHASE_NAMES).issubset(set(seen))


def test_lokr_add_grouped_delta_triton_backend_falls_back_to_eager(monkeypatch):
    base = torch.zeros(1, 8)
    x = torch.zeros(1, 6)
    w1 = torch.zeros(2, 2)
    w2 = torch.zeros(4, 3)
    gate = torch.ones(1, 1)
    calls = []

    monkeypatch.setattr(
        lokr_autograd,
        "_can_use_lokr_grouped_delta_triton",
        lambda *_args, **_kwargs: False,
    )

    def _fake_eager(*args):
        calls.append("eager")
        return args[0]

    def _fail_triton(*_args):
        raise AssertionError("unexpected Triton dispatch")

    monkeypatch.setattr(
        lokr_autograd.LoKrAddGroupedDeltaFn, "apply", staticmethod(_fake_eager)
    )
    monkeypatch.setattr(
        lokr_autograd.LoKrAddGroupedDeltaTritonFn,
        "apply",
        staticmethod(_fail_triton),
    )

    out = lokr_add_grouped_delta_(
        base,
        x,
        w1,
        w2,
        gate,
        2,
        3,
        4,
        2,
        backend="triton",
    )

    assert out is base
    assert calls == ["eager"]


def test_lokr_add_grouped_delta_triton_backend_dispatches_when_available(monkeypatch):
    base = torch.zeros(1, 8)
    x = torch.zeros(1, 6)
    w1 = torch.zeros(2, 2)
    w2 = torch.zeros(4, 3)
    gate = torch.ones(1, 1)
    calls = []

    monkeypatch.setattr(
        lokr_autograd,
        "_can_use_lokr_grouped_delta_triton",
        lambda *_args, **_kwargs: True,
    )

    def _fail_eager(*_args):
        raise AssertionError("unexpected eager fallback")

    def _fake_triton(*args):
        calls.append("triton")
        return args[0]

    monkeypatch.setattr(
        lokr_autograd.LoKrAddGroupedDeltaFn, "apply", staticmethod(_fail_eager)
    )
    monkeypatch.setattr(
        lokr_autograd.LoKrAddGroupedDeltaTritonFn,
        "apply",
        staticmethod(_fake_triton),
    )

    out = lokr_add_grouped_delta_(
        base,
        x,
        w1,
        w2,
        gate,
        2,
        3,
        4,
        2,
        backend="triton",
    )

    assert out is base
    assert calls == ["triton"]


def test_lokr_module_custom_forward_matches_kron_path():
    torch.manual_seed(2)
    x = torch.randn(3, 6, requires_grad=True)
    grad = torch.randn(3, 8)

    base = torch.nn.Linear(6, 8, bias=False)
    lokr = LoKrModule(
        "lora_unet_test",
        base,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
        factor=2,
        lokr_use_einsum=False,
    )
    lokr.apply_to()
    lokr.train()
    lokr.use_custom_lokr_autograd = True
    with torch.no_grad():
        lokr.org_module_ref[0].weight.copy_(torch.randn_like(lokr.org_module_ref[0].weight))
        lokr.lokr_w1.copy_(torch.randn_like(lokr.lokr_w1))
        lokr.lokr_w2.copy_(torch.randn_like(lokr.lokr_w2))

    y = lokr.org_module_ref[0](x)
    y.backward(grad)
    grads = [x.grad.clone(), lokr.lokr_w1.grad.clone(), lokr.lokr_w2.grad.clone()]

    x_ref = x.detach().clone().requires_grad_()
    w1_ref = lokr.lokr_w1.detach().clone().requires_grad_()
    w2_ref = lokr.lokr_w2.detach().clone().requires_grad_()
    org_weight = lokr.org_module_ref[0].weight.detach()
    y_ref = F.linear(x_ref, org_weight) + F.linear(x_ref.float(), torch.kron(w1_ref, w2_ref))
    y_ref.backward(grad)

    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(grads[0], x_ref.grad)
    torch.testing.assert_close(grads[1], w1_ref.grad)
    torch.testing.assert_close(grads[2], w2_ref.grad)


def test_lokr_module_custom_forward_reuses_base_output(monkeypatch):
    torch.manual_seed(4)
    x = torch.randn(3, 6, requires_grad=True)

    base = torch.nn.Linear(6, 8, bias=False)
    lokr = LoKrModule(
        "lora_unet_test",
        base,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
        factor=2,
        lokr_use_einsum=False,
    )
    lokr.apply_to()
    lokr.train()
    lokr.use_custom_lokr_autograd = True

    def _fail_empty_like(*_args, **_kwargs):
        raise AssertionError("custom LoKr forward must not allocate a full result")

    monkeypatch.setattr(torch, "empty_like", _fail_empty_like)

    y = lokr.org_module_ref[0](x)
    y.sum().backward()

    assert y.shape == (3, 8)
    assert x.grad is not None
    assert lokr.lokr_w1.grad is not None
    assert lokr.lokr_w2.grad is not None


def test_lokr_module_custom_forward_uses_configured_factor_group(monkeypatch):
    torch.manual_seed(6)
    x = torch.randn(3, 16, requires_grad=True)

    base = torch.nn.Linear(16, 16, bias=False)
    lokr = LoKrModule(
        "lora_unet_test",
        base,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
        factor=4,
        lokr_factor_group_size=2,
        lokr_project_chunk_bytes=1234,
        lokr_grouped_delta_backend="triton",
        lokr_grouped_delta_backward_backend="triton_grad_x",
        lokr_use_einsum=False,
    )
    lokr.apply_to()
    lokr.train()
    lokr.use_custom_lokr_autograd = True

    calls = []
    original = lokr_autograd.lokr_add_grouped_delta_

    def _tracking_project(*args, **kwargs):
        calls.append(
            (
                args[8],
                args[9],
                kwargs.get("backend"),
                kwargs.get("backward_backend"),
            )
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "networks.plugins.lokr.module.lokr_add_grouped_delta_",
        _tracking_project,
    )

    y = lokr.org_module_ref[0](x)
    y.sum().backward()

    assert calls == [(2, 1234, "triton", "triton_grad_x")]


def test_lokr_module_custom_forward_accepts_triton_grad_w2_partial(monkeypatch):
    torch.manual_seed(9)
    x = torch.randn(3, 16, requires_grad=True)

    base = torch.nn.Linear(16, 16, bias=False)
    lokr = LoKrModule(
        "lora_unet_test",
        base,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
        factor=4,
        lokr_factor_group_size=2,
        lokr_project_chunk_bytes=2048,
        lokr_grouped_delta_backend="triton",
        lokr_grouped_delta_backward_backend="triton_grad_w2_partial",
        lokr_use_einsum=False,
    )
    lokr.apply_to()
    lokr.train()
    lokr.use_custom_lokr_autograd = True

    calls = []
    original = lokr_autograd.lokr_add_grouped_delta_

    def _tracking_project(*args, **kwargs):
        calls.append(
            (
                args[8],
                args[9],
                kwargs.get("backend"),
                kwargs.get("backward_backend"),
            )
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "networks.plugins.lokr.module.lokr_add_grouped_delta_",
        _tracking_project,
    )

    y = lokr.org_module_ref[0](x)
    y.sum().backward()

    assert calls == [(2, 2048, "triton", "triton_grad_w2_partial")]


def test_lokr_module_custom_forward_accepts_triton_grad_w2_grad_x(monkeypatch):
    torch.manual_seed(10)
    x = torch.randn(3, 16, requires_grad=True)

    base = torch.nn.Linear(16, 16, bias=False)
    lokr = LoKrModule(
        "lora_unet_test",
        base,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
        factor=4,
        lokr_factor_group_size=2,
        lokr_project_chunk_bytes=2048,
        lokr_grouped_delta_backend="triton",
        lokr_grouped_delta_backward_backend="triton_grad_w2_grad_x",
        lokr_use_einsum=False,
    )
    lokr.apply_to()
    lokr.train()
    lokr.use_custom_lokr_autograd = True

    calls = []
    original = lokr_autograd.lokr_add_grouped_delta_

    def _tracking_project(*args, **kwargs):
        calls.append(
            (
                args[8],
                args[9],
                kwargs.get("backend"),
                kwargs.get("backward_backend"),
            )
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "networks.plugins.lokr.module.lokr_add_grouped_delta_",
        _tracking_project,
    )

    y = lokr.org_module_ref[0](x)
    y.sum().backward()

    assert calls == [(2, 2048, "triton", "triton_grad_w2_grad_x")]


@pytest.mark.skipif(
    _LOKR_TRITON_TEST_DEVICE is None,
    reason="No CUDA device with Triton support available",
)
def test_lokr_add_grouped_delta_triton_matches_eager_cuda():
    torch.manual_seed(11)
    device = _LOKR_TRITON_TEST_DEVICE
    factor = 4
    in_dim = 8
    out_dim = 16
    x = torch.randn(3, 2, factor * in_dim, device=device, dtype=torch.float16, requires_grad=True)
    base_leaf = torch.randn(
        3, 2, factor * out_dim, device=device, dtype=torch.float16, requires_grad=True
    )
    w1 = torch.randn(factor, factor, device=device, dtype=torch.float32, requires_grad=True)
    w2 = torch.randn(out_dim, in_dim, device=device, dtype=torch.float32, requires_grad=True)
    gate = torch.tensor([[0.75]], device=device, dtype=torch.float32)
    grad = torch.randn_like(base_leaf)

    y = lokr_add_grouped_delta_(
        base_leaf + 0,
        x,
        w1,
        w2,
        gate,
        factor,
        in_dim,
        out_dim,
        2,
        backend="triton",
    )
    y.backward(grad)
    grads = [base_leaf.grad.clone(), x.grad.clone(), w1.grad.clone(), w2.grad.clone()]

    base_ref = base_leaf.detach().clone().requires_grad_()
    x_ref = x.detach().clone().requires_grad_()
    w1_ref = w1.detach().clone().requires_grad_()
    w2_ref = w2.detach().clone().requires_grad_()
    y_ref = lokr_add_grouped_delta_(
        base_ref + 0,
        x_ref,
        w1_ref,
        w2_ref,
        gate,
        factor,
        in_dim,
        out_dim,
        2,
        backend="eager",
    )
    y_ref.backward(grad)

    torch.testing.assert_close(y, y_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(grads[0], base_ref.grad)
    torch.testing.assert_close(grads[1], x_ref.grad)
    torch.testing.assert_close(grads[2], w1_ref.grad)
    torch.testing.assert_close(grads[3], w2_ref.grad)


@pytest.mark.skipif(
    _LOKR_TRITON_TEST_DEVICE is None,
    reason="No CUDA device with Triton support available",
)
def test_lokr_add_grouped_delta_triton_grad_x_backward_matches_eager_cuda():
    torch.manual_seed(17)
    device = _LOKR_TRITON_TEST_DEVICE
    factor = 4
    in_dim = 8
    out_dim = 16
    x = torch.randn(3, 2, factor * in_dim, device=device, dtype=torch.float16, requires_grad=True)
    base_leaf = torch.randn(
        3, 2, factor * out_dim, device=device, dtype=torch.float16, requires_grad=True
    )
    w1 = torch.randn(factor, factor, device=device, dtype=torch.float32, requires_grad=True)
    w2 = torch.randn(out_dim, in_dim, device=device, dtype=torch.float32, requires_grad=True)
    gate = torch.tensor([[0.75]], device=device, dtype=torch.float32)
    grad = torch.randn_like(base_leaf)

    y = lokr_add_grouped_delta_(
        base_leaf + 0,
        x,
        w1,
        w2,
        gate,
        factor,
        in_dim,
        out_dim,
        2,
        backend="triton",
        backward_backend="triton_grad_x",
    )
    y.backward(grad)
    grads = [base_leaf.grad.clone(), x.grad.clone(), w1.grad.clone(), w2.grad.clone()]

    base_ref = base_leaf.detach().clone().requires_grad_()
    x_ref = x.detach().clone().requires_grad_()
    w1_ref = w1.detach().clone().requires_grad_()
    w2_ref = w2.detach().clone().requires_grad_()
    y_ref = lokr_add_grouped_delta_(
        base_ref + 0,
        x_ref,
        w1_ref,
        w2_ref,
        gate,
        factor,
        in_dim,
        out_dim,
        2,
        backend="triton",
        backward_backend="eager",
    )
    y_ref.backward(grad)

    torch.testing.assert_close(y, y_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(grads[0], base_ref.grad)
    torch.testing.assert_close(grads[1], x_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(grads[2], w1_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(grads[3], w2_ref.grad, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(
    _LOKR_TRITON_TEST_DEVICE is None,
    reason="No CUDA device with Triton support available",
)
def test_lokr_add_grouped_delta_triton_grad_w2_partial_backward_matches_eager_cuda():
    torch.manual_seed(19)
    device = _LOKR_TRITON_TEST_DEVICE
    factor = 4
    in_dim = 8
    out_dim = 16
    x = torch.randn(3, 2, factor * in_dim, device=device, dtype=torch.float16, requires_grad=True)
    base_leaf = torch.randn(
        3, 2, factor * out_dim, device=device, dtype=torch.float16, requires_grad=True
    )
    w1 = torch.randn(factor, factor, device=device, dtype=torch.float32, requires_grad=True)
    w2 = torch.randn(out_dim, in_dim, device=device, dtype=torch.float32, requires_grad=True)
    gate = torch.tensor([[0.75]], device=device, dtype=torch.float32)
    grad = torch.randn_like(base_leaf)

    y = lokr_add_grouped_delta_(
        base_leaf + 0,
        x,
        w1,
        w2,
        gate,
        factor,
        in_dim,
        out_dim,
        2,
        backend="triton",
        backward_backend="triton_grad_w2_partial",
    )
    y.backward(grad)
    grads = [base_leaf.grad.clone(), x.grad.clone(), w1.grad.clone(), w2.grad.clone()]

    base_ref = base_leaf.detach().clone().requires_grad_()
    x_ref = x.detach().clone().requires_grad_()
    w1_ref = w1.detach().clone().requires_grad_()
    w2_ref = w2.detach().clone().requires_grad_()
    y_ref = lokr_add_grouped_delta_(
        base_ref + 0,
        x_ref,
        w1_ref,
        w2_ref,
        gate,
        factor,
        in_dim,
        out_dim,
        2,
        backend="triton",
        backward_backend="eager",
    )
    y_ref.backward(grad)

    torch.testing.assert_close(y, y_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(grads[0], base_ref.grad)
    torch.testing.assert_close(grads[1], x_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(grads[2], w1_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(grads[3], w2_ref.grad, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(
    _LOKR_TRITON_TEST_DEVICE is None,
    reason="No CUDA device with Triton support available",
)
def test_lokr_add_grouped_delta_triton_grad_w2_grad_x_backward_matches_eager_cuda():
    torch.manual_seed(23)
    device = _LOKR_TRITON_TEST_DEVICE
    factor = 4
    in_dim = 8
    out_dim = 16
    x = torch.randn(
        3,
        2,
        factor * in_dim,
        device=device,
        dtype=torch.float16,
        requires_grad=True,
    )
    base_leaf = torch.randn(
        3, 2, factor * out_dim, device=device, dtype=torch.float16, requires_grad=True
    )
    w1 = torch.randn(
        factor, factor, device=device, dtype=torch.float32, requires_grad=True
    )
    w2 = torch.randn(
        out_dim, in_dim, device=device, dtype=torch.float32, requires_grad=True
    )
    gate = torch.tensor([[0.75]], device=device, dtype=torch.float32)
    grad = torch.randn_like(base_leaf)

    y = lokr_add_grouped_delta_(
        base_leaf + 0,
        x,
        w1,
        w2,
        gate,
        factor,
        in_dim,
        out_dim,
        2,
        backend="triton",
        backward_backend="triton_grad_w2_grad_x",
    )
    y.backward(grad)
    grads = [base_leaf.grad.clone(), x.grad.clone(), w1.grad.clone(), w2.grad.clone()]

    base_ref = base_leaf.detach().clone().requires_grad_()
    x_ref = x.detach().clone().requires_grad_()
    w1_ref = w1.detach().clone().requires_grad_()
    w2_ref = w2.detach().clone().requires_grad_()
    y_ref = lokr_add_grouped_delta_(
        base_ref + 0,
        x_ref,
        w1_ref,
        w2_ref,
        gate,
        factor,
        in_dim,
        out_dim,
        2,
        backend="triton",
        backward_backend="eager",
    )
    y_ref.backward(grad)

    torch.testing.assert_close(y, y_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(grads[0], base_ref.grad)
    torch.testing.assert_close(grads[1], x_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(grads[2], w1_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(grads[3], w2_ref.grad, atol=2e-2, rtol=2e-2)


def test_lokr_peak_probe_records_delta_events(tmp_path):
    x = torch.randn(2, 4)
    base = torch.nn.Linear(4, 4, bias=False)
    lokr = LoKrModule(
        "lora_unet_blocks_3_mlp_layer1",
        base,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
        factor=2,
    )
    lokr.original_name = "blocks.3.mlp.layer1"
    lokr.apply_to()
    lokr.train()
    lokr.use_custom_lokr_autograd = True
    lokr._peak_probe = PeakProbe(str(tmp_path / "peak.jsonl"), max_steps=0, level="full")

    lokr._peak_probe.begin_step(0)
    _ = lokr.org_module_ref[0](x)
    lokr._peak_probe.end_step()

    rows = [
        json.loads(line)
        for line in (tmp_path / "peak.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    labels = [row.get("label") for row in rows]
    assert "lokr_after_base" in labels
    assert "lokr_before_delta_apply" in labels
    assert "lokr_after_delta_apply" in labels
    delta = next(row for row in rows if row.get("label") == "lokr_after_delta_apply")
    assert delta["block_idx"] == 3
    assert delta["block_phase"] == "mlp"
    assert delta["op_name"] == "mlp_layer1"


def test_lokr_full_factor_keeps_full_w2_independent_of_lora_dim():
    base = torch.nn.Linear(2048, 2048, bias=False)
    m = LoKrModule(
        "test",
        base,
        lora_dim=32,
        alpha=32,
        factor=8,
        lokr_full_factor=True,
        lokr_decompose_w2=False,
    )
    assert m.lokr_full_factor is True
    assert m._use_decomposed_w2 is False
    assert m.lokr_w1.shape == (8, 8)
    assert m.lokr_w2.shape == (256, 256)
    assert m.scale == 1.0


def test_lokr_without_full_factor_can_decompose_w2():
    base = torch.nn.Linear(2048, 2048, bias=False)
    m = LoKrModule(
        "test",
        base,
        lora_dim=32,
        alpha=32,
        factor=8,
        lokr_full_factor=False,
        lokr_decompose_w2=True,
    )
    assert m._use_decomposed_w2 is True
    assert m.lokr_w2_a.shape == (256, 32)
    assert m.lokr_w2_b.shape == (32, 256)


def test_lokr_full_factor_and_decompose_w2_are_exclusive():
    base = torch.nn.Linear(64, 64, bias=False)
    with pytest.raises(ValueError, match="mutually exclusive"):
        LoKrModule(
            "test",
            base,
            lora_dim=4,
            alpha=4,
            factor=4,
            lokr_full_factor=True,
            lokr_decompose_w2=True,
        )
