from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from networks.plugins.lokr import autograd as lokr_autograd
from networks.plugins.lokr.autograd import (
    lokr_add_grouped_delta_,
    normalize_lokr_grouped_delta_backward_backend,
)


def _find_lokr_triton_test_device():
    if lokr_autograd.triton is None or not torch.cuda.is_available():
        return None
    for idx in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{idx}")
        if lokr_autograd._device_supports_lokr_triton(device):
            return device
    return None


_LOKR_TRITON_TEST_DEVICE = _find_lokr_triton_test_device()
_FULL_BACKWARD_BACKEND = "triton_grad_w1_w2_grad_x"


def test_normalize_lokr_grouped_delta_full_backward_backend():
    assert (
        normalize_lokr_grouped_delta_backward_backend(_FULL_BACKWARD_BACKEND)
        == _FULL_BACKWARD_BACKEND
    )


def test_lokr_grouped_delta_full_backward_backend_falls_back_on_cpu():
    torch.manual_seed(31)
    factor = 2
    in_dim = 3
    out_dim = 4
    x = torch.randn(2, 5, factor * in_dim)
    w1 = torch.randn(factor, factor)
    w2 = torch.randn(out_dim, in_dim)
    gate = torch.tensor([[0.75]], dtype=torch.float32)
    grad = torch.randn(2, 5, factor * out_dim)
    args = (
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

    actual = lokr_autograd._lokr_add_grouped_delta_backward(
        *args,
        backward_backend=_FULL_BACKWARD_BACKEND,
    )
    expected = lokr_autograd._lokr_add_grouped_delta_backward(
        *args,
        backward_backend="eager",
    )

    for actual_grad, expected_grad in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


def _assert_full_backward_matches_eager_cuda(
    *,
    outer_shape,
    factor,
    in_dim,
    out_dim,
    group_size,
    activation_dtype,
    weight_dtype=torch.float32,
):
    device = _LOKR_TRITON_TEST_DEVICE
    x = torch.randn(
        *outer_shape,
        factor * in_dim,
        device=device,
        dtype=activation_dtype,
        requires_grad=True,
    )
    base_leaf = torch.randn(
        *outer_shape,
        factor * out_dim,
        device=device,
        dtype=activation_dtype,
        requires_grad=True,
    )
    w1 = torch.randn(
        factor,
        factor,
        device=device,
        dtype=weight_dtype,
        requires_grad=True,
    )
    w2 = torch.randn(
        out_dim,
        in_dim,
        device=device,
        dtype=weight_dtype,
        requires_grad=True,
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
        group_size,
        backend="triton",
        backward_backend=_FULL_BACKWARD_BACKEND,
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
        group_size,
        backend="triton",
        backward_backend="eager",
    )
    y_ref.backward(grad)

    torch.testing.assert_close(y, y_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(grads[0], base_ref.grad)
    torch.testing.assert_close(grads[1], x_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(grads[2], w1_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(grads[3], w2_ref.grad, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    ("activation_dtype", "weight_dtype"),
    [
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.bfloat16),
    ],
)
@pytest.mark.skipif(
    _LOKR_TRITON_TEST_DEVICE is None,
    reason="No CUDA device with Triton support available",
)
def test_lokr_grouped_delta_full_backward_matches_eager_cuda(
    activation_dtype,
    weight_dtype,
):
    torch.manual_seed(37)
    _assert_full_backward_matches_eager_cuda(
        outer_shape=(3, 2),
        factor=4,
        in_dim=8,
        out_dim=16,
        group_size=2,
        activation_dtype=activation_dtype,
        weight_dtype=weight_dtype,
    )


@pytest.mark.skipif(
    _LOKR_TRITON_TEST_DEVICE is None,
    reason="No CUDA device with Triton support available",
)
def test_lokr_grouped_delta_full_backward_handles_tail_group_and_tiles_cuda():
    torch.manual_seed(41)
    _assert_full_backward_matches_eager_cuda(
        outer_shape=(17,),
        factor=3,
        in_dim=11,
        out_dim=19,
        group_size=2,
        activation_dtype=torch.bfloat16,
    )


@pytest.mark.skipif(
    _LOKR_TRITON_TEST_DEVICE is None,
    reason="No CUDA device with Triton support available",
)
def test_lokr_grouped_delta_full_backward_rejects_noncontiguous_cuda():
    device = _LOKR_TRITON_TEST_DEVICE
    factor = 2
    in_dim = 3
    out_dim = 5
    x_storage = torch.randn(4, factor * in_dim, 2, device=device)
    x = x_storage[..., 0]
    grad = torch.randn(4, factor * out_dim, device=device)
    w1 = torch.randn(factor, factor, device=device)
    w2 = torch.randn(out_dim, in_dim, device=device)
    gate = torch.ones(1, 1, device=device)

    assert not x.is_contiguous()
    assert (
        not lokr_autograd._can_use_lokr_grouped_delta_backward_triton_grad_w1_w2_grad_x(
            grad,
            x,
            w1,
            w2,
            gate,
            factor,
            in_dim,
            out_dim,
        )
    )


@pytest.mark.skipif(
    _LOKR_TRITON_TEST_DEVICE is None,
    reason="No CUDA device with Triton support available",
)
def test_lokr_grouped_delta_full_backward_checkpoint_recompute_cuda():
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(43)
    device = _LOKR_TRITON_TEST_DEVICE
    factor = 2
    in_dim = 4
    out_dim = 6
    x = torch.randn(
        5,
        factor * in_dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    base_weight = torch.randn(
        factor * out_dim,
        factor * in_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    w1 = torch.randn(
        factor,
        factor,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    w2 = torch.randn(
        out_dim,
        in_dim,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    gate = torch.tensor([[0.75]], device=device, dtype=torch.float32)
    grad = torch.randn(5, factor * out_dim, device=device, dtype=torch.bfloat16)

    def projected(x_arg, w1_arg, w2_arg):
        return lokr_add_grouped_delta_(
            F.linear(x_arg, base_weight),
            x_arg,
            w1_arg,
            w2_arg,
            gate,
            factor,
            in_dim,
            out_dim,
            2,
            backend="triton",
            backward_backend=_FULL_BACKWARD_BACKEND,
        )

    y = checkpoint(projected, x, w1, w2, use_reentrant=False)
    y.backward(grad)
    grads = [x.grad.clone(), w1.grad.clone(), w2.grad.clone()]

    x_ref = x.detach().clone().requires_grad_()
    w1_ref = w1.detach().clone().requires_grad_()
    w2_ref = w2.detach().clone().requires_grad_()
    y_ref = lokr_add_grouped_delta_(
        F.linear(x_ref, base_weight),
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

    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(grads[0], x_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(grads[1], w1_ref.grad, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(grads[2], w2_ref.grad, atol=2e-2, rtol=2e-2)
