"""Real CUDA dX and nontrivial-scale contracts, explicit opt-in."""

import os
import pytest
import torch
from library.runtime.convrot.experimental_policy import experimental_w8a8_forward
from library.runtime.convrot.fused import fused_w8a8_forward
from library.runtime.convrot.gemm import quantize_activation_absmax_int8
from library.runtime.convrot.triton_kernels import quantize_rowwise
from library.runtime.convrot.rht import group_rht, normalized_hadamard

pytestmark = pytest.mark.skipif(
    os.environ.get("CONVROT_RUN_CUDA_TESTS") != "1",
    reason="explicit CUDA opt-in required",
)


@pytest.mark.parametrize("layout", ["kn", "nk"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("policy", ["triton_quant", "triton_gemm", "triton_both"])
def test_engineering_exact_forward_dx(layout, dtype, policy):
    torch.manual_seed(35)
    x = torch.randn(2, 17, 256, device="cuda", dtype=dtype)
    w = torch.randint(-120, 121, (256, 384), device="cuda", dtype=torch.int8)
    if layout == "nk":
        w = w.T.contiguous()
    scale = torch.rand(384, device="cuda") * 0.01
    h = normalized_hadamard(256, device=x.device, dtype=torch.bfloat16)
    a = x.clone().requires_grad_()
    b = x.clone().requires_grad_()
    ya = fused_w8a8_forward(
        a, w, scale, group_size=256, hadamard=h, weight_layout=layout
    )
    yb = experimental_w8a8_forward(
        b,
        w,
        scale,
        group_size=256,
        hadamard=h,
        weight_layout=layout,
        forward_policy=policy,
    )
    dy = torch.randn_like(ya)
    ya.backward(dy)
    yb.backward(dy)
    assert torch.equal(ya, yb)
    assert torch.equal(a.grad, b.grad)


@pytest.mark.parametrize("layout", ["kn", "nk"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_int8_dx_matches_declared_formula(layout, dtype):
    torch.manual_seed(36)
    x = torch.randn(2, 17, 256, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randint(-120, 121, (256, 384), device="cuda", dtype=torch.int8)
    if layout == "nk":
        w = w.T.contiguous()
    scale = torch.rand(384, device="cuda") * 0.01
    h = normalized_hadamard(256, device=x.device, dtype=torch.bfloat16)
    y = experimental_w8a8_forward(
        x,
        w,
        scale,
        group_size=256,
        hadamard=h,
        weight_layout=layout,
        backward_policy="int8",
    )
    dy = torch.randn_like(y)
    y.backward(dy)
    folded = dy.to(torch.bfloat16).float() * scale
    q, s = quantize_activation_absmax_int8(folded)
    right = w.T if layout == "kn" else w
    # Torch CUDA has no generic int32 addmm. CPU integer reference avoids
    # borrowing the Triton implementation under test or float dot rounding.
    acc = (q.reshape(-1, 384).cpu().int() @ right.cpu().int()).to(
        device=x.device, dtype=torch.float32
    )
    rot = (acc * s.reshape(-1, 1)).to(torch.bfloat16).reshape(2, 17, 256)
    expected = group_rht(rot, 256, hadamard=h).to(dtype)
    assert torch.equal(x.grad, expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_nontrivial_scale_half_integer_neighborhood(dtype):
    # max fixes scale near .0037; adjacent FP32 numbers probe rounding ties.
    scale = 0.0037
    values = torch.tensor([127.0, -3.5, -2.5, -0.5, 0.5, 2.5, 3.5]) * scale
    values = torch.cat(
        [
            values,
            torch.nextafter(values, torch.full_like(values, float("inf"))),
            torch.nextafter(values, torch.full_like(values, -float("inf"))),
        ]
    )
    x = values.to(device="cuda", dtype=dtype).reshape(1, -1)
    q, s = quantize_rowwise(x)
    qr, sr = quantize_activation_absmax_int8(x)
    assert torch.equal(s, sr) and torch.equal(q, qr)
