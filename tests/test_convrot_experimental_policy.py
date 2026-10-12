"""CPU contracts only; these tests do not validate CUDA/Triton numerical gates."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from library.runtime.convrot.apply import apply_convrot_to_lora_network
from library.runtime.convrot.experimental_policy import (
    apply_experimental_policy,
    experimental_w8a8_forward,
)
from library.runtime.convrot.fused import fused_w8a8_forward
from library.runtime.convrot.rht import group_rht, normalized_hadamard


class _LoRA(nn.Module):
    def __init__(self, base, index=0):
        super().__init__()
        self.org_module_ref = [base]
        self.org_forward = base.forward
        self.original_name = f"blocks.{index}.mlp.layer1"
        self.lora_name = f"lora_{index}"
        self.down = nn.Linear(base.in_features, 2, bias=False)
        self.up = nn.Linear(2, base.out_features, bias=False)

    def forward(self, x):
        return self.org_forward(x) + self.up(self.down(x))


def _network(mode="w8a8"):
    base = nn.Linear(16, 8, bias=False)
    base.weight.requires_grad_(False)
    net = nn.Module()
    net.unet_loras = nn.ModuleList([_LoRA(base)])
    apply_convrot_to_lora_network(net, mode=mode, group_size=4)
    return net, base


@pytest.mark.parametrize("layout", ["kn", "nk"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_fp32_backward_and_forward_match_existing_fused(layout, dtype, monkeypatch):
    monkeypatch.setenv("ANIMA_CONVROT_INT8_GEMM", "float")
    monkeypatch.setenv("ANIMA_CONVROT_STE_TF32", "0")
    torch.manual_seed(10)
    w = torch.randint(-127, 128, (16, 8), dtype=torch.int8)
    w = w if layout == "kn" else w.t().contiguous()
    scale = torch.rand(8) / 127
    h = normalized_hadamard(4)
    x1 = torch.randn(2, 3, 16, dtype=dtype, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_()
    y1 = fused_w8a8_forward(
        x1, w, scale, group_size=4, hadamard=h, weight_layout=layout
    )
    y2 = experimental_w8a8_forward(
        x2, w, scale, group_size=4, hadamard=h, weight_layout=layout
    )
    grad = torch.randn_like(y1)
    y1.backward(grad)
    y2.backward(grad)
    assert torch.equal(y1, y2)
    assert torch.equal(x1.grad, x2.grad)
    assert y2.dtype == dtype and x2.grad.dtype == dtype


def test_fp32_residual_compute_cast_stays_outside_function(monkeypatch):
    import library.runtime.convrot.fused as fused
    import library.runtime.convrot.experimental_policy as policy

    monkeypatch.setenv("ANIMA_CONVROT_INT8_GEMM", "float")
    monkeypatch.setattr(fused, "_resolve_compute_dtype", lambda x: torch.bfloat16)
    monkeypatch.setattr(policy, "_resolve_compute_dtype", lambda x: torch.bfloat16)
    torch.manual_seed(83)
    w = torch.randint(-127, 128, (16, 8), dtype=torch.int8)
    scale = torch.rand(8) / 127
    h = normalized_hadamard(4, dtype=torch.bfloat16)
    a = torch.randn(17, 16, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    ya = fused_w8a8_forward(a, w, scale, group_size=4, hadamard=h, weight_layout="kn")
    yb = experimental_w8a8_forward(b, w, scale, group_size=4, hadamard=h)
    gradient = torch.randn_like(ya)
    ya.backward(gradient)
    yb.backward(gradient)
    assert torch.equal(ya, yb)
    assert torch.equal(a.grad, b.grad)


@pytest.mark.parametrize("layout", ["kn", "nk"])
def test_bf16_backward_is_weight_first_rounding(layout):
    torch.manual_seed(11)
    w = torch.randint(-127, 128, (16, 8), dtype=torch.int8)
    w = w if layout == "kn" else w.t().contiguous()
    scale = torch.rand(8) / 127
    h = normalized_hadamard(4)
    x = torch.randn(2, 16, requires_grad=True)
    grad = torch.randn(2, 8)
    experimental_w8a8_forward(
        x,
        w,
        scale,
        group_size=4,
        hadamard=h,
        weight_layout=layout,
        backward_policy="bf16",
    ).backward(grad)
    dense = w.to(torch.bfloat16)
    weight = (
        dense * scale.to(torch.bfloat16)
        if layout == "kn"
        else dense * scale.to(torch.bfloat16)[:, None]
    )
    right = weight.t() if layout == "kn" else weight
    expected = group_rht(grad.to(torch.bfloat16) @ right, 4, hadamard=h).float()
    assert torch.equal(x.grad, expected)


@pytest.mark.parametrize(
    "keyword,value", [("forward_policy", "auto"), ("backward_policy", "fp16")]
)
def test_invalid_policy_rejected_without_mutation(keyword, value):
    net, _ = _network()
    previous = net.unet_loras[0].org_forward
    with pytest.raises(ValueError, match=keyword):
        apply_experimental_policy(net, **{keyword: value})
    assert net.unet_loras[0].org_forward is previous


def test_owner_payload_and_hadamard_are_read_live():
    net, base = _network()
    lora = net.unet_loras[0]
    manifest = apply_experimental_policy(net)
    assert manifest["patched_count"] == 1
    x = torch.randn(2, 16)
    before = lora.org_forward(x)
    assert before.abs().sum() > 0
    old_h = lora._convrot_hadamard
    lora._convrot_hadamard = torch.zeros_like(old_h)
    assert torch.equal(lora.org_forward(x), torch.zeros_like(before))
    lora._convrot_hadamard = old_h
    old_q = base._convrot_quantized_weight.weight
    base._convrot_quantized_weight.weight = torch.zeros_like(old_q)
    assert torch.equal(lora.org_forward(x), torch.zeros_like(before))
    base._convrot_quantized_weight.weight = old_q
    base._convrot_scale.weight = torch.zeros_like(base._convrot_scale.weight)
    assert torch.equal(lora.org_forward(x), torch.zeros_like(before))
    # No base/carrier payload is added to persistent state.
    assert not any("convrot" in key for key in net.state_dict())


def test_policy_does_not_change_other_network_or_lora_parameters():
    active, _ = _network()
    untouched, _ = _network()
    old_forward = untouched.unet_loras[0].org_forward
    apply_experimental_policy(active, backward_policy="bf16")
    assert untouched.unet_loras[0].org_forward is old_forward
    x = torch.randn(2, 16, requires_grad=True)
    active.unet_loras[0](x).square().mean().backward()
    for param in active.parameters():
        assert param.dtype == torch.float32 and param.grad is not None


def test_w8a16_and_unpatched_are_not_rewritten():
    net, _ = _network("w8a16")
    previous = net.unet_loras[0].org_forward
    with pytest.raises(ValueError, match="already-patched W8A8"):
        apply_experimental_policy(net)
    assert net.unet_loras[0].org_forward is previous


def test_invalid_second_candidate_does_not_partially_patch():
    net, _ = _network()
    second, base = _network()
    net.unet_loras.append(second.unet_loras[0])
    old = [lora.org_forward for lora in net.unet_loras]
    del base._modules["_convrot_quantized_weight"]
    with pytest.raises(RuntimeError, match="missing ConvRot block payload"):
        apply_experimental_policy(net)
    assert all(
        lora.org_forward is previous for lora, previous in zip(net.unet_loras, old)
    )


def test_compiled_network_is_rejected():
    net, _ = _network()
    net._anima_blocks_compiled = True
    with pytest.raises(RuntimeError, match="already compiled"):
        apply_experimental_policy(net)


@pytest.mark.parametrize("policy", ["triton_quant", "triton_gemm", "triton_both"])
def test_triton_policy_rejects_cpu_without_fallback(policy):
    net, _ = _network()
    apply_experimental_policy(net, forward_policy=policy)
    with pytest.raises(ValueError, match="CUDA tensor"):
        net.unet_loras[0].org_forward(torch.randn(2, 16))


def test_int8_backward_rejects_cpu_without_fallback():
    net, _ = _network()
    apply_experimental_policy(net, backward_policy="int8")
    x = torch.randn(2, 16, requires_grad=True)
    with pytest.raises(ValueError, match="CUDA tensor"):
        net.unet_loras[0].org_forward(x).sum().backward()
