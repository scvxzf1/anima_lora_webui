"""Numerics and lifecycle regressions, independent of large model assets."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from library.models.krea2_raw.quantize import quantize_dit_to_nf4
from networks.lora_anima.application import set_enabled
from networks.lora_modules.dora import DoRALoRAModule
from networks.lora_modules.ortho import OrthoHydraLoRAModule, OrthoLoRAModule
from networks.lora_modules.weight_access import (
    is_nf4_weight,
    materialize_module_weight,
)


def _base(nf4=True, bias=True):
    torch.manual_seed(37)
    model = torch.nn.Sequential(torch.nn.Linear(16, 8, bias=bias))
    if nf4:
        model.bfloat16()
        quantize_dit_to_nf4(model, torch.device("cpu"))
    return model[0]


def _adapter(cls=DoRALoRAModule, *, nf4=True, channel_scale=None):
    base = _base(nf4)
    adapter = cls("adapter", base, lora_dim=4, alpha=2, channel_scale=channel_scale)
    adapter.apply_to()
    if nf4:
        adapter.bfloat16()
    return base, adapter


def _activate(adapter):
    with torch.no_grad():
        if isinstance(adapter, DoRALoRAModule):
            adapter.lora_up.weight.normal_(std=0.1)
            adapter.magnitude.mul_(1.2)
        else:
            adapter.lambda_layer.fill_(0.2)
            adapter.S_p.normal_(std=0.05)
            adapter.S_q.normal_(std=0.05)


def test_materialized_dense_weight_does_not_alias_base():
    base = _base(nf4=False)
    original = base.weight.detach().clone()
    materialize_module_weight(base).zero_()
    torch.testing.assert_close(base.weight, original, rtol=0, atol=0)


def test_materialization_rejects_incomplete_packed_state():
    base = _base()
    base.weight.quant_state = None
    with pytest.raises(ValueError, match="missing quant_state"):
        materialize_module_weight(base)


def test_materialization_recognizes_params4bit_subclasses_and_restored_state():
    base = _base()
    expected = materialize_module_weight(base)
    subclass = type("Restored4bit", (type(base.weight),), {})
    base.weight.__class__ = subclass
    base.weight.bnb_quantized = False
    assert is_nf4_weight(base.weight)
    torch.testing.assert_close(materialize_module_weight(base), expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_dora_cache_keeps_values_not_just_fp32_dtype(dtype):
    base = _base()
    adapter = DoRALoRAModule("adapter", base, lora_dim=4)
    original = adapter._base_weight_norm_sq.clone()
    adapter.apply_to()
    parent = torch.nn.Sequential(adapter)
    parent.to(dtype=dtype)
    assert adapter._base_weight_norm_sq.dtype == torch.float32
    torch.testing.assert_close(adapter._base_weight_norm_sq, original, rtol=0, atol=0)
    assert "_base_weight_norm_sq" not in adapter.state_dict()


def test_nf4_dora_zero_init_is_exact_base_after_bf16_cast():
    base, adapter = _adapter()
    x = torch.randn(3, 16, dtype=torch.bfloat16)
    torch.testing.assert_close(base(x), adapter.org_forward(x), rtol=0, atol=0)


@pytest.mark.parametrize("nf4", [False, True])
@pytest.mark.parametrize("strength", [0.0, 0.5, -0.5, 1.5])
def test_dora_strength_scales_the_entire_adapter(nf4, strength):
    base, adapter = _adapter(nf4=nf4)
    _activate(adapter)
    dtype = torch.bfloat16 if nf4 else torch.float32
    x = torch.randn(2, 5, 16, dtype=dtype)
    original = adapter.org_forward(x)
    full = base(x)
    adapter.multiplier = strength
    expected = original + strength * (full - original)
    torch.testing.assert_close(base(x), expected, rtol=0, atol=0)
    if strength == 0:
        torch.testing.assert_close(
            adapter.get_weight(), torch.zeros(8, 16), rtol=0, atol=0
        )


@pytest.mark.parametrize("strength", [0.0, 0.5, 1.0, -0.5])
def test_dense_dora_merge_fuse_and_weight_access_agree(strength):
    base, adapter = _adapter(nf4=False)
    _activate(adapter)
    adapter.eval()
    adapter.multiplier = strength
    x = torch.randn(2, 16)
    expected = base(x)
    dense = base.weight.detach() + adapter.get_weight()
    torch.testing.assert_close(F.linear(x, dense, base.bias), expected)
    merge_base = _base(nf4=False)
    merger = DoRALoRAModule(
        "merge", merge_base, lora_dim=4, alpha=2, multiplier=strength
    )
    merger.merge_to(adapter.state_dict(), torch.float32, "cpu")
    torch.testing.assert_close(merge_base(x), expected)
    adapter.fuse_weight()
    torch.testing.assert_close(base(x), expected)
    adapter.unfuse_weight()
    torch.testing.assert_close(base(x), expected)


@pytest.mark.parametrize("with_channel_scale", [False, True])
def test_nf4_norm_is_bias_free_and_autocast_safe(with_channel_scale):
    scale = torch.linspace(0.1, 3, 16) if with_channel_scale else None
    base, adapter = _adapter(channel_scale=scale)
    _activate(adapter)
    with torch.no_grad():
        base.bias.fill_(4096)
    down = adapter.lora_down.weight.detach().float()
    if with_channel_scale:
        down = down * adapter.inv_scale.float()
    up = adapter.lora_up.weight.detach().float()
    merged = materialize_module_weight(base) + adapter.scale * (up @ down)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = adapter._nf4_merged_norm(torch.zeros(1, 8, dtype=torch.bfloat16))
    assert actual.dtype == torch.float32
    assert not actual.requires_grad
    torch.testing.assert_close(actual, merged.norm(dim=1), atol=2e-3, rtol=5e-3)


def test_nf4_dora_input_and_parameter_gradients_match_detached_dense_reference():
    base, adapter = _adapter()
    _activate(adapter)
    x = torch.randn(3, 16, dtype=torch.bfloat16, requires_grad=True)
    actual = base(x)
    params = [x, adapter.lora_down.weight, adapter.lora_up.weight, adapter.magnitude]
    grads = torch.autograd.grad(actual.float().square().mean(), params)
    x_ref, down, up, magnitude = [p.detach().float().requires_grad_() for p in params]
    weight = materialize_module_weight(base)
    merged = weight + adapter.scale * (up @ down)
    normalized = merged * (magnitude / merged.detach().norm(dim=1))[:, None]
    reference = F.linear(x_ref, normalized, base.bias.float())
    expected_grads = torch.autograd.grad(
        reference.square().mean(), [x_ref, down, up, magnitude]
    )
    torch.testing.assert_close(actual.float(), reference, rtol=2e-2, atol=1e-2)
    for actual_grad, expected_grad in zip(grads, expected_grads):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(
            actual_grad.float(), expected_grad, rtol=4e-2, atol=2e-3
        )


def test_nf4_dora_merge_and_fuse_are_explicitly_rejected():
    base = _base()
    adapter = DoRALoRAModule("adapter", base, lora_dim=4)
    packed = base.weight.data.clone()
    with pytest.raises(RuntimeError, match="not supported for NF4"):
        adapter.merge_to(adapter.state_dict(), torch.bfloat16, "cpu")
    adapter.apply_to()
    with pytest.raises(RuntimeError, match="not supported for NF4"):
        adapter.fuse_weight()
    torch.testing.assert_close(base.weight.data, packed, rtol=0, atol=0)


@pytest.mark.parametrize("cls", [OrthoLoRAModule, OrthoHydraLoRAModule])
@pytest.mark.parametrize("rank", [0, 9])
def test_ortho_rank_rejected_before_svd(cls, rank):
    with pytest.raises(ValueError, match="rank must be"):
        cls("adapter", _base(), lora_dim=rank)


@pytest.mark.parametrize("cls", [OrthoLoRAModule, OrthoHydraLoRAModule])
def test_ortho_cpu_initialization_does_not_select_default_gpu(cls, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    original = torch.svd_lowrank
    devices = []

    def checked_svd(weight, **kwargs):
        devices.append(weight.device.type)
        return original(weight, **kwargs)

    monkeypatch.setattr(torch, "svd_lowrank", checked_svd)
    _adapter(cls)
    assert devices == ["cpu"]


@pytest.mark.parametrize("cls", [DoRALoRAModule, OrthoLoRAModule, OrthoHydraLoRAModule])
def test_nf4_adapters_disable_and_reenable_without_changing_base(cls):
    base, adapter = _adapter(cls)
    _activate(adapter)
    packed = base.weight.data.clone()
    x = torch.randn(2, 5, 16, dtype=torch.bfloat16)
    network = SimpleNamespace(
        text_encoder_loras=[],
        unet_loras=[adapter],
        text_encoder_refts=[],
        unet_refts=[],
    )
    enabled_output = base(x)
    set_enabled(network, False)
    torch.testing.assert_close(base(x), adapter.org_forward(x), rtol=0, atol=0)
    set_enabled(network, True)
    torch.testing.assert_close(base(x), enabled_output, rtol=0, atol=0)
    torch.testing.assert_close(base.weight.data, packed, rtol=0, atol=0)


@pytest.mark.parametrize("cls", [DoRALoRAModule, OrthoLoRAModule, OrthoHydraLoRAModule])
def test_nf4_adapter_native_state_and_optimizer_resume(cls):
    base, adapter = _adapter(cls)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
    x = torch.randn(2, 5, 16, dtype=torch.bfloat16)

    def step(model, opt):
        opt.zero_grad(set_to_none=True)
        loss = model(x).float().square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        return loss.detach()

    for _ in range(3):
        step(base, optimizer)
    state = copy.deepcopy(adapter.state_dict())
    restored_base, restored = _adapter(cls)
    restored.load_state_dict(state, strict=True)
    resumed_opt = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    resumed_opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    torch.testing.assert_close(base(x), restored_base(x), rtol=0, atol=0)
    torch.testing.assert_close(
        step(base, optimizer), step(restored_base, resumed_opt), rtol=0, atol=0
    )
    for key, value in adapter.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
