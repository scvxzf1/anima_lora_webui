"""CPU numerical and policy boundaries for explicit native Krea-2 AMP."""
from types import MethodType, SimpleNamespace as NS

import pytest
import torch

from library.models.krea2_raw.dit import SingleMMDiTConfig, SingleStreamDiT
from library.models.krea2_raw.native_precision import enable_native_guards, observe_guards
from library.training.adaptive_runtime.native_manifest import base_metadata
from library.training.adaptive_runtime.native_precision import configuration_errors
from library.training.adaptive_runtime.native_family import execution_guards, report_guards
from networks.lora_modules.lora import LoRAModule


def tiny():
    return SingleStreamDiT(SingleMMDiTConfig(
        features=16, tdim=8, txtdim=16, heads=1, multiplier=1, layers=2,
        patch=2, channels=2, txtlayers=2, txtheads=1, txtkvheads=1)).half()


@pytest.mark.parametrize("checkpoint", [False, True])
def test_native_forward_backward_preserves_adapter_and_compute_boundary(checkpoint):
    torch.manual_seed(9)
    model = tiny()
    enable_native_guards(model)
    if checkpoint:
        model.enable_gradient_checkpointing()
    adapter = LoRAModule("final", model.last.linear, lora_dim=2)
    adapter.apply_to()
    adapter.lora_up.weight.data.fill_(0.01)
    seen = []
    final_dtypes = []
    model.last.linear.register_forward_hook(lambda _m, _i, out: final_dtypes.append(out.dtype))
    model.blocks[0].attn.wq.register_forward_pre_hook(lambda _m, inp: seen.append(inp[0].dtype))
    x = torch.randn(1, 1, 8, dtype=torch.float16, requires_grad=True)
    context = torch.randn(1, 1, 2, 16, dtype=torch.float16)
    pos = torch.zeros(1, 2, 3)
    with torch.autocast("cpu", dtype=torch.float16):
        output = model(x, context, torch.tensor([0.4]), pos, torch.ones(1, 2, dtype=torch.bool))
    assert final_dtypes == [torch.float32]
    assert output.dtype == torch.float16 and torch.isfinite(output).all()
    output.float().square().mean().backward()
    assert seen and all(dtype == torch.float16 for dtype in seen)
    assert adapter.lora_up.weight.grad is not None
    assert adapter.lora_up.weight.grad.abs().sum() > 0
    assert adapter.lora_down.weight.grad is not None
    assert torch.isfinite(adapter.lora_down.weight.grad).all()
    assert all(p.dtype == torch.float32 for p in adapter.parameters())
    assert all(p.dtype == torch.float16 and not p.requires_grad for p in model.parameters())
    assert observe_guards(model)["final_linear_fp32"]
    base_metadata(model, family="krea2_raw")
    assert seen.count(torch.float16) >= (2 if checkpoint else 1)


def test_bf16_existing_path_is_unaffected():
    model = tiny().bfloat16()
    block = model.blocks[0]
    x = torch.randn(1, 2, 16, dtype=torch.bfloat16)
    vec = torch.randn(1, 1, 96, dtype=torch.bfloat16)
    freqs = model.posemb(torch.zeros(1, 2, 3))
    output = block(x, vec, freqs)
    assert output.dtype == torch.bfloat16 and torch.isfinite(output).all()
    assert not observe_guards(model)["fp32_residual"]


def test_policy_refuses_unsupported_combinations_and_auto():
    config = dict(adaptive_precision="native_fp16", model_family="krea2_raw")
    assert not configuration_errors(config)
    for change in ({"mixed_precision": "bf16"}, {"base_compute": "nf4"},
                   {"torch_compile": True}, {"attn_mode": "flash"},
                   {"adaptive_oom_retry": True}, {"adaptive_precision": "auto",
                    "adaptive_resolved_mode": "native_fp16"}):
        assert configuration_errors({**config, **change})


def test_policy_detects_guard_and_attention_drift_without_anima_claims(monkeypatch):
    model = tiny()
    enable_native_guards(model)
    # Structural production identity is enforced independently of numerical tiny tests.
    with pytest.raises(ValueError, match="official structure"):
        execution_guards(model, "krea2_raw")
    monkeypatch.setattr(SingleMMDiTConfig, "krea2_raw", classmethod(lambda cls: model.config))
    model.enable_gradient_checkpointing()
    assert execution_guards(model, "krea2_raw")["text_fp32_residual"]
    model.txtfusion.refiner_blocks[0].fp32_residual = False
    with pytest.raises(ValueError, match="guards missing"):
        execution_guards(model, "krea2_raw")
    model.txtfusion.refiner_blocks[0].fp32_residual = True
    model.blocks[0].attn.attn_mode = "flash"
    with pytest.raises(ValueError, match="attention drifted"):
        execution_guards(model, "krea2_raw")
    assert "Anima" not in report_guards(model, "krea2_raw")["evidence"]
    assert report_guards(NS(), "z_image")["family"] == "z_image"


def test_nonfinite_weights_rejected_at_install_only():
    model = tiny()
    model.first.weight.data[0, 0] = float("inf")
    with pytest.raises(ValueError, match="nonfinite"):
        enable_native_guards(model)


@pytest.mark.parametrize("drift", ["base_forward", "adapter_forward", "adapter_original"])
def test_final_guard_checks_actual_forward_and_adapter_original(monkeypatch, drift):
    model = tiny()
    enable_native_guards(model)
    model.enable_gradient_checkpointing()
    monkeypatch.setattr(SingleMMDiTConfig, "krea2_raw", classmethod(lambda cls: model.config))
    linear = model.last.linear
    assert execution_guards(model, "krea2_raw")["final_linear_fp32"]
    if drift != "base_forward":
        adapter = LoRAModule("final", linear, lora_dim=2)
        adapter.apply_to()
        assert execution_guards(model, "krea2_raw")["final_linear_fp32"]
    original = MethodType(torch.nn.Linear.forward, linear)
    if drift == "adapter_original":
        adapter.org_forward = original
    else:
        linear.forward = original
    assert linear._native_fp32_linear is True
    assert observe_guards(model)["final_linear_fp32"] is False
    with pytest.raises(ValueError, match="guards missing"):
        execution_guards(model, "krea2_raw")
