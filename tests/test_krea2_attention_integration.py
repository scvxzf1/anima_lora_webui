from __future__ import annotations

from types import SimpleNamespace

import torch

from library.models.krea2_raw import attention_backend, inference_runner, weights
from library.training import model_loading


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.attn_mode = None

    def set_attention_mode(self, mode: str) -> None:
        self.attn_mode = mode


def test_training_validates_attention_before_loading_dit(monkeypatch) -> None:
    order = []
    model = _FakeModel()

    def fake_validate(mode, *, dtype, compile_enabled):
        order.append(("validate", mode, dtype, compile_enabled))
        return "flash"

    def fake_load(*args, **kwargs):
        order.append(("load", args, kwargs))
        return model

    def fake_prepare(target, mode, *, dtype, compile_enabled):
        order.append(("prepare", mode, dtype, compile_enabled))
        target.set_attention_mode(mode)
        return mode

    monkeypatch.setattr(attention_backend, "validate_krea2_attention_mode", fake_validate)
    monkeypatch.setattr(attention_backend, "prepare_krea2_attention", fake_prepare)
    monkeypatch.setattr(weights, "load_krea2_dit", fake_load)
    monkeypatch.setattr(model_loading, "_maybe_probe_components", lambda *args, **kwargs: None)

    trainer = SimpleNamespace(is_swapping_blocks=False, peak_probe=None)
    args = SimpleNamespace(
        attn_mode="flash",
        torch_compile=True,
        base_compute="bf16",
        nf4_prequantized_path=None,
        pretrained_model_name_or_path="dit.safetensors",
        unsloth_offload_checkpointing=False,
        selective_checkpoint="off",
        blocks_to_swap=0,
        vr_loss_weight=0.0,
    )

    loaded, encoders = model_loading._load_krea2_dit(
        trainer,
        args,
        torch.bfloat16,
        SimpleNamespace(device=torch.device("cpu")),
        [object()],
    )

    assert order[0] == ("validate", "flash", torch.bfloat16, True)
    assert order[1][0] == "load"
    assert loaded is model
    assert model.attn_mode == "flash"
    assert len(encoders) == 1


def test_inference_validates_attention_before_loading_dit(monkeypatch) -> None:
    order = []
    model = _FakeModel()

    def fake_validate(mode, *, dtype, compile_enabled):
        order.append(("validate", mode, dtype, compile_enabled))
        return "flash"

    def fake_load(*args, **kwargs):
        order.append(("load", args, kwargs))
        return model

    def fake_prepare(target, mode, *, dtype, compile_enabled):
        order.append(("prepare", mode, dtype, compile_enabled))
        target.set_attention_mode(mode)
        return mode

    monkeypatch.setattr(inference_runner, "validate_krea2_attention_mode", fake_validate)
    monkeypatch.setattr(inference_runner, "prepare_krea2_attention", fake_prepare)
    monkeypatch.setattr(inference_runner, "load_krea2_dit", fake_load)
    args = SimpleNamespace(
        dit="dit.safetensors",
        attn_mode="flash",
        compile=False,
        compile_blocks=True,
        lora_weight=None,
    )

    loaded, network = inference_runner.load_krea2_dit_for_inference(
        args,
        torch.device("cpu"),
        torch.bfloat16,
    )

    assert order[0] == ("validate", "flash", torch.bfloat16, True)
    assert order[1][0] == "load"
    assert loaded is model
    assert model.attn_mode == "flash"
    assert network is None
