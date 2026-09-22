from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bench.adaptive_runtime import probe_krea_train


@pytest.mark.parametrize("precision", ["fp16-islands", "fp32-reference"])
def test_mixed_precision_rejects_nf4_before_loading(monkeypatch, precision):
    monkeypatch.setattr(probe_krea_train, "inspect_nf4_checkpoint",
                        lambda path: SimpleNamespace(is_nf4=True))

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid precision must be rejected before loading weights")

    monkeypatch.setattr(probe_krea_train, "load_krea2_dit", forbidden)
    args = SimpleNamespace(precision=precision, weights=Path("nf4.safetensors"))
    with pytest.raises(ValueError, match="not NF4"):
        probe_krea_train.build(args, {})


def test_bf16_probe_rejects_turing_before_loading(monkeypatch):
    monkeypatch.setattr(probe_krea_train.torch.cuda, "get_device_capability", lambda: (7, 5))
    with pytest.raises(ValueError, match="native BF16"):
        probe_krea_train.build(SimpleNamespace(precision="bf16"), {})


@pytest.mark.parametrize("scaled", [False, True])
def test_updates_passes_scaler_to_save_and_restore(tmp_path, monkeypatch, scaled):
    weights = tmp_path / "weights"
    weights.touch()
    network = torch.nn.Linear(1, 1)
    sentinel = object()
    monkeypatch.setattr(probe_krea_train, "make_scaler", lambda scale: sentinel)
    monkeypatch.setattr(probe_krea_train, "load_krea_inputs", lambda *a, **k: (
        {"hidden": torch.ones(1), "mask": torch.ones(1),
         "latents": torch.ones(1), "noise": torch.zeros(1)}, "digest"))
    monkeypatch.setattr(probe_krea_train, "write_result", lambda *a, **k: None)
    monkeypatch.setattr(probe_krea_train, "read_checkpoint", lambda *a, **k: {"step": 0})
    calls = []

    def restore(payload, **kwargs):
        assert kwargs["scaler"] is (sentinel if scaled else None)
        calls.append("restore")
        return [p.detach().clone() for p in network.parameters()], []

    def save(*args, **kwargs):
        assert kwargs["scaler"] is (sentinel if scaled else None)
        assert kwargs["updates"][-1]["peak_reserved"] == 42
        calls.append("save")
        raise RuntimeError("save inspected")

    tensor = torch.tensor
    monkeypatch.setattr(probe_krea_train.torch, "tensor",
                        lambda *a, **k: tensor(*a, **{**k, "device": "cpu"}))
    monkeypatch.setattr(probe_krea_train, "forward_for_loss",
                        lambda model, noisy, text, sigma: network(noisy))
    monkeypatch.setattr(probe_krea_train, "backward_unscaled",
                        lambda loss, *a: loss.backward())
    scaler = SimpleNamespace(step=lambda opt: opt.step(), update=lambda: None)
    sentinel = scaler
    monkeypatch.setattr(probe_krea_train.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(probe_krea_train.torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(probe_krea_train.torch.cuda, "max_memory_reserved", lambda: 42)
    monkeypatch.setattr(probe_krea_train, "restore_checkpoint", restore)
    monkeypatch.setattr(probe_krea_train, "save_checkpoint", save)
    args = SimpleNamespace(inputs="inputs", resolution=256, loss_scale=1024 if scaled else 1,
                           weights=weights, precision="fp16-islands", steps=1,
                           capture_training=False, resume=tmp_path / "resume.pt",
                           output=tmp_path / "result.json", scaled_checkpoint=scaled,
                           swap=24, checkpoint_every_step=True)
    with pytest.raises(RuntimeError, match="save inspected"):
        probe_krea_train.updates(args, None, network, torch.float32,
                                 {"fp32_modules": [], "updates": []}, [])
    assert calls == ["restore", "save"]
