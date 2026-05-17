from __future__ import annotations

from types import SimpleNamespace

import torch

import train
from networks.lora_modules.lokr import LoKrModule


class _FakeAccelerator:
    def unwrap_model(self, model):
        return model


class _FakeNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mask = torch.zeros(1, 1)
        self.was_eval_during_sample = None
        self.mask_during_sample = None

    def clear_timestep_mask(self):
        self.mask.fill_(1.0)


def test_lokr_sampling_uses_inference_mode_and_restores_training(monkeypatch):
    network = _FakeNetwork()
    network.train()
    trainer = train.AnimaTrainer()
    trainer._network = network

    monkeypatch.setattr(
        train.AnimaTrainer,
        "get_models_for_text_encoding",
        lambda self, args, accelerator, text_encoders: text_encoders,
    )
    monkeypatch.setattr(
        train.text_strategies.TextEncodingStrategy,
        "get_strategy",
        staticmethod(lambda: object()),
    )
    monkeypatch.setattr(
        train.text_strategies.TokenizeStrategy,
        "get_strategy",
        staticmethod(lambda: object()),
    )

    def fake_sample_images(*args, **kwargs):
        network.was_eval_during_sample = not network.training
        network.mask_during_sample = network.mask.clone()

    monkeypatch.setattr(train.anima_train_utils, "sample_images", fake_sample_images)

    trainer.sample_images(
        _FakeAccelerator(),
        SimpleNamespace(),
        epoch=None,
        global_step=1,
        device=torch.device("cpu"),
        vae=None,
        tokenizer=None,
        text_encoder=object(),
        unet=object(),
    )

    assert network.was_eval_during_sample is True
    torch.testing.assert_close(network.mask_during_sample, torch.ones(1, 1))
    assert network.training is True


def test_lokr_eval_forward_ignores_stale_timestep_mask():
    base = torch.nn.Linear(4, 4, bias=False)
    lokr = LoKrModule(
        "lora_unet_test",
        base,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
        factor=2,
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
