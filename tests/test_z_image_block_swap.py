from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from library.models.z_image import block_swap as block_swap_module
from library.training import model_loading


class RecordingOffloader:
    def __init__(self, layers, blocks_to_swap, device, **_kwargs):
        self.layers = layers
        self.blocks_to_swap = blocks_to_swap
        self.device = torch.device(device)
        self.cuda_available = False
        self.forward_only = False
        self.futures = {}
        self.events = []

    def wait_for_block(self, index):
        self.events.append(("wait", index))

    def submit_move_blocks(self, _layers, index):
        self.events.append(("submit", index))

    def prepare_block_devices_before_forward(self, _layers, free_cache=True):
        self.events.append(("prepare", free_cache))

    def set_forward_only(self, value):
        self.forward_only = value

    def restore_blocks_to_device(self, _layers, device):
        self.events.append(("restore", torch.device(device).type))

    def flush_profile_events(self, blocking=False):
        self.events.append(("flush", blocking))


def _tiny_model(n_layers: int = 4):
    from diffusers import ZImageTransformer2DModel

    model = ZImageTransformer2DModel(
        dim=128,
        n_layers=n_layers,
        n_refiner_layers=1,
        n_heads=1,
        n_kv_heads=1,
        cap_feat_dim=32,
        axes_dims=[32, 48, 48],
        axes_lens=[64, 64, 64],
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.ndim >= 2:
                parameter.normal_(0, 0.01)
            elif name.endswith("weight"):
                parameter.fill_(1)
            else:
                parameter.zero_()
    return model


def _inputs():
    latents = [torch.randn(16, 1, 4, 4, requires_grad=True)]
    prompts = [torch.randn(3, 32)]
    return latents, prompts


def test_checkpointed_block_swap_dispatches_once_per_main_layer(monkeypatch) -> None:
    monkeypatch.setattr(block_swap_module, "ModelOffloader", RecordingOffloader)
    model = _tiny_model()
    model.enable_gradient_checkpointing()
    adapter = block_swap_module.enable_z_image_block_swap(model, 2, torch.device("cpu"))
    # Training bootstrap re-enables checkpointing after applying LoRA modules.
    model.enable_gradient_checkpointing()
    latents, prompts = _inputs()

    model(x=latents, t=torch.tensor([0.5]), cap_feats=prompts).sample[
        0
    ].mean().backward()

    forward_events = [
        event for event in adapter.offloader.events if event[0] != "prepare"
    ]
    assert forward_events == [
        event for index in range(4) for event in (("wait", index), ("submit", index))
    ]
    assert latents[0].grad is not None


def test_no_grad_preview_uses_layer_forward_dispatch(monkeypatch) -> None:
    monkeypatch.setattr(block_swap_module, "ModelOffloader", RecordingOffloader)
    model = _tiny_model()
    model.enable_gradient_checkpointing()
    adapter = block_swap_module.enable_z_image_block_swap(model, 2, torch.device("cpu"))
    latents, prompts = _inputs()

    model.switch_block_swap_for_inference()
    adapter.offloader.events.clear()
    with torch.no_grad():
        model(x=latents, t=torch.tensor([0.5]), cap_feats=prompts)

    assert adapter.offloader.forward_only is True
    assert adapter.offloader.events == [
        event for index in range(4) for event in (("wait", index), ("submit", index))
    ]


def test_block_swap_protocol_places_non_layer_modules_and_pauses(monkeypatch) -> None:
    monkeypatch.setattr(block_swap_module, "ModelOffloader", RecordingOffloader)
    model = _tiny_model()
    state_keys = set(model.state_dict())
    adapter = block_swap_module.enable_z_image_block_swap(model, 2, torch.device("cpu"))

    assert set(model.state_dict()) == state_keys
    model.move_to_device_except_swap_blocks(torch.device("cpu"))
    assert model.layers is adapter.layers
    model.prepare_block_swap_before_forward(free_cache=False)
    assert ("prepare", False) in adapter.offloader.events
    assert model.pause_block_swap() is True
    assert model.blocks_to_swap == 0
    assert model.resume_block_swap() is True
    assert model.blocks_to_swap == 2


@pytest.mark.parametrize("blocks_to_swap", [0, 3])
def test_block_swap_rejects_invalid_range(monkeypatch, blocks_to_swap) -> None:
    monkeypatch.setattr(block_swap_module, "ModelOffloader", RecordingOffloader)
    with pytest.raises(ValueError, match="between 1 and 2"):
        block_swap_module.enable_z_image_block_swap(
            _tiny_model(), blocks_to_swap, torch.device("cpu")
        )


def test_z_image_loader_stages_on_cpu_and_enables_block_swap(monkeypatch) -> None:
    captured = {}

    class FakeModel(torch.nn.Module):
        def enable_gradient_checkpointing(self):
            captured["gradient_checkpointing"] = True

    def fake_load(path, *, dtype, device):
        captured["load"] = (path, dtype, torch.device(device))
        return FakeModel()

    def fake_enable(model, blocks_to_swap, device, **kwargs):
        captured["enable"] = (model, blocks_to_swap, torch.device(device), kwargs)

    monkeypatch.setattr(
        "library.models.z_image.weights.load_z_image_transformer", fake_load
    )
    monkeypatch.setattr(block_swap_module, "enable_z_image_block_swap", fake_enable)
    monkeypatch.setattr(
        model_loading, "resolve_block_swap_profile_jsonl", lambda _args: "profile.jsonl"
    )
    monkeypatch.setattr(
        model_loading, "_maybe_probe_components", lambda *_args, **_kwargs: None
    )
    args = SimpleNamespace(
        pretrained_model_name_or_path="z-image",
        gradient_checkpointing=True,
        blocks_to_swap=20,
        block_swap_transfer_dtype="bf16",
        block_swap_restore_mode="slab",
    )
    trainer = SimpleNamespace(is_swapping_blocks=False)
    accelerator = SimpleNamespace(device=torch.device("cpu"))

    model, text_encoders = model_loading._load_z_image_dit(
        trainer,
        args,
        torch.bfloat16,
        accelerator,
        [None],
    )

    assert model is captured["enable"][0]
    assert text_encoders == [None]
    assert trainer.is_swapping_blocks is True
    assert captured["gradient_checkpointing"] is True
    assert captured["load"] == ("z-image", torch.bfloat16, torch.device("cpu"))
    assert captured["enable"][1:] == (
        20,
        torch.device("cpu"),
        {
            "profile_jsonl": "profile.jsonl",
            "transfer_dtype": "bf16",
            "restore_mode": "slab",
        },
    )
