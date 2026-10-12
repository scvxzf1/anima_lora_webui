"""CPU contracts for the two Qwen Image 2.1 LoRA-family plugins."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from diffusers import QwenImage21Transformer2DModel
from safetensors import safe_open

from library.models.qwen_image_2_1.family import forward_for_loss
from library.models.qwen_image_2_1.lora_targets import qwen_image_2_1_target_kwargs
from networks.lora_anima import create_network, create_network_from_weights


@pytest.mark.parametrize("flag", ["dora_wd", "use_vera", "use_glora"])
def test_qwen_factory_rejects_other_network_specs(flag) -> None:
    model = QwenImage21Transformer2DModel(
        num_layers=1, num_attention_heads=1, attention_head_dim=128,
        context_in_dim=4096, mlp_ratio=1,
    )
    with pytest.raises(ValueError, match="does not support this network spec"):
        create_network(
            multiplier=1.0, network_dim=4, network_alpha=4,
            vae=None, text_encoders=[], unet=model,
            model_family="qwen_image_2_1", **{flag: "true"},
            **qwen_image_2_1_target_kwargs(),
        )


@pytest.mark.parametrize("variant", ["lokr", "loha"])
@pytest.mark.parametrize("task", ["t2i", "edit"])
def test_qwen_plugin_train_and_checkpoint_roundtrip(tmp_path, variant, task) -> None:
    torch.manual_seed(31)
    model = QwenImage21Transformer2DModel(
        num_layers=1, num_attention_heads=1, attention_head_dim=128,
        context_in_dim=4096, mlp_ratio=1,
    )
    model.requires_grad_(False)
    base = deepcopy(model)
    kwargs = qwen_image_2_1_target_kwargs()
    network = create_network(
        multiplier=1.0, network_dim=4, network_alpha=4,
        vae=None, text_encoders=[], unet=model,
        model_family="qwen_image_2_1", **{f"use_{variant}": "true"}, **kwargs,
    )
    assert network._network_spec.name == variant
    assert len(network.unet_loras) == 4
    assert {type(module.org_module_ref[0]) for module in network.unet_loras} == {torch.nn.Linear}
    network.apply_to([], model, False, True)
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.normal_(0, 0.25)

    latents = torch.randn(1, 64, 2, 2, requires_grad=True)
    hiddens = torch.randn(1, 5, 4096)
    mask = torch.ones(1, 5, dtype=torch.bool)
    extra = {}
    if task == "edit":
        extra = {
            "reference_latents": torch.randn(1, 64, 2, 2),
            "image_slot_mask": torch.tensor([[True, False, False, False, False]]),
        }
    sigma = torch.tensor([0.5])
    baseline_latents = latents.detach().requires_grad_(True)
    baseline = forward_for_loss(base, baseline_latents, hiddens, mask, sigma, **extra)
    output = forward_for_loss(model, latents, hiddens, mask, sigma, **extra)
    assert output.shape == latents.shape
    assert not torch.allclose(output, baseline, atol=1e-6, rtol=1e-6)
    output.square().mean().backward()
    baseline.square().mean().backward()
    assert latents.grad is not None and torch.count_nonzero(latents.grad)
    assert baseline_latents.grad is not None
    assert not torch.allclose(latents.grad, baseline_latents.grad, atol=1e-8, rtol=1e-5)
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in network.parameters())

    checkpoint = tmp_path / f"qwen-{task}-{variant}.safetensors"
    network.save_weights(str(checkpoint), torch.float32, metadata={})
    with safe_open(str(checkpoint), framework="pt") as saved:
        assert saved.metadata()["ss_model_family"] == "qwen_image_2_1"
        assert len(saved.keys()) > 0

    reloaded_model = deepcopy(base)
    restored, _ = create_network_from_weights(
        multiplier=1.0, file=str(checkpoint), ae=None,
        text_encoders=[], unet=reloaded_model,
    )
    assert restored._network_spec.name == variant
    restored.apply_to([], reloaded_model, False, True)
    restored.load_weights(str(checkpoint))
    roundtrip = forward_for_loss(reloaded_model, latents.detach(), hiddens, mask, sigma, **extra)
    torch.testing.assert_close(roundtrip, output.detach(), atol=1e-5, rtol=1e-5)
