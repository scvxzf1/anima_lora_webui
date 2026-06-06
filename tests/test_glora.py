from __future__ import annotations

import torch
import torch.nn.functional as F

from networks.plugins.glora.module import GLoRAModule


def test_glora_zero_init_preserves_base_forward():
    linear = torch.nn.Linear(5, 7, bias=False)
    x = torch.randn(3, 5)
    expected = linear(x)

    glora = GLoRAModule("lora_unet_blocks_0_q_proj", linear, lora_dim=2, alpha=2)
    glora.apply_to()

    assert torch.allclose(linear(x), expected)


def test_glora_forward_matches_dense_delta_in_eval():
    linear = torch.nn.Linear(5, 7, bias=False)
    base_weight = linear.weight.detach().clone()
    x = torch.randn(4, 5)

    glora = GLoRAModule(
        "lora_unet_blocks_0_q_proj",
        linear,
        multiplier=0.75,
        lora_dim=3,
        alpha=1.5,
    )
    with torch.no_grad():
        glora.a1.weight.normal_(std=0.2)
        glora.a2.weight.normal_(std=0.2)
        glora.b1.weight.normal_(std=0.2)
        glora.b2.weight.normal_(std=0.2)
    glora.eval()
    glora.apply_to()

    scale = 0.75 * (1.5 / 3)
    dense_delta = ((base_weight @ glora.a1.weight) @ glora.a2.weight) + (
        glora.b1.weight @ glora.b2.weight
    )
    expected = F.linear(x, base_weight + dense_delta * scale)

    assert torch.allclose(linear(x), expected, atol=1e-6)


def test_glora_fuse_unfuse_preserves_original_weight():
    linear = torch.nn.Linear(5, 7, bias=False)
    original = linear.weight.detach().clone()
    glora = GLoRAModule("lora_unet_blocks_0_q_proj", linear, lora_dim=3, alpha=3)
    with torch.no_grad():
        glora.a1.weight.normal_(std=0.2)
        glora.a2.weight.normal_(std=0.2)
        glora.b1.weight.normal_(std=0.2)
        glora.b2.weight.normal_(std=0.2)

    glora.fuse_weight()
    assert not torch.allclose(linear.weight, original)

    glora.unfuse_weight()
    assert torch.allclose(linear.weight, original)
