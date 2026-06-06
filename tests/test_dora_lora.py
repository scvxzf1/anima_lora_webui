from __future__ import annotations

import copy

import torch

from networks.lora_modules import DoRALoRAModule


def test_dora_initial_forward_matches_base_linear():
    torch.manual_seed(11)
    base = torch.nn.Linear(4, 3, bias=True)
    reference = copy.deepcopy(base)
    dora = DoRALoRAModule(
        "lora_unet_blocks_0_mlp_layer1",
        base,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
    )
    dora.apply_to()
    base.eval()
    reference.eval()

    x = torch.randn(2, 5, 4)

    with torch.no_grad():
        torch.testing.assert_close(base(x), reference(x), rtol=1e-6, atol=1e-6)


def test_dora_fuse_unfuse_restores_original_weight():
    torch.manual_seed(23)
    base = torch.nn.Linear(4, 3, bias=False)
    dora = DoRALoRAModule(
        "lora_unet_blocks_0_mlp_layer1",
        base,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
    )
    torch.nn.init.normal_(dora.lora_up.weight, std=0.05)
    original = base.weight.detach().clone()

    dora.fuse_weight()

    assert not torch.allclose(base.weight, original)

    dora.unfuse_weight()

    torch.testing.assert_close(base.weight, original, rtol=1e-6, atol=1e-6)
