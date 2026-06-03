from __future__ import annotations

import torch
import torch.nn.functional as F

from networks.plugins.loha.module import LoHaModule


def test_loha_initial_forward_is_zero_delta():
    base = torch.nn.Linear(4, 3, bias=False)
    loha = LoHaModule(
        "lora_unet_test",
        base,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
    )
    loha.apply_to()

    x = torch.randn(2, 5, 4)
    base_out = F.linear(x, loha.org_module_ref[0].weight)

    torch.testing.assert_close(loha.org_module_ref[0](x), base_out)


def test_loha_forward_matches_hadamard_weight():
    base = torch.nn.Linear(4, 3, bias=False)
    loha = LoHaModule(
        "lora_unet_test",
        base,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
    )
    loha.apply_to()

    with torch.no_grad():
        loha.org_module_ref[0].weight.zero_()
        loha.hada_w1_a.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        loha.hada_w1_b.copy_(torch.tensor([[0.5, 1.0, 1.5, 2.0], [2.5, 3.0, 3.5, 4.0]]))
        loha.hada_w2_a.copy_(torch.tensor([[1.5, 0.5], [0.25, 1.25], [2.0, 0.75]]))
        loha.hada_w2_b.copy_(torch.tensor([[1.0, 0.5, 0.25, 0.75], [1.5, 2.0, 2.5, 3.0]]))

    x = torch.randn(2, 5, 4)
    weight = (loha.hada_w1_a @ loha.hada_w1_b) * (loha.hada_w2_a @ loha.hada_w2_b)

    torch.testing.assert_close(loha.org_module_ref[0](x), F.linear(x, weight))


def test_loha_eval_forward_ignores_stale_timestep_mask():
    base = torch.nn.Linear(4, 3, bias=False)
    loha = LoHaModule(
        "lora_unet_test",
        base,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
    )
    loha.apply_to()

    with torch.no_grad():
        loha.org_module_ref[0].weight.zero_()
        loha.hada_w1_a.fill_(1.0)
        loha.hada_w1_b.fill_(1.0)
        loha.hada_w2_a.fill_(1.0)
        loha.hada_w2_b.fill_(1.0)
        loha._timestep_mask.zero_()

    x = torch.ones(1, 4)

    loha.train()
    train_out = loha.org_module_ref[0](x)

    loha.eval()
    eval_out = loha.org_module_ref[0](x)

    torch.testing.assert_close(train_out, torch.zeros_like(train_out))
    assert torch.count_nonzero(eval_out).item() == eval_out.numel()
