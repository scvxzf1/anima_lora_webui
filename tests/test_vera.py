from __future__ import annotations

import torch
import torch.nn.functional as F

from networks.plugins.vera.module import VeRAModule, make_projection_bank


def test_vera_forward_matches_formula_and_zero_init_is_identity():
    torch.manual_seed(11)
    base = torch.nn.Linear(4, 6, bias=False)
    bank = make_projection_bank(
        rank=3,
        max_in_features=4,
        max_out_features=6,
        projection_prng_key=123,
    )
    vera = VeRAModule(
        "lora_unet_test",
        base,
        multiplier=1.0,
        lora_dim=3,
        alpha=3,
        projection_bank=bank,
        d_initial=0.1,
    )
    vera.apply_to()

    x = torch.randn(2, 4)
    with torch.no_grad():
        base_out = F.linear(x, vera.org_module_ref[0].weight)
    zero_out = vera.org_module_ref[0](x)
    torch.testing.assert_close(zero_out, base_out)

    with torch.no_grad():
        vera.vera_lambda_b.copy_(torch.randn_like(vera.vera_lambda_b))
        vera.vera_lambda_d.copy_(torch.randn_like(vera.vera_lambda_d))

    y = vera.org_module_ref[0](x)
    vera_A, vera_B = bank.slice(4, 6)
    expected_delta = F.linear(
        vera.vera_lambda_d * F.linear(x, vera_A),
        vera_B,
    ) * vera.vera_lambda_b
    expected = base_out + expected_delta
    torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-6)


def test_vera_eval_ignores_stale_timestep_mask():
    base = torch.nn.Linear(4, 4, bias=False)
    bank = make_projection_bank(
        rank=2,
        max_in_features=4,
        max_out_features=4,
        projection_prng_key=1,
    )
    vera = VeRAModule(
        "lora_unet_test",
        base,
        multiplier=1.0,
        lora_dim=2,
        alpha=2,
        projection_bank=bank,
    )
    vera.apply_to()
    with torch.no_grad():
        vera.org_module_ref[0].weight.zero_()
        vera.vera_lambda_b.fill_(1.0)
        vera.vera_lambda_d.fill_(1.0)
        vera._timestep_mask.zero_()

    x = torch.ones(1, 4)
    vera.train()
    train_out = vera.org_module_ref[0](x)
    vera.eval()
    eval_out = vera.org_module_ref[0](x)

    torch.testing.assert_close(train_out, torch.zeros_like(train_out))
    assert torch.count_nonzero(eval_out).item() > 0
