import pytest
import torch
import torch.nn.functional as F

from scripts.krea2.probe_nf4_correction import (
    CorrectionFactors,
    FixedCorrectionBank,
    fit_activation_weighted_factors,
)

pytestmark = pytest.mark.probe


def test_activation_weighted_factors_recover_low_rank_error() -> None:
    generator = torch.Generator().manual_seed(5)
    activation = torch.randn(32, 8, generator=generator, dtype=torch.bfloat16)
    quantized = torch.randn(6, 8, generator=generator, dtype=torch.bfloat16)
    left = torch.randn(6, 3, generator=generator)
    right = torch.randn(3, 8, generator=generator)
    bf16_weight = (quantized.float() + left @ right).to(torch.bfloat16)

    down, up = fit_activation_weighted_factors(
        activation,
        bf16_weight,
        quantized,
        rank=3,
        input_rank=8,
        niter=2,
        seed=7,
    )

    expected = activation.float() @ (bf16_weight.float() - quantized.float()).T
    actual = F.linear(F.linear(activation, down), up).float()
    relative_l2 = (actual - expected).norm() / expected.norm()
    assert relative_l2 < 0.02


def test_fixed_correction_bank_adds_and_removes_hook() -> None:
    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Module()])
            self.blocks[0].attn = torch.nn.Module()
            self.blocks[0].attn.wq = torch.nn.Linear(4, 3, bias=False)

        def forward(self, activation: torch.Tensor) -> torch.Tensor:
            return self.blocks[0].attn.wq(activation)

    model = TinyModel()
    activation = torch.randn(2, 4)
    down = torch.randn(2, 4)
    up = torch.randn(3, 2)
    baseline = model(activation)
    expected_delta = F.linear(F.linear(activation, down), up)
    bank = FixedCorrectionBank(
        [CorrectionFactors("blocks.0.attn.wq", down, up)]
    )

    bank.attach(model)
    assert torch.equal(model(activation), baseline + expected_delta)
    bank.enabled = False
    assert torch.equal(model(activation), baseline)
    bank.enabled = True
    bank.remove()
    assert torch.equal(model(activation), baseline)
