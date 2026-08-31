import math

import pytest
import torch

from scripts.krea2.probe_nf4_ex_spectrum import (
    _aggregate_layers,
    _spectrum_summary,
)

pytestmark = pytest.mark.probe


def test_spectrum_summary_exact_energy() -> None:
    matrix = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))

    result = _spectrum_summary(
        matrix,
        ranks=(1, 2, 4),
        oversample=2,
        niter=1,
        exact_max=8,
        seed=1,
    )

    assert result["method"] == "exact"
    assert result["total_energy"] == 30.0
    assert math.isclose(result["captured_energy"]["1"], 16.0 / 30.0)
    assert math.isclose(result["captured_energy"]["2"], 25.0 / 30.0)
    assert result["captured_energy"]["4"] == 1.0
    assert math.isclose(result["residual_l2_factor"]["2"], math.sqrt(5.0 / 30.0))


def test_spectrum_summary_randomized_recovers_low_rank_matrix() -> None:
    generator = torch.Generator().manual_seed(7)
    left, _ = torch.linalg.qr(torch.randn(96, 8, generator=generator))
    right, _ = torch.linalg.qr(torch.randn(64, 8, generator=generator))
    singular_values = torch.tensor([9.0, 7.0, 5.0, 3.0, 2.0, 1.5, 1.0, 0.5])
    matrix = (left * singular_values) @ right.T

    result = _spectrum_summary(
        matrix,
        ranks=(4, 8),
        oversample=4,
        niter=2,
        exact_max=0,
        seed=11,
    )

    expected_rank4 = singular_values[:4].square().sum() / singular_values.square().sum()
    assert result["method"] == "randomized"
    assert math.isclose(
        result["captured_energy"]["4"], float(expected_rank4), rel_tol=1e-5
    )
    assert math.isclose(result["captured_energy"]["8"], 1.0, rel_tol=1e-5)


def test_aggregate_layers_weights_captured_energy() -> None:
    layers = [
        {
            "kind": "attn.wq",
            "total_energy": 3.0,
            "captured_energy": {"4": 0.5},
        },
        {
            "kind": "attn.wq",
            "total_energy": 1.0,
            "captured_energy": {"4": 1.0},
        },
    ]

    result = _aggregate_layers(layers, (4,))

    assert result["all"]["layers"] == 2
    assert result["all"]["rank_4"]["energy_weighted"] == 0.625
    assert result["by_kind"]["attn.wq"]["rank_4"]["layers_ge_50pct"] == 2
