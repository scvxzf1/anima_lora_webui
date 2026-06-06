from __future__ import annotations

import torch

from library.anima.training import compute_loss_weighting_for_anima


def test_min_snr_weighting_downweights_high_snr_low_sigma():
    sigmas = torch.tensor([0.1, 0.5, 0.9])

    weights = compute_loss_weighting_for_anima(
        "min_snr", sigmas, min_snr_gamma=5.0
    )

    assert weights[0] < weights[1]
    assert torch.allclose(weights[1:], torch.ones(2), atol=1e-5)
    assert torch.isfinite(weights).all()


def test_p2_weighting_downweights_high_snr_low_sigma():
    sigmas = torch.tensor([0.1, 0.5, 0.9])

    weights = compute_loss_weighting_for_anima(
        "p2", sigmas, p2_gamma=1.0, p2_k=1.0
    )

    assert weights[0] < weights[1] < weights[2]
    assert torch.isfinite(weights).all()


def test_snr_weighting_is_finite_at_edges():
    sigmas = torch.tensor([0.0, 1.0])

    min_snr = compute_loss_weighting_for_anima("min_snr", sigmas, min_snr_gamma=5.0)
    p2 = compute_loss_weighting_for_anima("p2", sigmas, p2_gamma=0.5, p2_k=1.0)

    assert torch.isfinite(min_snr).all()
    assert torch.isfinite(p2).all()
    assert (min_snr >= 0).all()
    assert (p2 >= 0).all()
