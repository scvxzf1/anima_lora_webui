from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from library.training.losses import LossContext, LOSS_REGISTRY
from library.training.mask_loss import reduce_masked_loss
from library.training.cli_args import add_masked_loss_arguments


def test_foreground_mean_is_independent_of_mask_area():
    loss = torch.tensor([[[[2.0, 2.0], [2.0, 2.0]]]])
    small = {"alpha_masks": torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])}
    large = {"alpha_masks": torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])}

    assert reduce_masked_loss(
        loss, small, normalize="foreground_mean"
    ).item() == pytest.approx(2.0)
    assert reduce_masked_loss(
        loss, large, normalize="foreground_mean"
    ).item() == pytest.approx(2.0)


def test_foreground_mean_empty_mask_falls_back_to_full_mean():
    loss = torch.tensor([[[[1.0, 3.0], [5.0, 7.0]]]])
    batch = {"alpha_masks": torch.zeros(1, 2, 2)}

    result = reduce_masked_loss(loss, batch, normalize="foreground_mean")

    assert result.item() == pytest.approx(4.0)
    assert torch.isfinite(result).all()


def test_foreground_mean_zero_weight_is_respected():
    loss = torch.ones(1, 1, 2, 2)
    batch = {"alpha_masks": torch.ones(1, 2, 2)}

    result = reduce_masked_loss(
        loss,
        batch,
        normalize="foreground_mean",
        foreground_weight=0.0,
    )

    assert result.item() == 0.0


def test_flow_match_uses_foreground_mean_after_timestep_weighting():
    pred = torch.tensor([[[[1.0, 3.0], [5.0, 7.0]]]])
    target = torch.zeros_like(pred)
    ctx = LossContext(
        args=SimpleNamespace(
            loss_type="l2",
            masked_loss=True,
            mask_loss_normalize="foreground_mean",
            foreground_loss_weight=0.5,
        ),
        batch={"alpha_masks": torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])},
        model_pred=pred,
        target=target,
        timesteps=torch.zeros(1),
        weighting=None,
        huber_c=None,
        loss_weights=torch.ones(1),
        network=SimpleNamespace(),
        aux={},
    )

    result = LOSS_REGISTRY["flow_match"](ctx)

    assert result.item() == pytest.approx(2.5, abs=1e-6)


def test_mask_loss_cli_defaults_and_choices():
    import argparse

    parser = argparse.ArgumentParser()
    add_masked_loss_arguments(parser)

    defaults = parser.parse_args([])
    configured = parser.parse_args(
        ["--mask_loss_normalize", "foreground_mean", "--foreground_loss_weight", "0.25"]
    )

    assert defaults.mask_loss_normalize == "none"
    assert defaults.foreground_loss_weight == 1.0
    assert configured.mask_loss_normalize == "foreground_mean"
    assert configured.foreground_loss_weight == pytest.approx(0.25)
