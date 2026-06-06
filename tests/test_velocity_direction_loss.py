from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest
import torch

from library.training.losses import LOSS_REGISTRY, LossContext


def _ctx(pred: torch.Tensor, target: torch.Tensor, **arg_overrides) -> LossContext:
    args = dict(
        loss_type="l2",
        masked_loss=False,
        velocity_direction_loss_weight=1.0,
        velocity_direction_loss_eps=1e-6,
    )
    args.update(arg_overrides)
    return LossContext(
        args=argparse.Namespace(**args),
        batch={},
        model_pred=pred,
        target=target,
        timesteps=torch.zeros(pred.shape[0]),
        weighting=None,
        huber_c=None,
        loss_weights=torch.ones(pred.shape[0]),
        network=SimpleNamespace(),
        aux={},
        is_train=True,
    )


def test_velocity_direction_loss_zero_when_vectors_match():
    pred = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    target = pred.clone()

    loss = LOSS_REGISTRY["velocity_direction"](_ctx(pred, target))

    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_velocity_direction_loss_is_per_pixel_channel_cosine():
    pred = torch.tensor([[[[1.0]], [[0.0]]]])
    target = torch.tensor([[[[0.0]], [[1.0]]]])

    loss = LOSS_REGISTRY["velocity_direction"](_ctx(pred, target))

    assert loss.shape == (1,)
    assert loss.item() == pytest.approx(1.0, abs=1e-6)


def test_velocity_direction_loss_respects_weight_and_train_gate():
    pred = torch.tensor([[[[1.0]], [[0.0]]]])
    target = torch.tensor([[[[0.0]], [[1.0]]]])

    weighted = LOSS_REGISTRY["velocity_direction"](
        _ctx(pred, target, velocity_direction_loss_weight=0.25)
    )
    val_ctx = _ctx(pred, target, velocity_direction_loss_weight=0.25)
    val_ctx.is_train = False
    validation = LOSS_REGISTRY["velocity_direction"](val_ctx)

    assert weighted.item() == pytest.approx(0.25, abs=1e-6)
    assert torch.allclose(validation, torch.zeros_like(validation))


def test_velocity_direction_loss_is_finite_for_zero_vectors():
    pred = torch.zeros(2, 4, 3, 3)
    target = torch.zeros_like(pred)

    loss = LOSS_REGISTRY["velocity_direction"](_ctx(pred, target))

    assert torch.isfinite(loss).all()
