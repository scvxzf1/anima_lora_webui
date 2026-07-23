from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from library.training.synchronous_affine import (
    affine_probabilities,
    apply_synchronous_affine,
)


def _args(**overrides):
    values = dict(
        adaptive_personalization_affine=True,
        adaptive_personalization_min_bin_samples=1,
        adaptive_personalization_affine_probability_max=1.0,
        adaptive_personalization_affine_rotation_deg=0.0,
        adaptive_personalization_affine_translation=0.0,
        adaptive_personalization_affine_scale_delta=0.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_affine_probability_is_gamma_driven_and_independent_of_weight_policy():
    state = {"bins": [{"count": 2, "gamma": 0.25}, {"count": 2, "gamma": 0.75}]}
    probabilities = affine_probabilities(state, _args(), torch.tensor([0.1, 0.9]))

    assert probabilities.tolist() == pytest.approx([0.25, 0.75])


def test_synchronous_affine_on_off_preserves_alignment_and_logs_fraction():
    torch.manual_seed(0)
    latents = torch.arange(4.0).reshape(1, 1, 2, 2)
    noise = latents + 10.0
    noisy = latents + 20.0
    mask = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    batch = {"alpha_masks": mask.clone()}
    args = _args()

    out = apply_synchronous_affine(
        latents,
        noise,
        noisy,
        batch,
        torch.ones(1),
        args,
    )

    assert out[-1] == 1.0
    assert out[0].shape == latents.shape
    assert out[1].shape == noise.shape
    assert out[2].shape == noisy.shape
    assert batch["alpha_masks"].shape == mask.shape
    assert torch.all((batch["alpha_masks"] >= 0) & (batch["alpha_masks"] <= 1))

    disabled = apply_synchronous_affine(
        latents,
        noise,
        noisy,
        {"alpha_masks": mask.clone()},
        torch.zeros(1),
        args,
    )
    assert disabled[-1] == 0.0
    assert torch.equal(disabled[0], latents)
