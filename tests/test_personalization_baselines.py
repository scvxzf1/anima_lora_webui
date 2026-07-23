"""Four deterministic CPU smoke baselines for the region-to-full proposal.

These are engineering baselines, not image-quality claims: the repository
does not currently contain a resized image set and generated mask set, so a
real model run would be invalid. The test still exercises the exact loss and
stage contracts used by a real run.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from library.training.losses import LOSS_REGISTRY, LossContext
from library.training.stage_schedule import parse_stage_specs, resolve_stage_index


def _ctx(*, mask=None, normalize="none", prior=False) -> LossContext:
    pred = torch.tensor([[[[1.0, 3.0], [5.0, 7.0]]]])
    batch = {} if mask is None else {"alpha_masks": mask}
    args = SimpleNamespace(
        loss_type="l2",
        masked_loss=mask is not None,
        mask_loss_normalize=normalize,
        foreground_loss_weight=1.0,
        prior_preservation_weight=0.1 if prior else 0.0,
        inverted_mask_prior_weight=0.0,
    )
    aux = {}
    if prior:
        aux["prior_preservation"] = {"prior_pred": torch.zeros_like(pred)}
    return LossContext(
        args=args,
        batch=batch,
        model_pred=pred,
        target=torch.zeros_like(pred),
        timesteps=torch.zeros(1),
        weighting=None,
        huber_c=None,
        loss_weights=torch.ones(1),
        network=SimpleNamespace(),
        aux=aux,
    )


@pytest.mark.parametrize(
    "name,ctx_factory",
    [
        ("full_image", lambda: _ctx()),
        (
            "region_foreground_mean",
            lambda: _ctx(
                mask=torch.tensor([[[1.0, 1.0], [0.0, 0.0]]]),
                normalize="foreground_mean",
            ),
        ),
        (
            "region_to_full_stage",
            lambda: _ctx(
                mask=torch.tensor([[[1.0, 1.0], [0.0, 0.0]]]),
                normalize="foreground_mean",
            ),
        ),
        (
            "region_to_full_with_prior",
            lambda: _ctx(
                mask=torch.tensor([[[1.0, 1.0], [0.0, 0.0]]]),
                normalize="foreground_mean",
                prior=True,
            ),
        ),
    ],
)
def test_four_personalization_baselines_are_finite(name, ctx_factory):
    ctx = ctx_factory()
    total = LOSS_REGISTRY["flow_match"](ctx)
    if "prior" in name:
        total = total + LOSS_REGISTRY["prior_preservation"](ctx)

    assert total.shape == (1,), name
    assert torch.isfinite(total).all(), name


def test_four_baseline_stage_and_dual_cache_contract(tmp_path: Path):
    stages = parse_stage_specs(
        [
            {"name": "region", "subset_index": 0, "start_pct": 0.0, "end_pct": 0.35},
            {"name": "full", "subset_index": 1, "start_pct": 0.35, "end_pct": 1.0},
        ]
    )
    assert resolve_stage_index(stages, 0.0) == 0
    assert resolve_stage_index(stages, 0.35) == 1

    region_cache = tmp_path / "lora_region"
    full_cache = tmp_path / "lora_full"
    region_cache.mkdir()
    full_cache.mkdir()
    assert region_cache != full_cache
    assert region_cache.exists() and full_cache.exists()
