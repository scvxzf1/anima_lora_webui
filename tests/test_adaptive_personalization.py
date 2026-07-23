from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from library.training.adaptive_personalization import (
    dynamic_denoise_weights,
    metrics,
    should_probe,
    update_observation,
)
from library.training.losses import add_custom_train_arguments
from library.training.progress import ProgressSink


def _args(**overrides):
    values = dict(
        adaptive_personalization_probe_every_n_steps=2,
        adaptive_personalization_timestep_bins=2,
        adaptive_personalization_ema_decay=0.0,
        adaptive_personalization_gamma_scale=1.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_probe_interval_is_deterministic():
    state = {}
    args = _args()
    assert should_probe(state, args) is True
    assert should_probe(state, args) is False
    assert should_probe(state, args) is True
    assert state["calls"] == 3


def test_probe_interval_uses_optimizer_step_and_deduplicates_microbatches():
    state = {"global_step": 4}
    args = _args(adaptive_personalization_probe_every_n_steps=2)

    assert should_probe(state, args) is True
    assert should_probe(state, args) is False
    state["global_step"] = 5
    assert should_probe(state, args) is False
    state["global_step"] = 6
    assert should_probe(state, args) is True
    assert state["calls"] == 3


def test_observation_updates_gamma_by_timestep_bin_without_policy_output():
    state = {}
    args = _args()
    target = torch.zeros(2, 1, 1, 1)
    adapter = torch.zeros_like(target)
    base = torch.tensor([[[[2.0]]], [[[1.0]]]])
    timesteps = torch.tensor([0.1, 0.9])

    update_observation(
        state,
        args,
        timesteps=timesteps,
        adapter_pred=adapter,
        base_pred=base,
        target=target,
    )

    values = metrics(state)
    assert values["personalization/observed_bins"] == 2.0
    assert values["personalization/gamma_bin_00"] == pytest.approx(
        1.0 - torch.exp(torch.tensor(-4.0)).item()
    )
    assert values["personalization/gamma_bin_01"] == pytest.approx(
        1.0 - torch.exp(torch.tensor(-1.0)).item()
    )
    assert "personalization/gamma_max" in values
    assert "personalization/weight" not in values


def test_dynamic_loss_weighting_has_independent_on_off_ablation():
    state = {
        "bins": [
            {"count": 20, "gamma": 0.8},
            {"count": 20, "gamma": 0.1},
        ]
    }
    timesteps = torch.tensor([0.1, 0.9])
    disabled = dynamic_denoise_weights(
        state,
        _args(adaptive_personalization_loss_weighting=False),
        timesteps,
    )
    enabled = dynamic_denoise_weights(
        state,
        _args(
            adaptive_personalization_loss_weighting=True,
            adaptive_personalization_min_bin_samples=16,
            adaptive_personalization_denoise_weight_min=0.25,
        ),
        timesteps,
    )

    assert disabled is None
    assert enabled is not None
    assert enabled.tolist() == pytest.approx([0.25, 0.9])


def test_dynamic_loss_weighting_waits_for_bin_warmup():
    state = {"bins": [{"count": 2, "gamma": 0.9}, {"count": 20, "gamma": 0.5}]}
    weights = dynamic_denoise_weights(
        state,
        _args(
            adaptive_personalization_loss_weighting=True,
            adaptive_personalization_min_bin_samples=16,
            adaptive_personalization_denoise_weight_min=0.25,
        ),
        torch.tensor([0.1, 0.9]),
    )

    assert weights.tolist() == pytest.approx([1.0, 0.5])


def test_observer_fields_round_trip_through_progress_jsonl(tmp_path):
    state = {}
    args = _args()
    update_observation(
        state,
        args,
        timesteps=torch.tensor([0.1]),
        adapter_pred=torch.zeros(1, 1, 1, 1),
        base_pred=torch.ones(1, 1, 1, 1),
        target=torch.zeros(1, 1, 1, 1),
    )
    path = tmp_path / "progress.jsonl"
    sink = ProgressSink(str(path), run="test", method="lora", preset="default")
    sink.run_start(total_steps=1, total_epochs=1, pid=1)
    sink.log(metrics(state), global_step=1, epoch=1)
    sink.run_end(status="ok", final_step=1)

    records = [line for line in path.read_text().splitlines()]
    assert '"ev": "step"' in records[1]
    assert '"personalization/gamma_bin_00"' in records[1]
    assert records[-1].startswith('{"ev": "run_end"')


def test_observer_cli_fields_are_explicit_and_parseable():
    import argparse

    parser = argparse.ArgumentParser()
    add_custom_train_arguments(parser, support_weighted_captions=False)
    parsed = parser.parse_args(
        [
            "--adaptive_personalization_observe",
            "--adaptive_personalization_timestep_bins",
            "8",
            "--adaptive_personalization_probe_every_n_steps",
            "3",
        ]
    )

    assert parsed.adaptive_personalization_observe is True
    assert parsed.adaptive_personalization_timestep_bins == 8
    assert parsed.adaptive_personalization_probe_every_n_steps == 3
