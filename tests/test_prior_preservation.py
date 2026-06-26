from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest
import torch

from library.training.cli_args import verify_training_args
from library.training.losses import LOSS_REGISTRY, LossContext
from library.training.prior_preservation import build_diff_output_prior_caption
from library.training.prior_preservation_forward import run_prior_preservation_forward


class _FakeNetwork:
    def __init__(self) -> None:
        self.multiplier = 1.25
        self.history: list[float] = []

    def set_multiplier(self, value: float) -> None:
        self.history.append(float(value))
        self.multiplier = float(value)


def _ctx(pred: torch.Tensor, prior_pred: torch.Tensor, **arg_overrides) -> LossContext:
    args = dict(
        loss_type="l2",
        masked_loss=False,
        prior_preservation_weight=0.5,
        inverted_mask_prior_weight=0.0,
    )
    args.update(arg_overrides)
    return LossContext(
        args=argparse.Namespace(**args),
        batch={},
        model_pred=pred,
        target=torch.zeros_like(pred),
        timesteps=torch.zeros(pred.shape[0]),
        weighting=None,
        huber_c=None,
        loss_weights=torch.ones(pred.shape[0]),
        network=SimpleNamespace(),
        aux={"prior_preservation": {"prior_pred": prior_pred}},
        is_train=True,
    )


def test_prior_preservation_forward_zeros_and_restores_multiplier():
    net = _FakeNetwork()
    seen: dict[str, object] = {}

    def anima_call(x, timesteps, crossattn_emb, padding_mask=None, **kwargs):
        seen["multiplier"] = net.multiplier
        seen["x"] = x
        seen["timesteps"] = timesteps
        seen["crossattn_emb"] = crossattn_emb
        seen["padding_mask"] = padding_mask
        seen["kwargs"] = kwargs
        return x + 2

    noisy = torch.ones(2, 4, 1, 3, 3)
    timesteps = torch.tensor([0.1, 0.7])
    crossattn = torch.randn(2, 8, 16)
    padding_mask = torch.zeros(2, 1, 3, 3)

    out = run_prior_preservation_forward(
        anima_call=anima_call,
        network=net,
        noisy_model_input=noisy,
        timesteps=timesteps,
        crossattn_emb=crossattn,
        padding_mask=padding_mask,
        forward_kwargs={"foo": "bar"},
    )

    assert torch.equal(out, noisy + 2)
    assert seen["multiplier"] == 0.0
    assert seen["x"] is noisy
    assert seen["timesteps"] is timesteps
    assert seen["crossattn_emb"] is crossattn
    assert seen["padding_mask"] is padding_mask
    assert seen["kwargs"] == {"foo": "bar"}
    assert net.multiplier == pytest.approx(1.25)
    assert net.history == [0.0, 1.25]


def test_prior_preservation_forward_prepares_block_swap_reference_forward():
    net = _FakeNetwork()

    class _BlockSwapAnima:
        blocks_to_swap = 2

        def __init__(self) -> None:
            self.prepare_calls: list[bool] = []
            self.call_saw_prepare = False

        def prepare_block_swap_before_forward(self, free_cache: bool = True) -> None:
            self.prepare_calls.append(free_cache)

        def __call__(self, x, timesteps, crossattn_emb, padding_mask=None, **kwargs):
            self.call_saw_prepare = bool(self.prepare_calls)
            return x + 3

    class _WrappedAnima:
        def __init__(self, module) -> None:
            self.module = module

        def __call__(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    anima = _BlockSwapAnima()
    noisy = torch.ones(1, 4, 1, 2, 2)

    out = run_prior_preservation_forward(
        anima_call=_WrappedAnima(anima),
        network=net,
        noisy_model_input=noisy,
        timesteps=torch.tensor([0.2]),
        crossattn_emb=torch.randn(1, 8, 16),
        padding_mask=torch.zeros(1, 1, 2, 2),
        forward_kwargs={},
    )

    assert torch.equal(out, noisy + 3)
    assert anima.prepare_calls == [False]
    assert anima.call_saw_prepare is True
    assert net.history == [0.0, 1.25]


def test_prior_preservation_loss_matches_weighted_mse_per_sample():
    pred = torch.tensor(
        [
            [[[[1.0, 3.0]]]],
            [[[[2.0, 4.0]]]],
        ]
    ).squeeze(2)
    prior = torch.tensor(
        [
            [[[[0.0, 1.0]]]],
            [[[[1.0, 1.0]]]],
        ]
    )
    ctx = _ctx(pred, prior, prior_preservation_weight=0.25)
    ctx.loss_weights = torch.tensor([1.0, 2.0])

    loss = LOSS_REGISTRY["prior_preservation"](ctx)

    assert loss.shape == (2,)
    assert loss[0].item() == pytest.approx(0.625)
    assert loss[1].item() == pytest.approx(2.5)


def test_prior_preservation_loss_is_train_only_and_requires_aux():
    pred = torch.ones(2, 1, 2, 2)
    prior = torch.zeros(2, 1, 2, 2)
    ctx = _ctx(pred, prior, prior_preservation_weight=1.0)

    ctx.is_train = False
    assert torch.equal(
        LOSS_REGISTRY["prior_preservation"](ctx),
        torch.zeros(pred.shape[0]),
    )

    ctx.is_train = True
    ctx.aux = {}
    assert torch.equal(
        LOSS_REGISTRY["prior_preservation"](ctx),
        torch.zeros(pred.shape[0]),
    )


def test_inverted_mask_prior_loss_only_applies_outside_alpha_mask():
    pred = torch.ones(2, 1, 2, 2)
    prior = torch.zeros(2, 1, 2, 2)
    ctx = _ctx(
        pred,
        prior,
        prior_preservation_weight=0.0,
        inverted_mask_prior_weight=0.5,
    )
    ctx.aux = {"inverted_mask_prior": {"prior_pred": prior}}
    ctx.batch = {
        "alpha_masks": torch.tensor(
            [
                [[1.0, 1.0], [0.0, 0.0]],
                [[1.0, 1.0], [1.0, 1.0]],
            ]
        )
    }

    loss = LOSS_REGISTRY["inverted_mask_prior"](ctx)

    assert loss.shape == (2,)
    assert loss[0].item() == pytest.approx(0.25)
    assert loss[1].item() == pytest.approx(0.0)


def _minimal_training_args(**overrides) -> SimpleNamespace:
    args = dict(
        use_vae_cache=False,
        use_text_cache=False,
        prior_preservation_weight=0.0,
        blank_prompt_preservation=False,
        diff_output_preservation_trigger=None,
        diff_output_preservation_class=None,
        inverted_mask_prior_weight=0.0,
        cache_llm_adapter_outputs=False,
        highvram=False,
        blocks_to_swap=None,
        v2=False,
        clip_skip=None,
        adaptive_noise_scale=None,
        noise_offset=None,
        scale_v_pred_loss_like_noise_pred=False,
        v_parameterization=False,
        v_pred_like_loss=None,
        zero_terminal_snr=False,
        sample_every_n_epochs=None,
        sample_every_n_steps=None,
        vae_batch_size=1,
        min_snr_gamma=None,
        max_train_steps=1,
        save_every_n_epochs=None,
        save_every_n_steps=None,
        save_n_epoch_ratio=None,
        save_last_n_epochs=None,
        save_last_n_epochs_state=None,
        save_last_n_steps=None,
        save_last_n_steps_state=None,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def test_prior_preservation_weight_requires_prior_mode():
    args = _minimal_training_args(prior_preservation_weight=0.1)

    with pytest.raises(ValueError, match="blank_prompt_preservation=true"):
        verify_training_args(args)


def test_blank_prompt_preservation_requires_text_cache_then_crossattn_outputs():
    args = _minimal_training_args(
        prior_preservation_weight=0.1,
        blank_prompt_preservation=True,
    )

    with pytest.raises(ValueError, match="use_text_cache=true"):
        verify_training_args(args)

    args.use_text_cache = True
    with pytest.raises(ValueError, match="cache_llm_adapter_outputs=true"):
        verify_training_args(args)


def test_diff_output_prior_caption_replaces_trigger_or_uses_class_prompt():
    assert (
        build_diff_output_prior_caption(
            "sks woman, portrait, sks style",
            trigger="sks",
            class_prompt="woman",
        )
        == "woman woman, portrait, woman style"
    )
    assert (
        build_diff_output_prior_caption(
            "sks woman, portrait",
            trigger=None,
            class_prompt="woman",
        )
        == "woman"
    )


def test_diff_output_preservation_requires_text_cache_and_crossattn_outputs():
    args = _minimal_training_args(
        prior_preservation_weight=0.1,
        diff_output_preservation_class="woman",
    )

    with pytest.raises(ValueError, match="use_text_cache=true"):
        verify_training_args(args)

    args.use_text_cache = True
    with pytest.raises(ValueError, match="cache_llm_adapter_outputs=true"):
        verify_training_args(args)

    args.cache_llm_adapter_outputs = True
    verify_training_args(args)


def test_prior_preservation_modes_are_mutually_exclusive():
    args = _minimal_training_args(
        use_text_cache=True,
        prior_preservation_weight=0.1,
        blank_prompt_preservation=True,
        diff_output_preservation_class="woman",
        cache_llm_adapter_outputs=True,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        verify_training_args(args)


def test_inverted_mask_prior_requires_text_cache_and_crossattn_outputs():
    args = _minimal_training_args(inverted_mask_prior_weight=0.1)

    with pytest.raises(ValueError, match="use_text_cache=true"):
        verify_training_args(args)

    args.use_text_cache = True
    with pytest.raises(ValueError, match="cache_llm_adapter_outputs=true"):
        verify_training_args(args)

    args.cache_llm_adapter_outputs = True
    verify_training_args(args)
