from types import SimpleNamespace

import pytest
import torch

from bench.adaptive_runtime.scaling import backward_unscaled, make_scaler, validate_scaling


def gradients(dtype, scale):
    parameter = torch.nn.Parameter(torch.ones(2))
    noisy = torch.ones(2, requires_grad=True)
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler = make_scaler(scale, device="cpu")
    hidden = (parameter * noisy).to(dtype)
    prediction = torch.nn.functional.linear(hidden, torch.ones(2, 2, dtype=dtype)).float()
    backward_unscaled(prediction.sum() * 1e-8, noisy, optimizer, scaler)
    return parameter.grad.clone(), noisy.grad.clone()


def test_scaling_preserves_tiny_gradients_and_unscales_input_leaf():
    reference = gradients(torch.float32, 1)
    unscaled = gradients(torch.float16, 1)
    scaled = gradients(torch.float16, 1024)
    assert all(torch.count_nonzero(t) == 0 for t in unscaled)
    for actual, expected in zip(scaled, reference, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0)


@pytest.mark.parametrize("change", [{"loss_scale": 0}, {"loss_scale": float("nan")},
                                    {"disposable_probe": False}, {"capture_training": False},
                                    {"resume": "checkpoint"}, {"checkpoint_every_step": True},
                                    {"precision": "bf16"}])
def test_scaling_cannot_silently_change_regular_training_or_resume(change):
    values = dict(loss_scale=128, precision="fp16-islands", disposable_probe=True,
                  capture_training=True, resume=None, checkpoint_every_step=False)
    values.update(change)
    with pytest.raises(ValueError):
        validate_scaling(SimpleNamespace(**values))


def scaled_checkpoint_args(**changes):
    values = dict(loss_scale=1024, precision="fp16-islands", scaled_checkpoint=True,
                  checkpoint_every_step=True, inputs="inputs", disposable_probe=False,
                  capture_training=False, capture_linear=[], record_replay=False,
                  replay_reference=None, resume=None)
    return SimpleNamespace(**{**values, **changes})


def test_scaled_checkpoint_requires_explicit_flag_and_accepts_resume():
    validate_scaling(scaled_checkpoint_args())
    validate_scaling(scaled_checkpoint_args(resume="step-1.pt"))
    with pytest.raises(ValueError):
        validate_scaling(scaled_checkpoint_args(scaled_checkpoint=False))


def test_scaled_checkpoint_accepts_worker_without_replay_options():
    args = scaled_checkpoint_args()
    del args.record_replay
    del args.replay_reference
    validate_scaling(args)


@pytest.mark.parametrize("change", [{"loss_scale": 1}, {"precision": "bf16"},
                                    {"checkpoint_every_step": False}, {"inputs": None},
                                    {"disposable_probe": True}, {"capture_training": True},
                                    {"capture_linear": ["layer"]}, {"record_replay": True},
                                    {"replay_reference": "reference"}])
def test_scaled_checkpoint_rejects_incompatible_modes(change):
    with pytest.raises(ValueError, match="Scaled checkpoint"):
        validate_scaling(scaled_checkpoint_args(**change))
