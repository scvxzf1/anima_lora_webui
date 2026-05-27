from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from library.training.schedulers import (
    LULU_LOSS_GATED_COSINE,
    LULU_LLOSS_WEIGHTED_ANNEALED_COSINE,
    LuluLossWeightedAnnealedCosineScheduler,
    get_scheduler_fix,
)


def _optimizer():
    param = torch.nn.Parameter(torch.ones(()))
    return torch.optim.SGD([param], lr=1.0)


def _scheduler(**overrides):
    kwargs = {
        "num_warmup_steps": 0,
        "num_training_steps": 100,
        "gamma": 2.0,
        "loss_ema_beta": 0.0,
        "loss_min_delta_rel": 0.002,
        "loss_min_delta_abs": 0.0,
        "plateau_patience_steps": 8,
        "improvement_cooldown_steps": 4,
        "max_hold_steps": 32,
        "min_phase_advance_ratio": 0.05,
        "loss_gate_start_weight": 0.25,
        "min_lr_ratio": 0.0,
    }
    kwargs.update(overrides)
    opt = _optimizer()
    return opt, LuluLossWeightedAnnealedCosineScheduler(opt, **kwargs)


def _tick(optimizer, scheduler, loss):
    optimizer.step()
    scheduler.step(loss=loss)
    return scheduler.get_last_lr()[0]


def _cosine_lr(progress: float) -> float:
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _optimizer_args(**overrides):
    args = {
        "optimizer_type": "AdamW",
        "use_8bit_adam": False,
        "use_lion_optimizer": False,
        "fused_backward_pass": False,
        "gradient_accumulation_steps": 1,
        "optimizer_args": None,
        "learning_rate": 1.0,
        "lr_scheduler": LULU_LOSS_GATED_COSINE,
        "max_grad_norm": 1.0,
        "unet_lr": None,
        "text_encoder_lr": None,
        "max_train_steps": 100,
        "lr_warmup_steps": 0,
        "lr_decay_steps": 0,
        "lr_scheduler_num_cycles": 1,
        "lr_scheduler_power": 1.0,
        "lr_scheduler_timescale": None,
        "lr_scheduler_min_lr_ratio": None,
        "lr_scheduler_args": None,
        "lr_scheduler_type": "",
    }
    args.update(overrides)
    return SimpleNamespace(**args)


def test_before_loss_gate_threshold_matches_plain_cosine():
    opt, scheduler = _scheduler(loss_gate_start_weight=0.95)

    for step in range(1, 11):
        lr = _tick(opt, scheduler, loss=1.0)
        progress = step / scheduler.num_training_steps
        assert scheduler._last_phase_progress == pytest.approx(progress)
        assert lr == pytest.approx(_cosine_lr(progress))


def test_effective_loss_drop_holds_phase_with_minimum_advance():
    opt, scheduler = _scheduler(
        loss_gate_start_weight=0.0,
        min_phase_advance_ratio=0.1,
        plateau_patience_steps=20,
    )

    _tick(opt, scheduler, loss=1.0)
    phase_after_first = scheduler.loss_gate_phase
    _tick(opt, scheduler, loss=0.7)
    _tick(opt, scheduler, loss=0.4)

    assert scheduler.best_loss == pytest.approx(0.4)
    assert scheduler.loss_gate_phase > phase_after_first
    assert scheduler.loss_gate_phase < scheduler._step_progress()


def test_small_loss_wobble_is_not_effective_improvement():
    opt, scheduler = _scheduler(
        loss_gate_start_weight=0.0,
        loss_min_delta_rel=0.10,
        plateau_patience_steps=20,
    )

    _tick(opt, scheduler, loss=1.0)
    _tick(opt, scheduler, loss=0.95)

    assert scheduler.best_loss == pytest.approx(1.0)
    assert scheduler.plateau_steps == 2


def test_plateau_patience_releases_phase_to_step_progress():
    opt, scheduler = _scheduler(
        loss_gate_start_weight=0.0,
        plateau_patience_steps=2,
        improvement_cooldown_steps=0,
    )

    _tick(opt, scheduler, loss=1.0)
    _tick(opt, scheduler, loss=1.0)

    assert scheduler.plateau_steps == 2
    assert scheduler.loss_gate_phase == pytest.approx(scheduler._step_progress())


def test_improvement_cooldown_ignores_short_plateau():
    opt, scheduler = _scheduler(
        loss_gate_start_weight=0.0,
        plateau_patience_steps=1,
        improvement_cooldown_steps=2,
    )

    _tick(opt, scheduler, loss=1.0)
    _tick(opt, scheduler, loss=0.5)
    _tick(opt, scheduler, loss=0.5)

    assert scheduler.cooldown_steps == 1
    assert scheduler.plateau_steps == 0
    assert scheduler.loss_gate_phase < scheduler._step_progress()


def test_max_hold_steps_forces_phase_progress():
    opt, scheduler = _scheduler(
        loss_gate_start_weight=0.0,
        plateau_patience_steps=20,
        max_hold_steps=2,
    )

    _tick(opt, scheduler, loss=1.0)
    _tick(opt, scheduler, loss=0.8)
    _tick(opt, scheduler, loss=0.6)

    assert scheduler.hold_steps == 0
    assert scheduler.loss_gate_phase == pytest.approx(scheduler._step_progress())


def test_state_dict_roundtrip_keeps_next_lr_continuous():
    opt1, scheduler1 = _scheduler(loss_gate_start_weight=0.0)
    opt2, scheduler2 = _scheduler(loss_gate_start_weight=0.0)

    for loss in [1.0, 0.9, 0.8, 0.8, 0.8]:
        _tick(opt1, scheduler1, loss=loss)
    scheduler2.load_state_dict(scheduler1.state_dict())

    lr1 = _tick(opt1, scheduler1, loss=0.7)
    lr2 = _tick(opt2, scheduler2, loss=0.7)

    assert lr2 == pytest.approx(lr1)
    assert scheduler2.loss_ema == pytest.approx(scheduler1.loss_ema)
    assert scheduler2.best_loss == pytest.approx(scheduler1.best_loss)


def test_queued_loss_is_consumed_once_across_inner_accelerate_steps():
    opt, scheduler = _scheduler(loss_gate_start_weight=0.0, plateau_patience_steps=20)

    scheduler.set_step_loss(1.0)
    for _ in range(4):
        opt.step()
        scheduler.step()

    assert scheduler.last_epoch == 4
    assert scheduler.plateau_steps == 1
    assert scheduler.loss_ema == pytest.approx(1.0)
    assert scheduler.best_loss == pytest.approx(1.0)
    assert scheduler._next_step_loss is None


def test_get_scheduler_fix_builds_lulu_scheduler_with_args():
    opt = _optimizer()
    args = _optimizer_args(
        lr_scheduler_args=[
            "gamma=3.0",
            "plateau_patience_steps=2",
            "loss_gate_start_weight=0.5",
        ],
        lr_scheduler_min_lr_ratio=0.1,
    )

    scheduler = get_scheduler_fix(args, opt, num_processes=1)

    assert isinstance(scheduler, LuluLossWeightedAnnealedCosineScheduler)
    assert scheduler.gamma == pytest.approx(3.0)
    assert scheduler.plateau_patience_steps == 2
    assert scheduler.loss_gate_start_weight == pytest.approx(0.5)
    assert scheduler.min_lr_ratio == pytest.approx(0.1)


def test_get_scheduler_fix_keeps_legacy_lulu_name_as_alias():
    opt = _optimizer()
    args = _optimizer_args(lr_scheduler=LULU_LLOSS_WEIGHTED_ANNEALED_COSINE)

    scheduler = get_scheduler_fix(args, opt, num_processes=1)

    assert isinstance(scheduler, LuluLossWeightedAnnealedCosineScheduler)
