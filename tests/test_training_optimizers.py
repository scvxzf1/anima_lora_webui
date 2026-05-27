from __future__ import annotations

from types import SimpleNamespace

import pytest
import pytorch_optimizer
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.training.optimizers import get_optimizer, is_schedulefree_optimizer
from library.training.schedulers import get_scheduler_fix

try:
    from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree
except ImportError:  # pragma: no cover - dependency lock/install test will catch this
    ProdigyPlusScheduleFree = None


def _optimizer_args(**overrides):
    args = {
        "optimizer_type": "CAME",
        "use_8bit_adam": False,
        "use_lion_optimizer": False,
        "fused_backward_pass": False,
        "gradient_accumulation_steps": 1,
        "optimizer_args": None,
        "learning_rate": 2e-4,
        "lr_scheduler": "constant",
        "max_grad_norm": 1.0,
        "unet_lr": None,
        "text_encoder_lr": None,
        "max_train_steps": 10,
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


def test_came_optimizer_short_name_builds_with_learning_rate_and_args():
    param = torch.nn.Parameter(torch.ones(2, 2))
    args = _optimizer_args(
        learning_rate=1e-4,
        optimizer_args=[
            "weight_decay=0.01",
            "betas=0.9,0.999,0.9999",
        ],
    )

    optimizer_name, optimizer_args, optimizer = get_optimizer(args, [param])

    assert isinstance(optimizer, pytorch_optimizer.CAME)
    assert optimizer_name == "pytorch_optimizer.optimizer.came.CAME"
    assert optimizer_args == "weight_decay=0.01,betas=(0.9, 0.999, 0.9999)"
    group = optimizer.param_groups[0]
    assert group["lr"] == pytest.approx(1e-4)
    assert group["weight_decay"] == pytest.approx(0.01)
    assert group["betas"] == (0.9, 0.999, 0.9999)
    assert not is_schedulefree_optimizer(optimizer, args)


def test_came_optimizer_fully_qualified_name_still_builds():
    param = torch.nn.Parameter(torch.ones(2, 2))
    args = _optimizer_args(optimizer_type="pytorch_optimizer.CAME")

    optimizer_name, _, optimizer = get_optimizer(args, [param])

    assert isinstance(optimizer, pytorch_optimizer.CAME)
    assert optimizer_name == "pytorch_optimizer.optimizer.came.CAME"


def test_came_optimizer_step_updates_matrix_parameter():
    param = torch.nn.Parameter(torch.ones(2, 2))
    args = _optimizer_args(learning_rate=1e-3)
    _, _, optimizer = get_optimizer(args, [param])
    before = param.detach().clone()

    loss = (param.square()).sum()
    loss.backward()
    optimizer.step()

    assert not torch.equal(param.detach(), before)


@pytest.mark.skipif(
    ProdigyPlusScheduleFree is None,
    reason="prodigy-plus-schedule-free is not installed",
)
def test_prodigy_plus_schedule_free_builds_with_args_and_dummy_scheduler():
    param = torch.nn.Parameter(torch.ones(2, 2))
    args = _optimizer_args(
        optimizer_type="ProdigyPlusScheduleFree",
        learning_rate=1.0,
        max_grad_norm=0.0,
        optimizer_args=[
            "betas=0.95,0.99",
            "eps=None",
            "use_speed=True",
        ],
    )

    optimizer_name, optimizer_args, optimizer = get_optimizer(args, [param])
    scheduler = get_scheduler_fix(args, optimizer, num_processes=1)

    assert isinstance(optimizer, ProdigyPlusScheduleFree)
    assert optimizer_name == "prodigyplus.prodigy_plus_schedulefree.ProdigyPlusScheduleFree"
    assert optimizer_args == "betas=(0.95, 0.99),eps=None,use_speed=True"
    assert optimizer.param_groups[0]["betas"] == (0.95, 0.99)
    assert optimizer.param_groups[0]["eps"] is None
    assert optimizer.param_groups[0]["use_speed"] is True
    assert is_schedulefree_optimizer(optimizer, args)
    assert scheduler.__class__.__name__ == "DummyScheduler"


@pytest.mark.skipif(
    ProdigyPlusScheduleFree is None,
    reason="prodigy-plus-schedule-free is not installed",
)
def test_prodigy_plus_schedule_free_step_updates_parameter():
    param = torch.nn.Parameter(torch.ones(2, 2))
    args = _optimizer_args(
        optimizer_type="ProdigyPlusScheduleFree",
        learning_rate=1.0,
        max_grad_norm=0.0,
    )
    _, _, optimizer = get_optimizer(args, [param])
    before = param.detach().clone()

    loss = (param.square()).sum()
    loss.backward()
    optimizer.step()

    assert not torch.equal(param.detach(), before)
    assert "effective_lr" in optimizer.param_groups[0]


@pytest.mark.skipif(
    ProdigyPlusScheduleFree is None,
    reason="prodigy-plus-schedule-free is not installed",
)
def test_prodigy_plus_schedule_free_can_disable_schedulefree_for_real_scheduler():
    param = torch.nn.Parameter(torch.ones(2, 2))
    args = _optimizer_args(
        optimizer_type="ProdigyPlusScheduleFree",
        learning_rate=1.0,
        max_grad_norm=0.0,
        lr_scheduler_type="CosineAnnealingLR",
        optimizer_args=["use_schedulefree=False"],
        lr_scheduler_args=["T_max=10"],
    )

    _, _, optimizer = get_optimizer(args, [param])
    scheduler = get_scheduler_fix(args, optimizer, num_processes=1)

    assert isinstance(optimizer, ProdigyPlusScheduleFree)
    assert not is_schedulefree_optimizer(optimizer, args)
    assert isinstance(scheduler, CosineAnnealingLR)
