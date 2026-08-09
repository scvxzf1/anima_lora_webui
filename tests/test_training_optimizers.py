from __future__ import annotations

from types import SimpleNamespace

import pytest
import pytorch_optimizer
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.training.automagic import Automagic
from library.training.optimizers import (
    get_optimizer,
    is_schedulefree_optimizer,
    is_self_managed_lr_optimizer,
)
from library.training.schedulers import get_scheduler_fix
from networks.lora_anima import optimizer_groups

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


def test_automagic_optimizer_builds_and_uses_dummy_scheduler_with_effective_lrs():
    param = torch.nn.Parameter(torch.ones(2, 2))
    args = _optimizer_args(
        optimizer_type="Automagic",
        learning_rate=1e-6,
        optimizer_args=[
            "min_lr=1e-7",
            "max_lr=1e-5",
            "lr_bump=1e-6",
            "weight_decay=0.0",
        ],
    )

    optimizer_name, optimizer_args, optimizer = get_optimizer(args, [param])
    scheduler = get_scheduler_fix(args, optimizer, num_processes=1)

    assert isinstance(optimizer, Automagic)
    assert optimizer_name == "library.training.automagic.Automagic"
    assert optimizer_args == "min_lr=1e-07,max_lr=1e-05,lr_bump=1e-06,weight_decay=0.0"
    assert is_self_managed_lr_optimizer(optimizer, args)
    assert not is_schedulefree_optimizer(optimizer, args)
    assert scheduler.__class__.__name__ == "DummyScheduler"
    assert scheduler.get_last_lr() == pytest.approx([1e-6])

    before = param.detach().clone()
    loss = -param.sum()
    loss.backward()
    optimizer.step()

    assert not torch.equal(param.detach(), before)
    assert optimizer.get_learning_rates()[0] > 1e-6
    assert scheduler.get_last_lr() == pytest.approx(optimizer.get_learning_rates())


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


class FakeLora(torch.nn.Module):
    def __init__(self, original_name: str, lora_name: str) -> None:
        super().__init__()
        self.original_name = original_name
        self.lora_name = lora_name
        self.lora_down = torch.nn.Linear(2, 1, bias=False)
        self.lora_up = torch.nn.Linear(1, 2, bias=False)
        self.router = torch.nn.Linear(2, 2, bias=False)


class FakeNetwork(torch.nn.Module):
    LORA_PREFIX_TEXT_ENCODER = "lora_te1"

    def __init__(
        self,
        *,
        unet_loras=None,
        global_router=None,
        freq_router=None,
        content_router=None,
        cfg=None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or SimpleNamespace(
            reg_lrs=None,
            router_lr_scale=3.0,
            content_router_lr_scale=7.0,
            freq_router_lr_scale=5.0,
            register_lr_scale=11.0,
            use_chimera_hydra=False,
        )
        self.text_encoder_loras = torch.nn.ModuleList()
        self.unet_loras = torch.nn.ModuleList(unet_loras or [])
        self.text_encoder_refts = torch.nn.ModuleList()
        self.unet_refts = torch.nn.ModuleList()
        self.global_router = global_router
        self.freq_router = freq_router
        self.content_router = content_router
        self.register_injector = None
        self.register_tokens = None
        self.loraplus_lr_ratio = None
        self.loraplus_unet_lr_ratio = None
        self.loraplus_text_encoder_lr_ratio = None


def _optimizer_group_lrs(params, descriptions):
    return {description: group["lr"] for group, description in zip(params, descriptions)}


def test_lora_optimizer_groups_keep_reg_lrs_loraplus_and_router_order() -> None:
    cfg = SimpleNamespace(
        reg_lrs={r"blocks\.0\.q_proj": 3e-4},
        router_lr_scale=5.0,
        content_router_lr_scale=7.0,
        freq_router_lr_scale=1.0,
        register_lr_scale=1.0,
        use_chimera_hydra=True,
    )
    network = FakeNetwork(
        cfg=cfg,
        unet_loras=[
            FakeLora("blocks.0.q_proj", "lora_unet_blocks_0_q_proj"),
            FakeLora("blocks.1.q_proj", "lora_unet_blocks_1_q_proj"),
        ],
    )
    optimizer_groups.set_loraplus_lr_ratio(network, 2.0, None, None)

    params, descriptions = optimizer_groups.prepare_lora_optimizer_params(
        network, None, 1e-4, 9e-5
    )

    assert descriptions == [
        "unet reg_lr_0",
        "unet reg_lr_0 plus",
        "unet reg_lr_0 router",
        "unet",
        "unet plus",
        "unet router",
    ]
    lrs = _optimizer_group_lrs(params, descriptions)
    assert lrs["unet reg_lr_0"] == pytest.approx(3e-4)
    assert lrs["unet reg_lr_0 plus"] == pytest.approx(6e-4)
    assert lrs["unet reg_lr_0 router"] == pytest.approx(3e-4 * 5.0 * 7.0)
    assert lrs["unet"] == pytest.approx(1e-4)
    assert lrs["unet plus"] == pytest.approx(2e-4)
    assert lrs["unet router"] == pytest.approx(1e-4 * 5.0 * 7.0)


def test_lora_optimizer_groups_keep_network_router_lr_scales() -> None:
    network = FakeNetwork(
        global_router=torch.nn.Linear(2, 2, bias=False),
        freq_router=torch.nn.Linear(2, 2, bias=False),
        content_router=torch.nn.Linear(2, 2, bias=False),
    )

    params, descriptions = optimizer_groups.prepare_lora_optimizer_params(
        network, None, 1e-4, 9e-5
    )

    assert descriptions == [
        "global router",
        "chimera freq router",
        "chimera content router",
    ]
    lrs = _optimizer_group_lrs(params, descriptions)
    assert lrs["global router"] == pytest.approx(1e-4 * 3.0)
    assert lrs["chimera freq router"] == pytest.approx(1e-4 * 3.0 * 5.0)
    assert lrs["chimera content router"] == pytest.approx(1e-4 * 3.0 * 7.0)
