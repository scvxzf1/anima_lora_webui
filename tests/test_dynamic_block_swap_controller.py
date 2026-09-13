from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from library.training.auto_block_swap import online
from library.training.auto_block_swap.online_policy import Decision, OnlineSwapPolicy
from library.training.auto_block_swap.resources import GIB, HostMemory


@pytest.fixture
def controller(monkeypatch, tmp_path):
    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return 1000

    host = HostMemory(64 * GIB, 32 * GIB, 8 * GIB, 0, 0, 0)
    monkeypatch.setattr(online, "host_memory", lambda: host)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=20 * GIB),
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: 4 * GIB)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (16 * GIB, 20 * GIB))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _: 4 * GIB)
    model = SimpleNamespace(
        blocks_to_swap=4,
        blocks=[None] * 6,
        offloader=SimpleNamespace(reconfigure=Mock(return_value=True)),
    )
    state = SimpleNamespace(
        args=SimpleNamespace(
            _auto_swap_maximum=4,
            _auto_swap_block_bytes=[GIB] * 6,
            auto_block_swap_interval=4,
            max_train_steps=100,
            _auto_swap_report=str(tmp_path / "summary.json"),
        ),
        unet=model,
        accelerator=SimpleNamespace(
            unwrap_model=lambda model: model,
            device="cuda:0",
            sync_gradients=True,
            optimizer_step_was_skipped=False,
        ),
        global_step=0,
    )
    trainer = SimpleNamespace(is_swapping_blocks=True)
    return online.OnlineSwapController(trainer, state), state


def batch(size=8):
    return {"latents": torch.zeros(1, 16, size, size)}


def test_controller_custom_reserve_uses_capacity_not_free_memory(controller, monkeypatch):
    control, state = controller
    state.args.auto_block_swap_vram_reserve_percent = 25
    state.args.auto_block_swap_preference = "ram"
    control = online.OnlineSwapController(control.trainer, state)
    assert control.reserve == 5 * GIB
    assert control.policy.preference == "ram"
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (6 * GIB, 20 * GIB))
    assert control._safe_count() == 3


def test_switch_only_between_complete_accumulations(controller):
    control, state = controller
    control.before(state, batch())
    control.pending = Decision(2, "explore")
    state.accelerator.sync_gradients = False
    control.after(state, torch.tensor(1.0))
    control.before(state, batch())
    assert control.microsteps == 2
    control.offloader.reconfigure.assert_not_called()
    state.accelerator.sync_gradients = True
    control.after(state, torch.tensor(1.0))
    assert control.microsteps == 0
    control.pending = Decision(2, "explore")
    state.global_step = 1
    control.before(state, batch())
    control.offloader.reconfigure.assert_called_once_with(control.model.blocks, 2)
    assert control.args.blocks_to_swap == 2


def test_accumulation_layout_keeps_multiplicity(controller):
    control, state = controller
    control.before(state, batch(8))
    control.before(state, batch(16))
    control.before(state, batch(8))
    control.after(state, torch.tensor(1.0))
    assert (
        tuple(sorted([(1, 16, 8, 8), (1, 16, 16, 16), (1, 16, 8, 8)]))
        in control.policy.known_shapes
    )


def test_unseen_mid_update_shape_is_not_hot_migrated(controller):
    control, state = controller
    control.policy.current = 2
    control.policy.known_inputs.add((1, 16, 8, 8))
    control.before(state, batch())
    with pytest.raises(RuntimeError, match="uncalibrated shape"):
        control.before(state, batch(16))
    control.offloader.reconfigure.assert_not_called()


def test_pressure_preempts_pending_performance_decision(controller, monkeypatch):
    control, state = controller
    control.policy.current = 2
    control.policy.known_inputs.add((1, 16, 8, 8))
    control.pending = Decision(0, "explore")
    monkeypatch.setattr(control, "_safe_count", lambda **kwargs: 4)
    control.before(state, batch())
    assert control.policy.current == 4 and control.pending is None
    assert control.last_reason == "gpu_pressure"


def test_migration_oom_retains_count_and_sets_expiring_floor(controller):
    control, state = controller
    control.offloader.reconfigure.side_effect = torch.cuda.OutOfMemoryError("injected")
    control._change(Decision(2, "explore"), 5)
    assert control.policy.current == control.model.blocks_to_swap == 4
    assert control.policy.rejected_below == 2 and control.policy.retry_after > 5
    assert control.switches == 0


@pytest.mark.parametrize("loss,skipped", [(float("nan"), False), (1, True)])
def test_invalid_optimizer_updates_abort(controller, loss, skipped):
    control, state = controller
    control.before(state, batch())
    state.accelerator.optimizer_step_was_skipped = skipped
    with pytest.raises(RuntimeError, match="finite, completed"):
        control.after(state, torch.tensor(float(loss)))


def test_nonfinite_accumulation_microstep_aborts_immediately(controller):
    control, state = controller
    control.before(state, batch())
    state.accelerator.sync_gradients = False
    with pytest.raises(RuntimeError, match="finite"):
        control.after(state, torch.tensor(float("nan")))


def test_host_paging_guard_precedes_migration(controller, monkeypatch):
    control, state = controller
    monkeypatch.setattr(
        online,
        "host_memory",
        lambda: HostMemory(64 * GIB, 32 * GIB, 8 * GIB, 0, GIB, 0),
    )
    with pytest.raises(RuntimeError, match="RAM/paging"):
        control.before(state, batch())
    control.offloader.reconfigure.assert_not_called()


def test_budget_counts_allocator_free_once(controller, monkeypatch):
    control, state = controller
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (4 * GIB, 20 * GIB))
    assert control._safe_count() == 2
    assert control._safe_count(promotion=True) == 4


def test_shape_inventory_uses_dataloader_after_session_releases_group(controller, monkeypatch):
    control, state = controller
    dataset = object()
    state.train_dataset_group = None
    state.train_dataloader = SimpleNamespace(dataset=dataset)
    cases = Mock(return_value=[{"batch_size": 1, "height": 128, "width": 64}])
    monkeypatch.setattr(online, "representative_batches", cases)
    control = online.OnlineSwapController(control.trainer, state)
    cases.assert_called_once_with(dataset)
    assert control.required_shapes == {(1, 16, 8)}
    assert control._safe_count(promotion=True) == control.maximum
    control.observed_shapes.add((1, 16, 8))
    assert control._safe_count(promotion=True) == 0


def test_runtime_report_failure_is_nonfatal(controller, monkeypatch):
    control, state = controller
    control.path = control.path / "not-a-directory"
    control._record({"event": "update"})


def test_dynamic_inventory_does_not_run_search(tmp_path):
    from library.training.auto_block_swap.online_startup import initialize

    args = SimpleNamespace(auto_block_swap_timeout=60)
    runner = Mock(
        return_value={
            "status": "inventory",
            "model_swap_limit": 26,
            "host_swap_limit": 24,
            "block_bytes": [GIB] * 28,
        }
    )
    args.seed = 42
    initialize(args, tmp_path, runner=runner)
    assert runner.call_count == 1
    assert args._auto_swap_resolved and args.blocks_to_swap == 24


def test_policy_revisits_higher_swap_counts_when_faster():
    policy = OnlineSwapPolicy(26, interval=4, warmup=0)
    policy.current = 20
    policy.known_inputs = {"shape"}
    decisions = []
    for step in range(1, 53):
        timing = 12 if policy.current == 18 else 8 if policy.current == 22 else 10
        decision = policy.observe(
            "shape", timing, step=step, remaining=1000, promotion_floor=0
        )
        if decision:
            decisions.append(decision)
            policy.changed(decision.blocks)
    assert Decision(22, "explore") in decisions
