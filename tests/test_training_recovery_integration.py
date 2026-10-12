"""CPU integration of the production commit/restore facade (no GPU claim)."""
from types import SimpleNamespace

import pytest
import torch

from library.training.adaptive_runtime.resume_runtime import commit, restore, configuration_errors
from tests.test_training_recovery_checkpoint import setup
from tests.test_training_recovery_cursor import train_steps


class Layout(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.events = []

    def move_to_device_except_swap_blocks(self, device):
        self.events.append(("move", device.type))

    def prepare_block_swap_before_forward(self, free_cache=True):
        self.events.append(("reset", free_cache))


def runtime(directory):
    args = SimpleNamespace(adaptive_oom_retry_resume=True, adaptive_precision="fp16_fp32")
    return SimpleNamespace(directory=directory, args=args, precision_contract_id="native",
                           precision_manifest_id="p1", completed_steps=0, saved_state=None,
                           attempted_steps=0, effective_steps=0, skipped_steps=0,
                           validate_precision=lambda model: None)


@pytest.mark.parametrize("cut", [5, 12])
def test_committed_facade_restores_partial_optimizer_and_exact_cursor(tmp_path, cut):
    state, model, opt, sched = setup()
    _, iterator = train_steps(state.train_dataloader, model, opt, sched, cut)
    state.global_step = cut
    rt = runtime(tmp_path)
    rt.attempted_steps = rt.effective_steps = cut
    path = commit(rt, state)
    expected, _ = train_steps(state.train_dataloader, model, opt, sched, 18-cut, iterator=iterator)
    resumed, model2, opt2, sched2 = setup()
    # Model a failed optimizer that has already changed only some parameters.
    with torch.no_grad():
        next(model2.parameters()).add_(123)
    args = SimpleNamespace(adaptive_oom_retry_resume=True, _adaptive_training_worker={"directory": str(tmp_path)},
                           _adaptive_training_resume={"path": str(path), "owner_root": str(tmp_path)})
    rt2 = runtime(tmp_path)
    layout = Layout()
    plan = restore(SimpleNamespace(_adaptive_training_runtime=rt2), args, resumed.accelerator,
                   resumed.train_dataloader, layout, model2, opt2)
    assert plan.initial_step == 0 and plan.global_step == cut
    assert rt2.completed_steps == cut and rt2.effective_steps == cut
    assert layout.events == [("move", "cpu"), ("reset", False)]
    actual, _ = train_steps(resumed.train_dataloader, model2, opt2, sched2, 18-cut)
    assert [v["key"] for v in actual] == [v["key"] for v in expected]
    torch.testing.assert_close(model.state_dict(), model2.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(opt.state_dict(), opt2.state_dict(), rtol=0, atol=0)
    assert sched.state_dict() == sched2.state_dict()


def test_opt_in_requires_supervisor_and_rejects_schedule():
    values = {"adaptive_oom_retry_resume": True, "seed": 3, "sample_every_n_steps": 1}
    errors = configuration_errors(values.get)
    assert any("adaptive_oom_retry" in value for value in errors)
    assert any("sample_every_n_steps" in value for value in errors)


def test_cuda_device_alias_uses_current_index(monkeypatch):
    from library.training.adaptive_runtime.resume_runtime import canonical_device
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert canonical_device("cuda") == torch.device("cuda:1")
    assert canonical_device("cuda:0") == torch.device("cuda:0")
    assert canonical_device("cpu:0") == torch.device("cpu")
