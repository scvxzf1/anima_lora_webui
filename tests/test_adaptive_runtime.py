from dataclasses import replace

import pytest
import torch

from library.training.adaptive_runtime.precision import calibrate_unit, preferred_candidate
from library.training.adaptive_runtime.retry import MemoryPlan, RetryLimits, next_plan, run_recovery
from library.training.adaptive_runtime.islands import install_precision_islands


def test_capability_selection():
    assert preferred_candidate((8, 6)) == "bf16"
    assert preferred_candidate((7, 5)) == "fp16"


def test_calibration_preserves_module_and_detects_overflow():
    module = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        module.weight.fill_(1)
    original = module.weight.clone()
    safe = calibrate_unit(module, [((torch.ones(1, 2),), {})], device="cpu")
    assert safe["selected"] == "fp16"
    bad = calibrate_unit(module, [((torch.full((1, 2), 40000.0),), {})], device="cpu")
    assert bad["selected"] == "fp32"
    assert not bad["cases"][0]["output"]["finite"]
    assert module.weight.dtype == torch.float32
    assert torch.equal(module.weight, original)
    assert module.weight.grad is None


def test_reference_failure_and_empty_calibration_rejected():
    module = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="At least one"):
        calibrate_unit(module, [], device="cpu")
    with pytest.raises(ValueError, match="Nonfinite FP32"):
        calibrate_unit(module, [((torch.full((1, 2), float("nan")),), {})], device="cpu")


def test_preserved_constants_and_rng_are_not_modified():
    class Unit(torch.nn.Module):
        def forward(self, x, frequencies):
            assert frequencies.dtype == torch.float32
            assert not frequencies.requires_grad
            return x * frequencies

    state = torch.random.get_rng_state()
    report = calibrate_unit(Unit(), [((torch.ones(2), torch.ones(2)), {})],
                            device="cpu", preserve_input_indices=(1,))
    assert report["selected"] == "fp16"
    assert torch.equal(state, torch.random.get_rng_state())


def test_stage_specific_policy_and_exact_batch():
    plan = MemoryPlan(blocks_to_swap=4, gradient_checkpointing=True, micro_batch=3)
    limits = RetryLimits(max_blocks=4, allow_batch_change=True)
    changed, _ = next_plan(plan, limits, stage="backward")
    assert changed.micro_batch == 1 and changed.accumulation == 3
    assert next_plan(plan, limits, stage="optimizer") is None
    assert next_plan(plan, replace(limits, allow_batch_change=False), stage="forward") is None


def test_recovery_is_bounded_and_does_not_retry_other_errors():
    calls, reports = [], []

    def runner(plan, **kwargs):
        calls.append(plan)
        return {"status": "cuda_oom", "stage": "forward", "optimizer_started": False}

    result = run_recovery(MemoryPlan(), RetryLimits(10, max_attempts=3), runner, record=reports.append)
    assert result["status"] == "failed"
    assert [p.blocks_to_swap for p in calls] == [0, 2, 4]
    result = run_recovery(MemoryPlan(), RetryLimits(10), lambda *a, **k: {"status": "error"},
                          record=reports.append)
    assert len(result["attempts"]) == 1


def test_partial_optimizer_requires_checkpoint():
    result = run_recovery(MemoryPlan(), RetryLimits(10), lambda *a, **k: {
        "status": "cuda_oom", "stage": "backward", "optimizer_started": True,
    }, record=lambda value: None)
    assert result["attempts"][-1]["stop_reason"] == "no_committed_checkpoint"


def test_unknown_optimizer_progress_refuses_recovery():
    result = run_recovery(MemoryPlan(), RetryLimits(10), lambda *a, **k: {
        "status": "cuda_oom", "stage": "forward",
    }, record=lambda value: None)
    assert result["attempts"][-1]["stop_reason"] == "unknown_optimizer_progress"


def test_success_and_resume_propagation():
    calls = []

    def runner(plan, **kwargs):
        calls.append(kwargs)
        return ({"status": "cuda_oom", "stage": "backward", "optimizer_started": True,
                 "committed_checkpoint": "complete-state"} if len(calls) == 1 else {"status": "ok"})

    result = run_recovery(MemoryPlan(), RetryLimits(10), runner, record=lambda value: None)
    assert result["status"] == "ok"
    assert calls[1]["resume"] == "complete-state"


def test_precision_islands_keep_fp32_boundary_and_state_keys():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.SiLU(), torch.nn.Linear(4, 2))
    model.requires_grad_(False)
    keys = list(model.state_dict())
    install_precision_islands(model, {"0": "fp16", "2": "fp32"})
    x = torch.randn(2, 4, requires_grad=True)
    output = model(x)
    output.sum().backward()
    assert model[0].weight.dtype == torch.float16
    assert model[2].weight.dtype == output.dtype == x.grad.dtype == torch.float32
    assert list(model.state_dict()) == keys
    assert torch.isfinite(x.grad).all()


def test_precision_islands_reject_trainable_and_overflow_before_mutation():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    with pytest.raises(ValueError, match="frozen"):
        install_precision_islands(model, {"0": "fp16"})
    model.requires_grad_(False)
    model[1].weight.fill_(70000)
    with pytest.raises(ValueError, match="overflow"):
        install_precision_islands(model, {"0": "fp16", "1": "fp16"})
    assert model[0].weight.dtype == torch.float32
    assert not hasattr(model[0], "_adaptive_original_forward")


def test_precision_island_composes_with_production_lora():
    from networks.lora_modules.lora import LoRAModule

    model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
    model.requires_grad_(False)
    install_precision_islands(model, {"0": "fp16"})
    adapter = LoRAModule("test_island", model[0], lora_dim=2, alpha=2)
    adapter.apply_to()
    adapter.train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
    original = adapter.lora_up.weight.detach().clone()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.randn(2, 4)).square().mean()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in adapter.parameters())
        optimizer.step()
    assert not torch.equal(original, adapter.lora_up.weight)
    assert all(p.dtype == torch.float32 for p in adapter.parameters())
    assert model[0].weight.dtype == torch.float16


def test_finite_trace_records_first_leaf_and_removes_hooks():
    from library.training.adaptive_runtime.finite_trace import FiniteTrace

    model = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())
    report = {}
    trace = FiniteTrace(model, report)
    model(torch.tensor([1.0, float("inf")]))
    assert report["first_nonfinite"]["module"] == "0"
    assert report["first_nonfinite"]["outputs"][0]["max_abs_finite"] == 1.0
    trace.close()
    assert not model[0]._forward_hooks and not model[1]._forward_hooks


def test_conflicting_input_alias_precision_rejected():
    class Unit(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    x = torch.ones(2)
    with pytest.raises(ValueError, match="Aliased input"):
        calibrate_unit(Unit(), [((x, x), {})], device="cpu", preserve_input_indices=(1,))


def test_aliased_precision_modules_are_not_wrapped_twice():
    layer = torch.nn.Linear(2, 2).requires_grad_(False)
    model = torch.nn.ModuleDict({"a": layer, "b": layer})
    with pytest.raises(ValueError, match="Aliased precision"):
        install_precision_islands(model, {"a": "fp16", "b": "fp16"})
    assert not hasattr(layer, "_adaptive_original_forward")
