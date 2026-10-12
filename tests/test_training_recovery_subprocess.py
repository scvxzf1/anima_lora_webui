"""Production recovery protocol in fresh CPU workers; CUDA OOM is injected."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from library.training.adaptive_runtime.process import IsolatedRunner
from library.training.adaptive_runtime.retry import MemoryPlan, RetryLimits, run_recovery
from library.training.adaptive_runtime.resume_rng import capture_rng
from library.training.adaptive_runtime.resume_runtime import commit, restore
from library.training.adaptive_runtime.training_worker import _native_cuda_oom
from library.training.auto_block_swap.process import write_result
from tests.test_training_recovery_checkpoint import setup
from tests.test_training_recovery_cursor import train_steps
from tests.test_training_recovery_integration import Layout, runtime


class FakeScaler:
    """Exercise Accelerate's scaler persistence without claiming CPU FP16."""
    def __init__(self):
        self.scale = 1024.0
        self.growth_tracker = 0

    def state_dict(self):
        return {"scale": self.scale, "_growth_tracker": self.growth_tracker}

    def load_state_dict(self, state):
        self.scale = state["scale"]
        self.growth_tracker = state["_growth_tracker"]


def worker(request, *, cut=5, final=18, baseline=False, incomplete=False):
    request = Path(request)
    data = json.loads(request.read_text())
    directory = request.parent
    state, model, optimizer, scheduler = setup()
    scaler = FakeScaler()
    state.accelerator.scaler = scaler
    rt = runtime(directory)
    # Native precision guards are fixture metadata, not CPU FP16 certification.
    from library.training.adaptive_runtime.native_manifest import SCHEMA, fingerprint
    rt.args.adaptive_precision = "native_fp16"
    rt._base = {"parameter:fixture": {"dtype": "torch.float16", "shape": [1], "numel": 1}}
    rt._execution = {"blocks": 28, "fp32_residual": True, "native_amp": True,
                     "scaler_policy": {"growth_factor": 2.0, "backoff_factor": 0.5,
                                       "growth_interval": 2000},
                     "trainables": {"weight": {"dtype": "torch.float32", "shape": [1, 3], "numel": 3}},
                     "lora_fp32_compute": [False]}
    rt.precision_manifest_id = fingerprint({"schema": SCHEMA, "base": rt._base})
    trace = []
    layout = Layout()
    if data["resume"]:
        args = SimpleNamespace(adaptive_oom_retry_resume=True,
                               _adaptive_training_worker={"directory": str(directory)},
                               _adaptive_training_resume={"path": data["resume"],
                                                          "owner_root": str(directory.parent)})
        # Intentionally differing config values must be overwritten by saved scaler state.
        scaler.scale, scaler.growth_tracker = 8.0, 99
        plan = restore(SimpleNamespace(_adaptive_training_runtime=rt), args, state.accelerator,
                       state.train_dataloader, layout, model, optimizer)
        state.global_step = plan.global_step
        assert layout.events == [("move", "cpu"), ("reset", False)]
        assert scaler.state_dict() == {"scale": 512.0, "_growth_tracker": cut}
    iterator = None
    while state.global_step < final:
        batch_trace, iterator = train_steps(state.train_dataloader, model, optimizer, scheduler,
                                            1, iterator=iterator)
        trace.extend(batch_trace)
        state.global_step += 1
        rt.attempted_steps = rt.effective_steps = state.global_step
        scaler.scale = 512.0
        scaler.growth_tracker = state.global_step
        if state.global_step == cut and not baseline and not data["resume"]:
            path = commit(rt, state)
            if incomplete:
                (path / "scaler.pt").unlink()
            # Consume the next prefetched input, then partially mutate AdamW/model.
            train_steps(state.train_dataloader, model, optimizer, scheduler, 1, iterator=iterator)
            with torch.no_grad():
                next(model.parameters()).add_(123)
            next(iter(optimizer.state.values()))["exp_avg"].add_(456)
            exc = torch.OutOfMemoryError("CUDA out of memory: synthetic allocation failure")
            assert _native_cuda_oom(exc, rt)
            write_result(directory / "result.json", {
                "status": "cuda_oom", "stage": "backward", "optimizer_started": True,
                "committed_checkpoint": str(path), "precision_contract_id": "native",
                "precision_manifest_id": rt.precision_manifest_id, "pid": os.getpid(), "injected": True})
            return
    # Include next input across an epoch boundary and all training RNG streams.
    next_trace, _ = train_steps(state.train_dataloader, model, optimizer, scheduler, 1, iterator=iterator)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "rng": capture_rng(), "trace": trace, "next": next_trace,
                "layout": layout.events}, directory / "final.pt")
    write_result(directory / "result.json", {"status": "ok", "pid": os.getpid(),
                 "precision_contract_id": "native", "precision_manifest_id": rt.precision_manifest_id})


def command(request, cut, baseline=False, incomplete=False):
    script = ("from tests.test_training_recovery_subprocess import worker; "
              f"worker({str(request)!r}, cut={cut}, baseline={baseline!r}, incomplete={incomplete!r})")
    return [sys.executable, "-c", script]


@pytest.mark.parametrize("cut", [5, 12])
def test_fresh_pid_committed_recovery_exact_trajectory(tmp_path, cut):
    baseline = IsolatedRunner(tmp_path / "baseline", lambda p: command(p, cut, True), timeout=40,
                              env={"CUDA_VISIBLE_DEVICES": ""})
    assert baseline(MemoryPlan(), attempt=0, resume=None)["status"] == "ok"
    isolated = IsolatedRunner(tmp_path / "recovery", lambda p: command(p, cut), timeout=40,
                              env={"CUDA_VISIBLE_DEVICES": ""})
    report = run_recovery(MemoryPlan(), RetryLimits(4), isolated, record=lambda value: None,
                          expected_precision_contract="native")
    assert report["status"] == "ok"
    attempts = report["attempts"]
    assert len(attempts) == 2
    assert attempts[0]["result"]["pid"] != attempts[1]["result"]["pid"]
    assert [a["plan"]["blocks_to_swap"] for a in attempts] == [0, 2]
    expected = torch.load(tmp_path / "baseline/attempt-000/final.pt", weights_only=False)
    actual = torch.load(tmp_path / "recovery/attempt-001/final.pt", weights_only=False)
    for name in ("model", "optimizer", "scheduler", "scaler", "rng", "next"):
        assert_exact(actual[name], expected[name])
    assert [row["key"] for row in actual["trace"]] == [row["key"] for row in expected["trace"][cut:]]
    for a, b in zip(actual["trace"], expected["trace"][cut:]):
        assert_exact(a, b)


@pytest.mark.parametrize("message", ["CPU out of memory", "DefaultCPUAllocator allocation failed",
                                    "host out of memory", "out of memory"])
def test_typed_host_or_ambiguous_oom_does_not_retry(message):
    assert not _native_cuda_oom(torch.OutOfMemoryError(message), None)


def assert_exact(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_exact(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_exact(a, b)
    else:
        assert actual == expected


def test_incomplete_snapshot_stops_fresh_worker_recovery(tmp_path):
    isolated = IsolatedRunner(tmp_path, lambda p: command(p, 1, incomplete=True), timeout=40,
                              env={"CUDA_VISIBLE_DEVICES": ""})
    report = run_recovery(MemoryPlan(), RetryLimits(6), isolated, record=lambda value: None,
                          expected_precision_contract="native")
    assert report["status"] == "failed"
    assert len(report["attempts"]) == 2
    assert report["attempts"][1]["result"]["status"] == "error"
    assert report["attempts"][1]["result"]["returncode"] != 0
