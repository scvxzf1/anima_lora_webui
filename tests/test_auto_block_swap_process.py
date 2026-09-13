from argparse import Namespace
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace as NS

import pytest
import torch

from library.training.auto_block_swap import process, worker
from library.training.auto_block_swap.resources import GIB, HostMemory


def memory(available=8 * GIB):
    return HostMemory(16 * GIB, available, 128 * GIB, 0, 0, 0)


def test_worker_distinguishes_cuda_oom_from_other_errors(tmp_path, monkeypatch):
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps({"auto_block_swap": False, "_auto_swap_probe": {"inventory": False}})
    )

    class Trainer:
        error = None

        def train(self, args):
            raise self.error

    monkeypatch.setitem(sys.modules, "train", NS(AnimaTrainer=Trainer))
    monkeypatch.setattr(
        "library.training.train_bootstrap.install_stop_signal_handlers", lambda: None
    )
    for error, status in [
        (torch.cuda.OutOfMemoryError("CUDA out of memory"), "cuda_oom"),
        (RuntimeError("CPU out of memory"), "error"),
        (RuntimeError("mat2 is on cpu"), "error"),
        (
            process.ProbeComplete({"status": "inventory", "block_count": 30}),
            "inventory",
        ),
    ]:
        Trainer.error = error
        worker.main(request)
        assert json.loads((tmp_path / "result.json").read_text())["status"] == status


def test_timeout_joins_only_owned_process_and_removes_request(tmp_path, monkeypatch):
    original = subprocess.Popen
    children = []

    def popen(command, **kwargs):
        child = original(
            [sys.executable, "-c", "import time; time.sleep(30)"], **kwargs
        )
        children.append(child)
        assert kwargs["env"]["WANDB_MODE"] == "disabled"
        return child

    monkeypatch.setattr(process.subprocess, "Popen", popen)
    monkeypatch.setattr(process, "host_memory", memory)
    directory = tmp_path / "trial"
    result = process.run_process(Namespace(value=1), directory, timeout=1)
    assert result["status"] == "timeout"
    assert children[0].poll() is not None
    assert not (directory / "request.json").exists()
    assert (directory / "worker.log").exists()


def test_ram_guard_does_not_add_virtual_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(process, "host_memory", lambda: memory(available=GIB))
    monkeypatch.setattr(
        process.subprocess,
        "Popen",
        lambda *a, **k: pytest.fail("Must reject before launching"),
    )
    directory = tmp_path / "trial"
    result = process.run_process(Namespace(), directory, timeout=10)
    assert result["status"] == "host_limit"
    assert not (directory / "request.json").exists()


def test_success_requires_exit_code_and_structured_result(tmp_path, monkeypatch):
    original = subprocess.Popen

    def popen(command, **kwargs):
        result = Path(command[-1]).parent / "result.json"
        program = f'from pathlib import Path; Path({str(result)!r}).write_text(\'{{"status":"ok"}}\')'
        return original([sys.executable, "-c", program], **kwargs)

    monkeypatch.setattr(process.subprocess, "Popen", popen)
    monkeypatch.setattr(process, "host_memory", memory)
    result = process.run_process(Namespace(), tmp_path / "trial", timeout=10)
    assert result["status"] == "ok"
    assert result["host_before"]["swap_total"] == 128 * GIB
    assert result["host_min_available"] == 8 * GIB
    assert result["elapsed_seconds"] > 0


def test_interrupt_always_joins_child(tmp_path, monkeypatch):
    original = subprocess.Popen
    children = []

    def popen(command, **kwargs):
        child = original(
            [sys.executable, "-c", "import time; time.sleep(30)"], **kwargs
        )
        children.append(child)
        return child

    monkeypatch.setattr(process.subprocess, "Popen", popen)
    monkeypatch.setattr(process, "host_memory", memory)
    monkeypatch.setattr(
        time, "sleep", lambda _: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    with pytest.raises(KeyboardInterrupt):
        process.run_process(Namespace(), tmp_path / "trial", timeout=10)
    assert children[0].poll() is not None
    assert not (tmp_path / "trial" / "request.json").exists()
