import json
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil
import pytest

from library.training.adaptive_runtime.process import IsolatedRunner
from library.training.adaptive_runtime.retry import MemoryPlan, RetryLimits, run_recovery


def resources():
    return SimpleNamespace(available=100, reserve=10)


def test_real_subprocess_retry_and_distinct_pids(tmp_path):
    script = """
import json, os, pathlib, sys
p = pathlib.Path(sys.argv[1])
request = json.loads(p.read_text())
result = {'status': 'cuda_oom' if request['plan']['blocks_to_swap'] == 0 else 'ok',
          'stage': 'forward', 'pid': os.getpid(), 'optimizer_started': False}
(p.parent / 'result.json').write_text(json.dumps(result))
"""
    runner = IsolatedRunner(tmp_path, lambda p: [sys.executable, "-c", script, str(p)],
                            resources=resources)
    report = run_recovery(MemoryPlan(), RetryLimits(4), runner, record=lambda value: None)
    assert report["status"] == "ok"
    assert len({a["result"]["pid"] for a in report["attempts"]}) == 2
    assert not list(tmp_path.rglob("request.json"))
    assert len(list(tmp_path.rglob("supervisor.json"))) == 2


def test_worker_output_is_teeed_to_parent_stdout_and_worker_log(tmp_path, capsys):
    script = (
        "import pathlib, sys; "
        "print('worker-stdout-sentinel', flush=True); "
        "print('worker-stderr-sentinel', file=sys.stderr, flush=True); "
        "pathlib.Path(sys.argv[1]).with_name('result.json').write_text('{\"status\": \"ok\"}')"
    )
    runner = IsolatedRunner(
        tmp_path,
        lambda p: [sys.executable, "-c", script, str(p)],
        resources=resources,
    )

    assert runner(MemoryPlan(), attempt=0, resume=None)["status"] == "ok"

    captured = capsys.readouterr().out
    log = (tmp_path / "attempt-000" / "worker.log").read_text(encoding="utf-8")
    assert "worker-stdout-sentinel" in captured
    assert "worker-stderr-sentinel" in captured
    assert "worker-stdout-sentinel" in log
    assert "worker-stderr-sentinel" in log


def test_timeout_is_not_oom(tmp_path):
    runner = IsolatedRunner(tmp_path, lambda p: [sys.executable, "-c", "import time; time.sleep(10)"],
                            resources=resources, timeout=0.1)
    assert runner(MemoryPlan(), attempt=0, resume=None)["status"] == "timeout"


def test_process_failure_is_not_success(tmp_path):
    runner = IsolatedRunner(tmp_path, lambda p: [sys.executable, "-c", "raise SystemExit(2)"],
                            resources=resources)
    result = runner(MemoryPlan(), attempt=0, resume=None)
    assert result["status"] == "error" and result["returncode"] == 2


def test_host_pressure_prevents_launch(tmp_path):
    runner = IsolatedRunner(tmp_path, lambda p: ["nonexistent-command"],
                            resources=lambda: SimpleNamespace(available=1, reserve=10))
    assert runner(MemoryPlan(), attempt=0, resume=None)["status"] == "host_limit"


@pytest.mark.skipif(os.name != "posix", reason="POSIX process signal test")
def test_supervisor_sigterm_reaps_worker_and_records_cancellation(tmp_path):
    worker = (
        "import os, pathlib, sys, time; "
        "pathlib.Path(sys.argv[1]).with_name('worker.pid').write_text(str(os.getpid())); "
        "time.sleep(60)"
    )
    script = f"""
import pathlib, sys
from types import SimpleNamespace
from library.training.adaptive_runtime.process import IsolatedRunner
from library.training.adaptive_runtime.retry import MemoryPlan
runner = IsolatedRunner(pathlib.Path(sys.argv[1]),
    lambda p: [sys.executable, '-c', {worker!r}, str(p)],
    resources=lambda: SimpleNamespace(available=100, reserve=10))
result = runner(MemoryPlan(), attempt=0, resume=None)
assert result['status'] == 'cancelled'
"""
    process = subprocess.Popen([sys.executable, "-c", script, str(tmp_path)])
    child_pid = None
    try:
        deadline = time.monotonic() + 30
        pid_file = tmp_path / "attempt-000" / "worker.pid"
        while not pid_file.exists() and time.monotonic() < deadline:
            assert process.poll() is None
            time.sleep(0.05)
        child_pid = int(pid_file.read_text())
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=15) == 0
        report = json.loads((pid_file.parent / "supervisor.json").read_text())
        assert report["status"] == "cancelled" and report["signal"] == signal.SIGTERM
        assert not psutil.pid_exists(child_pid)
        assert not (pid_file.parent / "request.json").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        if child_pid is not None and psutil.pid_exists(child_pid):
            os.kill(child_pid, signal.SIGKILL)
