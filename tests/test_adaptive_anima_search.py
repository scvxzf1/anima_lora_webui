"""CPU-only search and monitor contracts; no CUDA worker launches."""

import json
import sys
import subprocess
from types import SimpleNamespace

import pytest

from bench.adaptive_runtime import probe_monitor as monitor
from bench.adaptive_runtime import search_anima_precision as search
from library.training.auto_block_swap.resources import GIB, HostMemory


def args(tmp_path, **changes):
    weights, inputs = tmp_path / "weights", tmp_path / "inputs"
    weights.write_bytes(b"w")
    inputs.write_bytes(b"i")
    return SimpleNamespace(**{**dict(weights=weights, inputs=inputs, output_dir=tmp_path / "search",
                                    gpu_uuid="GPU-test", swap=2, steps=3, worker_timeout=10,
                                    max_attempts=3, loss_scale=1024, profile=None), **changes})


def gpu(_):
    return {"uuid": "GPU-test", "compute_cap": "7.0", "name": "cpu-test"}


def memory():
    return HostMemory(16 * GIB, 12 * GIB, 0, 0, 0, 0)


def test_profiles_follow_real_capability_and_explicit_override():
    profiles, choice = search.profiles_for(None, (7, 0), 1024)
    assert choice == "fp16"
    assert [p.name for p in profiles] == ["seed", "conditioning", "conditioning-attention"]
    assert profiles[0].fp32_patterns == ()
    assert search.profiles_for(None, (8, 0), 1024)[0][0].name == "bf16"
    assert search.profiles_for(None, (5, 0), 1024)[0] == []
    assert search.profiles_for(["seed"], (7, 0), 1024)[0][0].precision == "fp16-islands"
    with pytest.raises(ValueError, match="Ampere"):
        search.profiles_for(["bf16"], (7, 0), 1024)
    with pytest.raises(ValueError, match="duplicate"):
        search.profiles_for(["seed", "seed"], (7, 0), 1024)


def test_conditioning_targets_actual_patch_projection():
    from fnmatch import fnmatchcase

    profiles, _ = search.profiles_for(["conditioning"], (8, 6), 1024)
    assert any(fnmatchcase("x_embedder.proj.1", p) for p in profiles[0].fp32_patterns)


def test_reference_failure_stops_search_and_preserves_log(tmp_path):
    called = []

    def fake_worker(argv, directory, uuid, timeout):
        called.append(directory.name)
        (directory / "worker.log").write_text("failed to load\n")
        return {"status": "error", "stage": "model_load", "returncode": 1}

    cfg = args(tmp_path)
    result = search.run(cfg, gpu_check=gpu, monitor=fake_worker)
    assert result["status"] == "reference_failed" and called == ["reference"]
    assert (cfg.output_dir / "reference" / "worker.log").read_text() == "failed to load\n"
    assert not (cfg.output_dir / "selected-plan.json").exists()
    with pytest.raises(FileExistsError):
        search.run(cfg, gpu_check=gpu, monitor=fake_worker)


def test_candidate_refuses_gpu_claimed_after_reference(tmp_path, monkeypatch):
    cfg = args(tmp_path, profile=["seed"])
    monkeypatch.setattr(search, "_complete", lambda *a, **kw: True)
    checks = 0
    launched = []

    def idle(uuid):
        nonlocal checks
        checks += 1
        if checks == 3:
            raise RuntimeError("Selected GPU has an existing compute process")
        return gpu(uuid)

    def worker(argv, directory, uuid, timeout):
        launched.append(directory.name)
        return {"status": "ok"}

    result = search.run(cfg, gpu_check=idle, monitor=worker)
    assert launched == ["reference"] and result["status"] == "failed"
    assert result["attempts"][0]["result"]["status"] == "gpu_unavailable"
    assert (cfg.output_dir / "attempt-000" / "worker.log").exists()


def test_bounded_search_writes_single_final_candidate_record(tmp_path, monkeypatch):
    cfg = args(tmp_path, profile=["seed"])
    monkeypatch.setattr(search, "_complete", lambda *a, **kw: True)

    def fake_worker(argv, directory, uuid, timeout):
        (directory / "worker.log").write_text("ok\n")
        (directory / "telemetry.jsonl").write_text("{}\n")
        if directory.name == "reference":
            return {"status": "ok", "returncode": 0}
        return {"status": "ok", "returncode": 0, "disposable_probe": True,
                "precision_profile_resolved": True, "precision": "fp16-islands",
                "loss_scale": 1024, "swap": 2, "fp32_modules": [],
                "precision_manifest": {"assignments": {"a": "fp16", "b": "fp16"}},
                "monitor": {"samples": 1}}

    def compare(*_):
        return {"status": "compared", "scope": "fixed_pre_step_state_and_inputs_not_quality_certificate",
                "cases": [{"within_experimental_tolerances": True}],
                "within_experimental_tolerances": True}

    result = search.run(cfg, gpu_check=gpu, monitor=fake_worker, comparator=compare)
    assert result["status"] == "probe_validated"
    lines = (cfg.output_dir / "candidates.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["comparison"]["status"] == "compared"
    plan = json.loads((cfg.output_dir / "selected-plan.json").read_text())
    assert plan["fp16_linear_fraction"] == 1 and not plan["production_ready"]
    assert plan["fp16_compute_fraction"] is None and not plan["performance_validated"]


def test_monitor_preserves_structured_failure_and_rejects_partial_success(tmp_path):
    directory = tmp_path / "failure"
    directory.mkdir()
    script = ("import json,sys; from pathlib import Path; "
              "Path(sys.argv[1]).write_text(json.dumps({'status':'error','failure_kind':'nonfinite',"
              "'stage':'forward','error_type':'FloatingPointError'})); print('worker detail'); sys.exit(1)")
    result = monitor.run_monitored([sys.executable, "-c", script, str(directory / "result.json")],
                                   directory, "GPU-test", 5, resources=memory,
                                   gpu=lambda _: {"memory.used": 123}, interval=0.01)
    assert result["status"] == "error" and result["failure_kind"] == "nonfinite"
    assert result["stage"] == "forward" and result["returncode"] == 1
    assert "worker detail" in (directory / "worker.log").read_text()
    assert (directory / "telemetry.jsonl").read_text().strip()

    directory = tmp_path / "partial"
    directory.mkdir()
    script = "import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({'status':'running'}))"
    result = monitor.run_monitored([sys.executable, "-c", script, str(directory / "result.json")],
                                   directory, "GPU-test", 5, resources=memory,
                                   gpu=lambda _: {"memory.used": 0}, interval=0.01)
    assert result["status"] == "error" and result["stop_reason"] == "incomplete_worker_result"


def test_monitor_gpu_failure_is_fatal_before_worker(tmp_path):
    directory = tmp_path / "gpu"
    directory.mkdir()
    result = monitor.run_monitored([sys.executable, "-c", "raise RuntimeError('started')"], directory,
                                   "GPU-test", 5, resources=memory,
                                   gpu=lambda _: (_ for _ in ()).throw(RuntimeError("smi failed")))
    assert result["status"] == "telemetry_error" and result["returncode"] is None
    assert "smi failed" in (directory / "worker.log").read_text()


def test_monitor_keeps_cuda_oom_with_nonzero_exit(tmp_path):
    directory = tmp_path / "oom"
    directory.mkdir()
    script = ("import json,sys; from pathlib import Path; "
              "Path(sys.argv[1]).write_text(json.dumps({'status':'cuda_oom','stage':'backward',"
              "'error_type':'OutOfMemoryError'})); sys.exit(1)")
    result = monitor.run_monitored([sys.executable, "-c", script, str(directory / "result.json")],
                                   directory, "GPU-test", 5, resources=memory,
                                   gpu=lambda _: {"memory.used": 5}, interval=0.01)
    assert result["status"] == "cuda_oom" and result["stage"] == "backward"
    assert result["error_type"] == "OutOfMemoryError" and result["returncode"] == 1


@pytest.mark.parametrize("persistent", [False, True])
def test_monitor_records_missing_samples_and_bounds_failure_gap(tmp_path, persistent):
    directory = tmp_path / "sample-gap"
    directory.mkdir()
    calls = 0

    def sample(_):
        nonlocal calls
        calls += 1
        if calls > 1 and (persistent or calls == 2):
            raise subprocess.TimeoutExpired("nvidia-smi", 5)
        return {"memory.used": 5}

    script = ("import json,sys,time; from pathlib import Path; time.sleep(0.15); "
              "Path(sys.argv[1]).write_text(json.dumps({'status':'ok'}))")
    result = monitor.run_monitored([sys.executable, "-c", script, str(directory / "result.json")],
                                   directory, "GPU-test", 5, resources=memory,
                                   gpu=sample, interval=0.01)
    rows = [json.loads(line) for line in (directory / "telemetry.jsonl").read_text().splitlines()]
    assert any(row["gpu"] is None and row["gpu_telemetry_error"] for row in rows)
    assert result["monitor"]["telemetry_failures"] == (3 if persistent else 1)
    assert result["status"] == ("telemetry_error" if persistent else "ok")
