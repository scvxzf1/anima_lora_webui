"""CPU orchestration and block identity gates for the bounded upstream batch."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.adaptive_runtime import anima_block_vjp as vjp
from bench.adaptive_runtime import diagnose_anima_upstream_vjp as batch


def _args(tmp_path):
    weights, inputs = tmp_path / "weights", tmp_path / "inputs"
    weights.write_bytes(b"weights")
    inputs.write_bytes(b"inputs")
    return SimpleNamespace(weights=weights, inputs=inputs, output_dir=tmp_path / "new-run",
                           gpu_uuid="test-gpu", swap=20, worker_timeout=150, analysis_timeout=300)


def test_capture_command_records_once_then_replays_exact_mlp_plan(tmp_path):
    args = _args(tmp_path)
    reference = args.output_dir / "reference"
    first = batch.capture_command(args, reference)
    second = batch.capture_command(args, args.output_dir / "candidate", reference)
    assert first.count("--capture-block-vjp") == second.count("--capture-block-vjp") == 1
    assert first[first.index("--block-vjp-indices") + 1] == "15,17,27"
    assert "--record-replay" in first and "--replay-reference" in second
    assert "--fp32-pattern" not in second
    assert second[second.index("--fp16-only-pattern") + 1] == "blocks.*.mlp.*"
    assert first[first.index("--loss-scale") + 1] == "1"
    assert second[second.index("--loss-scale") + 1] == "1024"


@pytest.mark.parametrize("index", (15, 17, 27))
def test_block_index_selects_its_training_root_and_analysis_paths(tmp_path, index):
    args = _args(tmp_path)
    child = args.output_dir / "reference/block-vjp" / f"block-{index:02d}"
    assert vjp.training_root(child, index) == args.output_dir / "reference"
    command = batch.analysis_command(args, args.output_dir / f"analysis-block-{index}", index)
    assert command[command.index("--block-index") + 1] == str(index)
    assert command[command.index("--reference") + 1] == str(child)
    with pytest.raises(ValueError, match="path/index mismatch"):
        vjp.training_root(args.output_dir / "reference/block-vjp/block-14", index)


def _mock_batch(monkeypatch, args, *, fail_index=None):
    commands, validated = [], []

    def launch(command, directory, limits, *_):
        commands.append((command, limits.worker_timeout))
        if "--analyze" in command:
            folder = directory / "matrix-evidence"
            folder.mkdir()
            (folder / "matrix.safetensors").write_bytes(b"bounded-matrix")
        return {"status": "ok", "returncode": 0}

    def validate_pair(reference, candidate, comparison, *, block_index):
        validated.append(block_index)
        assert reference.name == candidate.name == f"block-{block_index:02d}"
        assert reference.parent.parent == args.output_dir / "reference"
        assert comparison["status"] == "compared"

    def read_analysis(directory, result, environment, index, root):
        if index == fail_index:
            raise ValueError("incomplete analysis")
        return {"block_index": index, "scope": vjp.UPSTREAM_SCOPE, "production_ready": False}

    monkeypatch.setattr(batch, "launch_if_idle", launch)
    monkeypatch.setattr(batch, "_read_worker", lambda *_, **_kw: {})
    monkeypatch.setattr(batch, "_read_analysis", read_analysis)
    monkeypatch.setattr(vjp, "validate_pair", validate_pair)
    return commands, validated


def test_batch_runs_two_captures_and_three_sequential_analyses(monkeypatch, tmp_path):
    args = _args(tmp_path)
    commands, validated = _mock_batch(monkeypatch, args)
    result = batch.run(args, gpu_check=lambda _: {}, comparator=lambda *_: {"status": "compared"})
    assert result["status"] == "completed"
    assert validated == [15, 17, 27]
    assert [limit for _, limit in commands] == [150, 150, 300, 300, 300]
    assert set(result["analyses"]) == {"15", "17", "27"}
    assert result["matrix_disk_bytes"] == 3 * len(b"bounded-matrix")
    assert result["production_ready"] is False


def test_any_missing_analysis_leaves_batch_failed(monkeypatch, tmp_path):
    args = _args(tmp_path)
    commands, _ = _mock_batch(monkeypatch, args, fail_index=17)
    result = batch.run(args, gpu_check=lambda _: {}, comparator=lambda *_: {"status": "compared"})
    assert result["status"] == "failed" and set(result["analyses"]) == {"15"}
    assert len(commands) == 4
    assert "incomplete analysis" in result["stop_reason"]


def test_aggregate_matrix_budget_fails_batch(monkeypatch, tmp_path):
    args = _args(tmp_path)
    _mock_batch(monkeypatch, args)
    monkeypatch.setattr(batch, "MATRIX_TOTAL_MAX", 1)
    with pytest.raises(ValueError, match="Planned matrices"):
        batch.run(args, gpu_check=lambda _: pytest.fail("GPU idle check before budget validation"))
    assert not args.output_dir.exists()


def test_analysis_rejects_cross_block_report_before_matrix_reader(monkeypatch, tmp_path):
    directory = tmp_path / "analysis"
    directory.mkdir()
    environment = {key: key for key in ("gpu_uuid", "analysis_driver_sha256", "analysis_sha256",
                                        "weights_sha256", "inputs_sha256", "probe_sha256", "helper_sha256")}
    source = {key: environment[key] for key in ("weights_sha256", "inputs_sha256", "probe_sha256")}
    source["block_vjp_helper_sha256"] = environment["helper_sha256"]
    report = {"status": "ok", "uuid": environment["gpu_uuid"],
              "driver_sha256": environment["analysis_driver_sha256"],
              "analysis_sha256": environment["analysis_sha256"],
              "analysis": {"uuid": environment["gpu_uuid"], "status": "analyzed", "block_index": 27,
                           "scope": vjp.UPSTREAM_SCOPE, "cell_count": 8, "tokens": 4200,
                           "production_ready": False, "source": source}}
    (directory / "result.json").write_text(json.dumps(report))
    monkeypatch.setattr(batch.single, "validate_matrix_evidence", lambda *_a, **_kw: pytest.fail("wrong block"))
    with pytest.raises(ValueError, match="identity/source mismatch"):
        batch._read_analysis(directory, {"status": "ok", "returncode": 0}, environment, 15, Path(tmp_path))


def test_existing_directory_is_never_overwritten(tmp_path):
    args = _args(tmp_path)
    args.output_dir.mkdir()
    sentinel = args.output_dir / "user-data"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):
        batch.run(args)
    assert sentinel.read_text() == "preserve"
