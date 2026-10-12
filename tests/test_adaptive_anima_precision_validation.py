"""CPU orchestration contracts for the fixed Batch 2 Anima validation."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.adaptive_runtime import anima_precision_validation as batch


def fixture_args(tmp_path, **overrides):
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(b"weights")
    warmup = tmp_path / "old-inputs.safetensors"
    warmup.write_bytes(b"warmup")
    warmup.with_suffix(".json").write_text("{}")
    rows = []
    for role, count in (("representative", 4032), ("heldout", 4200)):
        path = tmp_path / f"{role}-{count}-inputs.safetensors"
        path.write_bytes(role.encode())
        path.with_suffix(".json").write_text("{}")
        rows.append({"role": role, "anima_token_count": count, "stem": role,
                     "inputs_path": str(path), "inputs_sha256": batch.sha256_file(path)})
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"schema": "adaptive_anima_batch2_selection_v1",
                                     "status": "ready_for_probe", "model_family": "anima",
                                     "batch1_excluded_stem": "old", "selection": rows}))
    values = dict(weights=weights, warmup_inputs=warmup, selection=selection,
                  output_dir=tmp_path / "output", gpu_uuid="GPU-test", swap=20,
                  loss_scale=1024, worker_timeout=240, batch_timeout=480,
                  max_new_workers=2, continue_run=False)
    return SimpleNamespace(**(values | overrides))


def _numeric(passes):
    return {"status": "compared", "cases": [
        {"within_experimental_tolerances": passes, "prediction": {"finite": True},
         "input_gradient": {"finite": True}, "adapter_gradient": {"finite": True}}
        for _ in batch.updates.CASES]}


def _deltas():
    return {"status": "compared", "cases": [
        {"state_label": label, "sigma": sigma, "global": {"absolute_l2": 0},
         "groups": {}} for label, sigma in batch.updates.CASES]}


def fake_backends(monkeypatch, *, failed=frozenset()):
    launched = []

    def monitor(argv, directory, uuid, timeout):
        launched.append(directory.name)
        assert timeout <= 240 and uuid == "GPU-test"
        result = {"status": "ok", "returncode": 0, "uuid": uuid, "steps": 1 if
                  directory.name == "warmup" else 6, "disposable_probe": True,
                  "updates": [{"step": i + 1, "effective_update": True,
                               "compute_seconds": 1, "capture_io_seconds": 2,
                               "seconds": 3, "peak_allocated": 4,
                               "peak_reserved": 5} for i in range(1 if
                  directory.name == "warmup" else 6)],
                  "monitor": {"elapsed_seconds": 7},
                  "precision_manifest": {"assignments": {
                      "other": "fp32", **{p: "fp16" for name, patterns in batch.PATTERNS.items()
                                          if directory.name.endswith(name) for p in patterns}}}}
        (directory / "result.json").write_text(json.dumps(result))
        (directory / "supervisor.json").write_text(json.dumps(result))
        (directory / "worker.log").write_text("ok")
        return result

    monkeypatch.setattr(batch.search, "_complete", lambda result, *_a, **_k:
                        result["status"] == "ok" and result["returncode"] == 0)
    monkeypatch.setattr(batch.warmup_updates, "_read", lambda *_a: None)
    monkeypatch.setattr(batch.updates, "_read", lambda *_a: None)
    monkeypatch.setattr(batch, "_bundle", lambda *_a: None)
    monkeypatch.setattr(batch, "validate_warmup_bundle", lambda *_a: None)
    monkeypatch.setattr(batch, "validate_validation_sources", lambda *_a: None)

    def compare(_reference, candidate):
        name = Path(candidate).parent.name
        return _numeric(name not in failed)

    return launched, monitor, compare, lambda *_a: _deltas()


def test_nine_workers_two_per_segment_and_exact_cli(tmp_path, monkeypatch):
    args = fixture_args(tmp_path)
    launched, monitor, compare, delta = fake_backends(monkeypatch)
    selected = batch._selection(args)
    warmup = batch.command(args, args.output_dir / "warmup", None, "warmup", selected)
    assert warmup[warmup.index("--steps") + 1] == "1"
    assert "--export-validation-states" in warmup and "--single-sigma" in warmup
    reference = batch.command(args, args.output_dir / "representative-reference",
                              "representative", "reference", selected)
    assert reference[reference.index("--steps") + 1] == "6"
    assert "--validation-states" in reference and "--single-sigma" not in reference
    control = batch.command(args, args.output_dir / "representative-fp32-control",
                            "representative", "fp32-control", selected)
    assert control[control.index("--fp32-pattern") + 1] == "*"
    for name, patterns in batch.PATTERNS.items():
        cmd = batch.command(args, args.output_dir / name, "heldout", name, selected)
        assert [cmd[i + 1] for i, value in enumerate(cmd) if value == "--fp16-only-pattern"] == list(patterns)
        assert "--fp32-pattern" not in cmd
    for segment in range(5):
        args.continue_run = segment > 0
        result = batch.run(args, gpu_check=lambda _: {"compute_cap": "8.6"},
                           monitor=monitor, comparator=compare, update_comparator=delta)
        assert result["status"] == ("completed" if segment == 4 else "screening")
        assert result["segments"][-1]["new_workers"] <= 2
    assert launched == [directory for _, _, directory in batch.TASKS]
    assert len(result["validation_table"]) == 36
    assert result["decision"] == "eligible_requires_production_training"
    assert set(result["eligible_candidates"]) == set(batch.PATTERNS)
    assert result["production_ready"] is False


@pytest.mark.parametrize("failed,eligible", [
    ({"heldout-candidate-05"}, ["candidate-04"]),
    ({"representative-fp32-control"}, []),
    ({"heldout-candidate-05", "heldout-candidate-04"}, []),
])
def test_all_twelve_cases_and_control_gate(tmp_path, monkeypatch, failed, eligible):
    args = fixture_args(tmp_path)
    _, monitor, compare, delta = fake_backends(monkeypatch, failed=failed)
    for segment in range(5):
        args.continue_run = segment > 0
        result = batch.run(args, gpu_check=lambda _: {}, monitor=monitor,
                           comparator=compare, update_comparator=delta)
    assert result["status"] == "completed"
    assert result["eligible_candidates"] == eligible
    assert result["decision"] == ("eligible_requires_production_training" if eligible
                                   else "no_candidate_passed_heldout_budget")
    assert sum(x["candidate"] == "candidate-05" for x in result["validation_table"]) == 12


def test_continue_rejects_identity_and_evidence_bytes(tmp_path, monkeypatch):
    args = fixture_args(tmp_path)
    launched, monitor, compare, delta = fake_backends(monkeypatch)
    result = batch.run(args, gpu_check=lambda _: {}, monitor=monitor,
                       comparator=compare, update_comparator=delta)
    assert result["status"] == "screening" and launched == ["warmup", "representative-reference"]
    args.continue_run = True
    sidecar = tmp_path / "heldout-4200-inputs.json"
    sidecar.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="identity/status"):
        batch.run(args, gpu_check=lambda _: {}, monitor=monitor,
                  comparator=compare, update_comparator=delta)
    sidecar.write_text("{}")
    (args.output_dir / "warmup" / "worker.log").write_text("tampered")
    with pytest.raises(ValueError, match="evidence changed"):
        batch.run(args, gpu_check=lambda _: {}, monitor=monitor,
                  comparator=compare, update_comparator=delta)
    assert len(launched) == 2


def test_budget_covers_warmup_and_prevents_following_worker(tmp_path, monkeypatch):
    args = fixture_args(tmp_path)
    launched, monitor, compare, delta = fake_backends(monkeypatch)
    monkeypatch.setattr(batch, "LIMIT", 16 * 1024)

    def large_warmup(argv, directory, uuid, timeout):
        result = monitor(argv, directory, uuid, timeout)
        (directory / "padding").write_bytes(b"x" * (17 * 1024))
        return result

    result = batch.run(args, gpu_check=lambda _: {}, monitor=large_warmup,
                       comparator=compare, update_comparator=delta)
    assert result["status"] == "failed" and result["partial_worker"] == "warmup"
    assert launched == ["warmup"]


def test_incomplete_worker_and_assignment_rejected(tmp_path, monkeypatch):
    args = fixture_args(tmp_path)
    launched, monitor, compare, delta = fake_backends(monkeypatch)
    monkeypatch.setattr(batch.search, "_complete", lambda *_a, **_k: False)
    result = batch.run(args, gpu_check=lambda _: {}, monitor=monitor,
                       comparator=compare, update_comparator=delta)
    assert result["status"] == "failed" and "Incomplete" in result["stop_reason"]
    with pytest.raises(ValueError, match="realized FP16"):
        batch._assignments({"precision_manifest": {"assignments": {"other": "fp16"}}},
                           "candidate-05")
