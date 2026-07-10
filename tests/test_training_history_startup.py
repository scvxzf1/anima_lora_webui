"""History startup repair tests."""

from __future__ import annotations

from tests import training_resume_test_support as _training_resume_support

globals().update(
    {
        name: value
        for name, value in vars(_training_resume_support).items()
        if name == "Path" or not name.startswith("__")
    }
)

# Split: history

def test_service_startup_marks_orphaned_running_tasks_interrupted(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    task_dir = _write_group_task(
        history_dir,
        "20260517-000001-training-imported-demo",
        started_at=1000.0,
        steps=[(1, 0.3), (2, 0.2)],
        state="running",
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    TrainingService(web.Application())

    meta = json.loads((task_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["state"] == "interrupted"
    assert "中断" in meta["message"]
    assert meta["finished_at"] == 1002.0
    assert meta["log_count"] == 2
    assert meta["metric_count"] == 2

def test_service_startup_keeps_history_available_when_orphan_repair_write_fails(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    bad_dir = _write_group_task(
        history_dir,
        "20260517-000001-training-imported-bad",
        started_at=1000.0,
        state="running",
    )
    good_dir = _write_group_task(
        history_dir,
        "20260517-000002-training-imported-good",
        started_at=2000.0,
        state="running",
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    original_write_json_atomic = training_service._write_json_atomic

    def flaky_write_json_atomic(path: Path, payload: dict) -> None:
        if path.parent == bad_dir:
            raise OSError("stale file handle")
        original_write_json_atomic(path, payload)

    monkeypatch.setattr(training_service, "_write_json_atomic", flaky_write_json_atomic)

    count = training_service._mark_orphaned_running_history_tasks()

    assert count == 1
    bad_meta = json.loads((bad_dir / "meta.json").read_text(encoding="utf-8"))
    good_meta = json.loads((good_dir / "meta.json").read_text(encoding="utf-8"))
    assert bad_meta["state"] == "running"
    assert good_meta["state"] == "interrupted"

    svc = TrainingService(web.Application())
    payload = svc.get_config_group_timeline("imported", "demo", "default")
    tasks = {task["id"]: task for task in payload["tasks"]}
    assert tasks[bad_dir.name]["state"] == "running"
    assert tasks[good_dir.name]["state"] == "interrupted"

