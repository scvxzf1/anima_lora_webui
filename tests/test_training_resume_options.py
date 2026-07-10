"""Resume options / checkpoint diagnostics tests split from test_training_resume.py."""

from __future__ import annotations

from tests import training_resume_test_support as _training_resume_support

globals().update(
    {
        name: value
        for name, value in vars(_training_resume_support).items()
        if name == "Path" or not name.startswith("__")
    }
)

# Split: resume_options

def test_resume_options_find_checkpoint_state(tmp_path, monkeypatch):
    history_dir, task_id, state_dir = _write_resume_history(tmp_path)
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    svc = TrainingService(web.Application())
    payload = svc.get_resume_options(task_id)

    assert payload["ok"] is True
    assert payload["default_checkpoint"] == str(state_dir)
    assert payload["checkpoints"][0]["kind"] == "checkpoint"
    assert payload["checkpoints"][0]["step"] == 42
    assert payload["checkpoints"][0]["scope"] == "task"
    assert payload["checkpoints"][0]["state_complete"] is True
    assert payload["checkpoints"][0]["state_integrity"]["optimizer"] is True
    assert payload["checkpoints"][0]["state_integrity"]["scheduler"] is True
    assert payload["diagnostic"]["complete_state_count"] == 1

def test_resume_options_mark_incomplete_state_unavailable(tmp_path, monkeypatch):
    history_dir, task_id, state_dir = _write_resume_history(tmp_path)
    (state_dir / "optimizer.bin").unlink()
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    svc = TrainingService(web.Application())
    payload = svc.get_resume_options(task_id)

    checkpoint = payload["checkpoints"][0]
    assert payload["default_checkpoint"] == ""
    assert checkpoint["state_complete"] is False
    assert checkpoint["missing_state_files"] == ["optimizer.bin"]
    assert checkpoint["resume_available"] is False
    assert "缺少 optimizer.bin" in checkpoint["unavailable_reason"]
    assert payload["diagnostic"]["incomplete_state_count"] == 1
    assert payload["diagnostic"]["missing_state_files"] == ["optimizer.bin"]
    with pytest.raises(ValueError, match="缺少 optimizer.bin"):
        asyncio.run(svc.resume_from_history_task(task_id, str(state_dir)))

def test_resume_options_mark_completed_checkpoint_unavailable(tmp_path, monkeypatch):
    history_dir, task_id, state_dir = _write_resume_history(tmp_path)
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    monkeypatch.setattr(
        config_service,
        "estimate_training_steps",
        lambda *_args, **_kwargs: {"total_steps": 42},
    )

    svc = TrainingService(web.Application())
    payload = svc.get_resume_options(task_id)

    checkpoint = payload["checkpoints"][0]
    assert payload["default_checkpoint"] == ""
    assert checkpoint["path"] == str(state_dir)
    assert checkpoint["target_total_steps"] == 42
    assert checkpoint["remaining_steps"] == 0
    assert checkpoint["resume_available"] is False
    assert checkpoint["unavailable_reason"] == (
        "这个检查点已训练到 step 42，当前配置目标是 42，"
        "继续训练不会产生新步数。请先增加 max_train_steps / max_train_epochs，或改用权重热启动。"
    )

    with pytest.raises(ValueError, match="继续训练不会产生新步数"):
        asyncio.run(svc.resume_from_history_task(task_id, str(state_dir)))
    with pytest.raises(ValueError, match="继续训练不会产生新步数"):
        asyncio.run(svc.resume_from_history_task(task_id))

def test_resume_options_find_numbered_checkpoint_states(tmp_path, monkeypatch):
    history_dir, task_id, state_dir = _write_resume_history(tmp_path)
    output_dir = state_dir.parent
    numbered_state = output_dir / "demo-checkpoint-000002-state"
    numbered_state.mkdir()
    (numbered_state / "train_state.json").write_text(
        json.dumps({"current_epoch": 2, "current_step": 60}),
        encoding="utf-8",
    )
    _write_fake_accelerate_state_files(numbered_state)
    (output_dir / "demo-checkpoint-000002.safetensors").write_bytes(b"numbered")
    os.utime(numbered_state / "train_state.json", (1501.0, 1501.0))
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    svc = TrainingService(web.Application())
    payload = svc.get_resume_options(task_id)

    assert payload["default_checkpoint"] == str(numbered_state)
    selected = payload["checkpoints"][0]
    assert selected["name"] == "demo-checkpoint-000002-state"
    assert selected["kind"] == "checkpoint"
    assert selected["paired_weight"] == str(output_dir / "demo-checkpoint-000002.safetensors")

def test_resume_options_hide_other_directory_states(tmp_path, monkeypatch):
    history_dir, task_id, state_dir = _write_resume_history(tmp_path)
    output_dir = state_dir.parent
    other_state = output_dir / "other-checkpoint-state"
    other_state.mkdir()
    (other_state / "train_state.json").write_text(
        json.dumps({"current_epoch": 9, "current_step": 999}),
        encoding="utf-8",
    )
    (output_dir / "other-checkpoint.safetensors").write_bytes(b"other")
    os.utime(other_state / "train_state.json", (3000.0, 3000.0))
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    svc = TrainingService(web.Application())
    payload = svc.get_resume_options(task_id)

    assert payload["ok"] is True
    assert [item["path"] for item in payload["checkpoints"]] == [str(state_dir)]
    assert all(item["scope"] == "task" for item in payload["checkpoints"])

def test_resume_options_diagnose_missing_output_dir(tmp_path, monkeypatch):
    task_id = "20260517-000000-training-imported-missing-output"
    history_dir = tmp_path / "history"
    task_dir = history_dir / task_id
    task_dir.mkdir(parents=True)
    missing_output_dir = tmp_path / "missing-output"
    (task_dir / "meta.json").write_text(
        json.dumps({
            "id": task_id,
            "job": "training",
            "state": "idle",
            "variant": "demo",
            "preset": "default",
            "methods_subdir": "imported",
            "output_dir": str(missing_output_dir),
            "started_at": 1000.0,
            "finished_at": 2000.0,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    svc = TrainingService(web.Application())
    payload = svc.get_resume_options(task_id)

    assert payload["ok"] is True
    assert payload["checkpoints"] == []
    assert payload["diagnostic"]["output_dir_exists"] is False
    assert "输出目录不存在" in payload["diagnostic"]["reason"]
    assert "输出目录不存在" in payload["message"]

def test_resume_options_diagnose_missing_train_state(tmp_path, monkeypatch):
    task_id = "20260517-000000-training-imported-no-state"
    history_dir = tmp_path / "history"
    task_dir = history_dir / task_id
    output_dir = tmp_path / "output-no-state"
    (output_dir / "demo-checkpoint-state").mkdir(parents=True)
    task_dir.mkdir(parents=True)
    (task_dir / "meta.json").write_text(
        json.dumps({
            "id": task_id,
            "job": "training",
            "state": "idle",
            "variant": "demo",
            "preset": "default",
            "methods_subdir": "imported",
            "output_dir": str(output_dir),
            "started_at": 1000.0,
            "finished_at": 2000.0,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    svc = TrainingService(web.Application())
    payload = svc.get_resume_options(task_id)

    assert payload["ok"] is True
    assert payload["checkpoints"] == []
    assert payload["diagnostic"]["output_dir_exists"] is True
    assert payload["diagnostic"]["state_dir_count"] == 1
    assert payload["diagnostic"]["train_state_count"] == 0
    assert "没有包含 train_state.json" in payload["diagnostic"]["reason"]

