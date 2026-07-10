"""History artifact / log path safety tests."""

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

def test_history_log_download_path_rejects_escape(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    task_id = "20260524-131153-training-imported-522"
    task_dir = history_dir / task_id
    task_dir.mkdir(parents=True)
    logs_path = task_dir / "logs.jsonl"
    logs_path.write_text('{"line": "ok"}\n', encoding="utf-8")
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    svc = TrainingService(web.Application())

    assert svc.get_history_log_path(task_id) == logs_path.resolve()
    with pytest.raises(ValueError, match="任务 ID 不合法"):
        svc.get_history_log_path("../outside")

    logs_path.unlink()
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"line": "outside"}\n', encoding="utf-8")
    logs_path.symlink_to(outside)
    with pytest.raises(ValueError, match="路径不合法"):
        svc.get_history_log_path(task_id)

def test_history_log_download_route_returns_full_jsonl(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    task_id = "20260524-131153-training-imported-522"
    task_dir = history_dir / task_id
    task_dir.mkdir(parents=True)
    logs_path = task_dir / "logs.jsonl"
    logs_path.write_text('{"line": "first"}\n{"line": "last"}\n', encoding="utf-8")
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    svc = TrainingService(web.Application())
    req = _FakeJsonRequest({}, {"training_service": svc}, match_info={"task_id": task_id})

    response = asyncio.run(training_routes.handle_history_log_download(req))

    assert response.status == 200
    assert response._path == logs_path.resolve()
    assert response.headers["Content-Disposition"].endswith(f"{task_id}.logs.jsonl")

def test_history_artifact_path_allows_whitelisted_task_and_runtime_files(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    output_root = tmp_path / "runs"
    task_id = "20260524-131153-training-imported-522"
    task_dir = history_dir / task_id
    run_dir = output_root / "demo-run"
    task_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (task_dir / "config.snapshot.toml").write_text('learning_rate = 0.0001\n', encoding="utf-8")
    runtime_config = run_dir / "config.runtime.toml"
    runtime_config.write_text('output_name = "demo"\n', encoding="utf-8")
    (task_dir / "meta.json").write_text(
        json.dumps({
            "id": task_id,
            "job": "training",
            "state": "idle",
            "run_dir": str(run_dir),
            "runtime_config_file": str(runtime_config),
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    monkeypatch.setattr(training_service, "resolve_output_root", lambda: output_root)
    svc = TrainingService(web.Application())

    assert svc.get_history_artifact_path(task_id, "config-snapshot") == (task_dir / "config.snapshot.toml").resolve()
    assert svc.get_history_artifact_path(task_id, "runtime-config") == runtime_config.resolve()

    outside = tmp_path / "outside.toml"
    outside.write_text("escape = true\n", encoding="utf-8")
    (task_dir / "meta.json").write_text(
        json.dumps({
            "id": task_id,
            "job": "training",
            "state": "idle",
            "run_dir": str(run_dir),
            "runtime_config_file": str(outside),
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="运行文件路径不合法"):
        svc.get_history_artifact_path(task_id, "runtime-config")

def test_history_artifact_path_keeps_old_runtime_after_output_root_changes(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    old_output_root = tmp_path / "old-runs"
    current_output_root = tmp_path / "current-runs"
    task_id = "20260524-131153-training-imported-522"
    task_dir = history_dir / task_id
    run_dir = old_output_root / "demo-run"
    task_dir.mkdir(parents=True)
    (run_dir / "model_cache").mkdir(parents=True)
    (run_dir / "dataset_cache").mkdir(parents=True)
    (run_dir / "training_output").mkdir(parents=True)
    runtime_config = run_dir / "config.runtime.toml"
    runtime_config.write_text('output_name = "demo"\n', encoding="utf-8")
    (task_dir / "meta.json").write_text(
        json.dumps({
            "id": task_id,
            "job": "training",
            "state": "idle",
            "run_dir": str(run_dir),
            "runtime_config_file": str(runtime_config),
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    monkeypatch.setattr(training_service, "resolve_output_root", lambda: current_output_root)
    svc = TrainingService(web.Application())

    assert svc.get_history_artifact_path(task_id, "runtime-config") == runtime_config.resolve()

    shutil.rmtree(run_dir / "model_cache")
    with pytest.raises(ValueError, match="运行文件路径不合法"):
        svc.get_history_artifact_path(task_id, "runtime-config")

def test_history_artifact_route_sets_inline_or_attachment(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    task_id = "20260524-131153-training-imported-522"
    task_dir = history_dir / task_id
    task_dir.mkdir(parents=True)
    snapshot = task_dir / "config.snapshot.toml"
    snapshot.write_text("max_train_epochs = 1\n", encoding="utf-8")
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    svc = TrainingService(web.Application())

    view_req = _FakeJsonRequest(
        {},
        {"training_service": svc},
        match_info={"task_id": task_id, "artifact_key": "config-snapshot"},
    )
    view_response = asyncio.run(training_routes.handle_history_artifact(view_req))
    assert view_response.status == 200
    assert view_response._path == snapshot.resolve()
    assert view_response.headers["Content-Disposition"].startswith("inline;")

    download_req = _FakeJsonRequest(
        {},
        {"training_service": svc},
        query={"download": "1"},
        match_info={"task_id": task_id, "artifact_key": "config-snapshot"},
    )
    download_response = asyncio.run(training_routes.handle_history_artifact(download_req))
    assert download_response.status == 200
    assert download_response.headers["Content-Disposition"].startswith("attachment;")

