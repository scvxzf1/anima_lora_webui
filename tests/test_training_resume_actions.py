"""Resume-from-history actions tests split from test_training_resume.py."""

from __future__ import annotations

from tests import training_resume_test_support as _training_resume_support

globals().update(
    {
        name: value
        for name, value in vars(_training_resume_support).items()
        if name == "Path" or not name.startswith("__")
    }
)

# Split: resume_actions

def test_handle_resume_passes_duration_overrides():
    class FakeService:
        async def resume_from_history_task(
            self,
            task_id,
            checkpoint=None,
            *,
            duration_overrides=None,
            gpu_whitelist=None,
        ):
            return {
                "ok": True,
                "task_id": task_id,
                "checkpoint": checkpoint,
                "duration_overrides": duration_overrides,
                "gpu_whitelist": gpu_whitelist,
            }

    req = _FakeJsonRequest(
        {
            "task_id": "task-a",
            "checkpoint": "state-dir",
            "duration_overrides": {"max_train_epochs": 1},
            "gpu_whitelist": [0],
        },
        {"training_service": FakeService()},
    )

    response = asyncio.run(training_routes.handle_resume(req))
    payload = json.loads(response.text)

    assert response.status == 200
    assert payload["task_id"] == "task-a"
    assert payload["checkpoint"] == "state-dir"
    assert payload["duration_overrides"] == {"max_train_epochs": 1}
    assert payload["gpu_whitelist"] == [0]

def test_resume_from_history_allows_remaining_steps_and_clones_runtime(tmp_path, monkeypatch):
    history_dir, task_id, state_dir = _write_resume_history(tmp_path)
    snapshot_path = history_dir / task_id / "config.snapshot.toml"
    snapshot_path.write_text(
        snapshot_path.read_text(encoding="utf-8")
        + 'network_weights = "weights/old-hotstart.safetensors"\n'
        + "dim_from_weights = true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    output_root = _patch_resume_runtime_output_root(monkeypatch, tmp_path)
    monkeypatch.setattr(
        config_service,
        "estimate_training_steps",
        lambda *_args, **_kwargs: {"total_steps": 100},
    )

    svc = TrainingService(web.Application())
    captured = {}

    async def fake_start(variant, preset, extra_args, methods_subdir, **kwargs):
        captured.update({
            "variant": variant,
            "preset": preset,
            "extra_args": extra_args,
            "methods_subdir": methods_subdir,
            **kwargs,
        })

    svc.start = fake_start

    result = asyncio.run(svc.resume_from_history_task(task_id))

    assert result["ok"] is True
    runtime_config = Path(captured["config_file"])
    assert runtime_config.name == "config.runtime.toml"
    assert output_root in runtime_config.parents
    assert runtime_config.exists()
    assert captured["extra_args"][:2] == ["--resume", str(state_dir)]
    assert captured["resume_info"]["target_total_steps"] == 100
    assert captured["resume_info"]["remaining_steps"] == 58
    runtime_cfg = toml.loads(runtime_config.read_text(encoding="utf-8"))
    assert runtime_cfg["output_dir"].startswith(str(output_root))
    assert runtime_cfg["output_dir"].endswith("/training_output")
    assert runtime_cfg["output_dir"] != str(state_dir.parent)
    assert "network_weights" not in runtime_cfg
    assert "dim_from_weights" not in runtime_cfg

def test_resume_from_history_duration_epoch_override_appends_steps(tmp_path, monkeypatch):
    history_dir, task_id, state_dir = _write_resume_history(tmp_path)
    resized_dir = tmp_path / "post_image_dataset" / "resized"
    for index in range(3):
        (resized_dir / f"image-{index}.png").write_bytes(b"png")
    snapshot_path = history_dir / task_id / "config.snapshot.toml"
    snapshot_path.write_text(
        snapshot_path.read_text(encoding="utf-8") + "max_train_epochs = 5\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    output_root = _patch_resume_runtime_output_root(monkeypatch, tmp_path)
    monkeypatch.setattr(
        config_service,
        "estimate_training_steps",
        lambda *_args, **_kwargs: {"total_steps": 42},
    )

    svc = TrainingService(web.Application())
    captured = {}

    async def fake_start(variant, preset, extra_args, methods_subdir, **kwargs):
        captured.update({
            "variant": variant,
            "preset": preset,
            "extra_args": extra_args,
            "methods_subdir": methods_subdir,
            **kwargs,
        })

    svc.start = fake_start

    result = asyncio.run(
        svc.resume_from_history_task(
            task_id,
            str(state_dir),
            duration_overrides={"max_train_epochs": 1},
        )
    )

    assert result["ok"] is True
    runtime_config = Path(captured["config_file"])
    assert output_root in runtime_config.parents
    runtime_cfg = toml.loads(runtime_config.read_text(encoding="utf-8"))
    assert "max_train_epochs" not in runtime_cfg
    assert runtime_cfg["max_train_steps"] == 45
    assert captured["resume_info"]["duration_overrides"] == {
        "mode": "epochs",
        "max_train_epochs": 1,
        "steps_per_epoch": 3,
        "resume_step": 42,
        "append_steps": 3,
        "target_total_steps": 45,
    }
    assert captured["resume_info"]["remaining_steps"] == 3

def test_resume_from_history_rejects_other_directory_state(tmp_path, monkeypatch):
    history_dir, task_id, state_dir = _write_resume_history(tmp_path)
    other_state = state_dir.parent / "other-checkpoint-state"
    other_state.mkdir()
    (other_state / "train_state.json").write_text(
        json.dumps({"current_epoch": 9, "current_step": 999}),
        encoding="utf-8",
    )
    os.utime(other_state / "train_state.json", (3000.0, 3000.0))
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    svc = TrainingService(web.Application())

    try:
        asyncio.run(svc.resume_from_history_task(task_id, str(other_state)))
    except ValueError as e:
        assert "未找到指定的检查点" in str(e)
    else:
        raise AssertionError("不应允许从同目录其他训练状态续训")

def test_resume_from_history_uses_snapshot_and_resume_args(tmp_path, monkeypatch):
    history_dir, task_id, state_dir = _write_resume_history(tmp_path)
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    output_root = _patch_resume_runtime_output_root(monkeypatch, tmp_path)

    svc = TrainingService(web.Application())
    captured = {}

    async def fake_start(variant, preset, extra_args, methods_subdir, **kwargs):
        captured.update({
            "variant": variant,
            "preset": preset,
            "extra_args": extra_args,
            "methods_subdir": methods_subdir,
            **kwargs,
        })

    svc.start = fake_start

    result = asyncio.run(svc.resume_from_history_task(task_id, str(state_dir)))

    assert result["ok"] is True
    assert captured["variant"] == "demo"
    assert captured["methods_subdir"] == "imported"
    assert captured["extra_args"] == [
        "--resume",
        str(state_dir),
        "--skip_until_initial_step",
    ]
    runtime_config = Path(captured["config_file"])
    assert runtime_config.name == "config.runtime.toml"
    assert output_root in runtime_config.parents
    assert runtime_config.exists()
    assert not captured["config_file"].endswith(f"{task_id}/config.snapshot.toml")
    runtime_cfg = toml.loads(runtime_config.read_text(encoding="utf-8"))
    assert runtime_cfg["output_dir"].endswith("/training_output")
    assert runtime_cfg["output_dir"] != str(state_dir.parent)
    assert captured["use_runtime_dir"] is False
    assert captured["resume_info"]["checkpoint"] == str(state_dir)
    assert captured["resume_info"]["history_group_key"] == "legacy:imported\u0001demo\u0001default"
    assert captured["resume_info"]["history_group_label"] == "imported / demo / default"

def test_resume_from_history_forwards_gpu_whitelist(tmp_path, monkeypatch):
    history_dir, task_id, state_dir = _write_resume_history(tmp_path)
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    _patch_resume_runtime_output_root(monkeypatch, tmp_path)

    svc = TrainingService(web.Application())
    captured = {}

    async def fake_start(variant, preset, extra_args, methods_subdir, **kwargs):
        captured.update(kwargs)

    svc.start = fake_start

    result = asyncio.run(
        svc.resume_from_history_task(task_id, str(state_dir), gpu_whitelist=["1", "bad", 2, 2])
    )

    assert result["ok"] is True
    assert captured["gpu_whitelist"] == ["1", "bad", 2, 2]

def test_resume_from_history_requires_config_snapshot(tmp_path, monkeypatch):
    history_dir, task_id, state_dir = _write_resume_history(tmp_path)
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    (history_dir / task_id / "config.snapshot.toml").unlink()

    svc = TrainingService(web.Application())

    try:
        asyncio.run(svc.resume_from_history_task(task_id, str(state_dir)))
    except ValueError as e:
        assert "配置快照" in str(e)
    else:
        raise AssertionError("缺少配置快照时不应允许续训")

def test_resume_history_meta_inherits_source_group(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    svc = TrainingService(web.Application())

    svc._start_history_task(
        job="training",
        variant="demo",
        preset="default",
        methods_subdir="imported",
        output_dir="output/runs/demo-resume-20260523-130000/training_output",
        sample_dir="output/runs/demo-resume-20260523-130000/training_output/sample",
        data_dirs={},
        sample_config={},
        command=["python", "train.py"],
        resume_info={
            "source_task_id": "source-task",
            "history_group_key": "source:configs/imported/demo.toml",
            "history_group_label": "configs/imported/demo.toml",
            "history_source_config_file": "configs/imported/demo.toml",
            "checkpoint_name": "demo-checkpoint-state",
        },
    )

    task = svc.list_history_tasks()[0]

    assert task["history_group_key"] == "source:configs/imported/demo.toml"
    assert task["history_group_label"] == "configs/imported/demo.toml"
    assert task["history_source_config_file"] == "configs/imported/demo.toml"
    assert task["history_run_label"] == "demo-resume-20260523-130000"

