from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace

from aiohttp import web

from library.training.checkpoints import save_checkpoint_state
from web.services import training_service
from web.services.training_service import TrainingService


def _write_resume_history(tmp_path):
    task_id = "20260517-000000-training-imported-demo"
    history_dir = tmp_path / "history"
    task_dir = history_dir / task_id
    output_dir = tmp_path / "output"
    state_dir = output_dir / "demo-checkpoint-state"
    task_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)

    started_at = 1000.0
    finished_at = 2000.0
    meta = {
        "id": task_id,
        "job": "training",
        "state": "idle",
        "variant": "demo",
        "preset": "default",
        "methods_subdir": "imported",
        "output_dir": str(output_dir),
        "sample_dir": str(output_dir / "sample"),
        "data_dirs": {},
        "sample_config": {},
        "started_at": started_at,
        "finished_at": finished_at,
    }
    (task_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (task_dir / "config.snapshot.toml").write_text(
        f'output_dir = "{output_dir.as_posix()}"\noutput_name = "demo"\n',
        encoding="utf-8",
    )
    (state_dir / "train_state.json").write_text(
        json.dumps({"current_epoch": 3, "current_step": 42}),
        encoding="utf-8",
    )
    (output_dir / "demo-checkpoint.safetensors").write_bytes(b"stub")
    os.utime(state_dir / "train_state.json", (1500.0, 1500.0))
    return history_dir, task_id, state_dir


def _write_group_task(
    history_dir,
    task_id,
    *,
    job="training",
    variant="demo",
    preset="default",
    methods_subdir="imported",
    started_at=1000.0,
    steps=None,
    archived=False,
    state="idle",
    finished_at=None,
    config_text=None,
    resume_from=None,
):
    task_dir = history_dir / task_id
    output_dir = history_dir / "output" / task_id
    task_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "id": task_id,
        "job": job,
        "state": state,
        "variant": variant,
        "preset": preset,
        "methods_subdir": methods_subdir,
        "output_dir": str(output_dir),
        "sample_dir": str(output_dir / "sample"),
        "started_at": started_at,
        "started_at_text": f"ts-{int(started_at)}",
        "finished_at": finished_at if finished_at is not None else (started_at + 10 if state != "running" else None),
        "finished_at_text": "" if state == "running" and finished_at is None else f"ts-{int((finished_at if finished_at is not None else started_at + 10))}",
        "archived": archived,
    }
    if resume_from is not None:
        meta["resume_from"] = resume_from
    (task_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (task_dir / "config.snapshot.toml").write_text(
        config_text or f'output_dir = "{output_dir.as_posix()}"\noutput_name = "{task_id}"\n',
        encoding="utf-8",
    )
    logs = []
    metrics = []
    for idx, (step, loss) in enumerate(steps or [], start=1):
        ts = started_at + idx
        logs.append({
            "id": idx,
            "kind": "progress",
            "line": f"steps: 1%| | {step}/100 [00:00<00:00, 1.00s/it, avr_loss={loss}]",
            "ts": ts,
        })
        metrics.append({"step": step, "loss": loss, "ts": ts})
    if logs:
        (task_dir / "logs.jsonl").write_text(
            "\n".join(json.dumps(item) for item in logs) + "\n",
            encoding="utf-8",
        )
    if metrics:
        (task_dir / "metrics.jsonl").write_text(
            "\n".join(json.dumps(item) for item in metrics) + "\n",
            encoding="utf-8",
        )
    return task_dir


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
    assert captured["config_file"].endswith(f"{task_id}/config.snapshot.toml")
    assert captured["resume_info"]["checkpoint"] == str(state_dir)


def test_resume_from_history_forwards_gpu_whitelist(tmp_path, monkeypatch):
    history_dir, task_id, state_dir = _write_resume_history(tmp_path)
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

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


class _FakeAccelerator:
    def __init__(self, *, step=2, fail=False):
        self.step = step
        self.fail = fail

    def save_state(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "train_state.json"), "w", encoding="utf-8") as f:
            json.dump({"current_epoch": 1, "current_step": self.step}, f)
        if self.fail:
            raise RuntimeError("boom")


def _checkpoint_args(tmp_path):
    return SimpleNamespace(output_dir=str(tmp_path), output_name="demo")


def test_save_checkpoint_state_replaces_state_after_success(tmp_path):
    state_dir = tmp_path / "demo-checkpoint-state"
    state_dir.mkdir()
    (state_dir / "train_state.json").write_text(
        json.dumps({"current_epoch": 1, "current_step": 1}),
        encoding="utf-8",
    )

    save_checkpoint_state(_checkpoint_args(tmp_path), _FakeAccelerator(step=7))

    assert json.loads((state_dir / "train_state.json").read_text(encoding="utf-8"))["current_step"] == 7
    assert not (tmp_path / "demo-checkpoint-state.tmp").exists()
    assert not (tmp_path / "demo-checkpoint-state.backup").exists()


def test_save_checkpoint_state_keeps_old_state_on_failure(tmp_path):
    state_dir = tmp_path / "demo-checkpoint-state"
    state_dir.mkdir()
    (state_dir / "train_state.json").write_text(
        json.dumps({"current_epoch": 1, "current_step": 1}),
        encoding="utf-8",
    )

    try:
        save_checkpoint_state(_checkpoint_args(tmp_path), _FakeAccelerator(step=8, fail=True))
    except RuntimeError as e:
        assert "boom" in str(e)
    else:
        raise AssertionError("save_state 失败时应继续抛出异常")

    assert json.loads((state_dir / "train_state.json").read_text(encoding="utf-8"))["current_step"] == 1
    assert not (tmp_path / "demo-checkpoint-state.tmp").exists()
    assert not (tmp_path / "demo-checkpoint-state.backup").exists()


def test_save_checkpoint_state_recovers_leftover_backup(tmp_path):
    state_dir = tmp_path / "demo-checkpoint-state"
    backup_dir = tmp_path / "demo-checkpoint-state.backup"
    tmp_dir = tmp_path / "demo-checkpoint-state.tmp"
    backup_dir.mkdir()
    tmp_dir.mkdir()
    (backup_dir / "train_state.json").write_text(
        json.dumps({"current_epoch": 1, "current_step": 3}),
        encoding="utf-8",
    )
    (tmp_dir / "train_state.json").write_text(
        json.dumps({"current_epoch": 1, "current_step": 99}),
        encoding="utf-8",
    )

    save_checkpoint_state(_checkpoint_args(tmp_path), _FakeAccelerator(step=4))

    assert json.loads((state_dir / "train_state.json").read_text(encoding="utf-8"))["current_step"] == 4
    assert not backup_dir.exists()
    assert not tmp_dir.exists()


def test_config_group_timeline_merges_by_file_identity(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    _write_group_task(
        history_dir,
        "20260517-000001-training-imported-demo",
        started_at=1000.0,
        steps=[(1, 0.3), (2, 0.2)],
        config_text='output_dir = "first"\n',
    )
    _write_group_task(
        history_dir,
        "20260517-000002-training-imported-demo",
        started_at=2000.0,
        steps=[(1, 0.19), (2, 0.18)],
        config_text='output_dir = "changed"\n',
    )
    _write_group_task(
        history_dir,
        "20260517-000003-training-imported-other",
        variant="other",
        started_at=3000.0,
        steps=[(1, 0.9)],
    )
    _write_group_task(
        history_dir,
        "20260517-000004-preprocess-imported-demo",
        job="preprocess",
        started_at=4000.0,
        steps=[(1, 0.8)],
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    svc = TrainingService(web.Application())
    payload = svc.get_config_group_timeline("imported", "demo", "default")

    assert payload["ok"] is True
    assert payload["summary"]["task_count"] == 2
    assert payload["summary"]["loss_count"] == 4
    assert payload["summary"]["log_count"] == 0
    assert payload["summary"]["progress_count"] == 4
    assert payload["summary"]["raw_log_count"] == 4
    assert [task["id"] for task in payload["tasks"]] == [
        "20260517-000001-training-imported-demo",
        "20260517-000002-training-imported-demo",
    ]
    assert [item["source_task_index"] for item in payload["metrics"]] == [1, 1, 2, 2]
    assert [item["visual_step"] for item in payload["metrics"]] == [1, 2, 3, 4]
    assert [item["display_step"] for item in payload["metrics"]] == [1, 2, 1, 2]
    assert payload["logs"] == []


def test_config_group_timeline_uses_resume_checkpoint_steps(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    _write_group_task(
        history_dir,
        "20260517-000001-training-imported-demo",
        started_at=1000.0,
        steps=[(1, 0.3), (2, 0.2)],
    )
    _write_group_task(
        history_dir,
        "20260517-000002-training-imported-demo",
        started_at=2000.0,
        steps=[(1, 0.19), (2, 0.18)],
        resume_from={"checkpoint_step": 2},
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    svc = TrainingService(web.Application())
    payload = svc.get_config_group_timeline("imported", "demo", "default")

    assert [item["step"] for item in payload["metrics"]] == [1, 2, 1, 2]
    assert [item["display_step"] for item in payload["metrics"]] == [1, 2, 3, 4]
    assert payload["metrics"][2]["stage_break_before"] is True
    assert payload["segments"][1]["display_step_offset"] == 2
    assert payload["segments"][1]["start_display_step"] == 3
    assert payload["segments"][1]["end_display_step"] == 4
    assert payload["summary"]["start_display_step"] == 1
    assert payload["summary"]["end_display_step"] == 4


def test_config_group_timeline_ignores_regressed_tail_steps(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    _write_group_task(
        history_dir,
        "20260517-000001-training-imported-demo",
        started_at=1000.0,
        steps=[(1, 0.3), (2, 0.2), (1, 0.4), (3, 0.1)],
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    svc = TrainingService(web.Application())
    payload = svc.get_config_group_timeline("imported", "demo", "default")

    assert [item["step"] for item in payload["metrics"]] == [1, 2, 3]
    assert [item["display_step"] for item in payload["metrics"]] == [1, 2, 3]


def test_config_group_timeline_respects_archived_filter(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    _write_group_task(
        history_dir,
        "20260517-000001-training-imported-demo",
        started_at=1000.0,
        steps=[(1, 0.3)],
    )
    _write_group_task(
        history_dir,
        "20260517-000002-training-imported-demo",
        started_at=2000.0,
        steps=[(1, 0.2)],
        archived=True,
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    svc = TrainingService(web.Application())
    payload = svc.get_config_group_timeline("imported", "demo", "default")
    with_archived = svc.get_config_group_timeline("imported", "demo", "default", include_archived=True)

    assert payload["summary"]["task_count"] == 1
    assert with_archived["summary"]["task_count"] == 2


def test_config_group_timeline_can_merge_selected_tasks_across_groups(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    _write_group_task(
        history_dir,
        "20260517-000001-training-imported-demo",
        variant="demo",
        started_at=1000.0,
        steps=[(1, 0.3)],
    )
    _write_group_task(
        history_dir,
        "20260517-000002-training-imported-other",
        variant="other",
        started_at=2000.0,
        steps=[(1, 0.2)],
    )
    _write_group_task(
        history_dir,
        "20260517-000003-training-imported-demo",
        variant="demo",
        started_at=3000.0,
        steps=[(1, 0.1)],
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    svc = TrainingService(web.Application())
    payload = svc.get_config_group_timeline(
        "imported",
        "demo",
        "default",
        task_ids=[
            "20260517-000001-training-imported-demo",
            "20260517-000002-training-imported-other",
        ],
    )

    assert payload["summary"]["selection_mode"] == "manual"
    assert payload["summary"]["task_count"] == 2
    assert payload["summary"]["group_count"] == 2
    assert payload["group"]["methods_subdir"] == "手动选择"
    assert [task["id"] for task in payload["tasks"]] == [
        "20260517-000001-training-imported-demo",
        "20260517-000002-training-imported-other",
    ]


def test_config_group_timeline_rejects_hidden_selected_archived_task(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    _write_group_task(
        history_dir,
        "20260517-000001-training-imported-demo",
        started_at=1000.0,
        steps=[(1, 0.3)],
        archived=True,
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    svc = TrainingService(web.Application())

    try:
        svc.get_config_group_timeline(
            "imported",
            "demo",
            "default",
            task_ids=["20260517-000001-training-imported-demo"],
        )
    except ValueError as e:
        assert "已隐藏" in str(e)
    else:
        raise AssertionError("隐藏的归档任务不应参与手动合并")


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

    svc = TrainingService(web.Application())
    payload = svc.get_config_group_timeline("imported", "demo", "default")
    assert payload["tasks"][0]["state"] == "interrupted"
