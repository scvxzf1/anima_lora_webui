from __future__ import annotations

import asyncio
import json
import os

from aiohttp import web

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
    config_text=None,
):
    task_dir = history_dir / task_id
    output_dir = history_dir / "output" / task_id
    task_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "id": task_id,
        "job": job,
        "state": "idle",
        "variant": variant,
        "preset": preset,
        "methods_subdir": methods_subdir,
        "output_dir": str(output_dir),
        "sample_dir": str(output_dir / "sample"),
        "started_at": started_at,
        "started_at_text": f"ts-{int(started_at)}",
        "finished_at": started_at + 10,
        "finished_at_text": f"ts-{int(started_at + 10)}",
        "archived": archived,
    }
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
    assert payload["logs"] == []


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
