"""Cross-domain delete/path boundary combination locks."""

from __future__ import annotations

from pathlib import Path

import pytest
import toml
from aiohttp import web

from web.services import path_safety
from web.services import preview_service
from web.services import settings_service
from web.services import training_service
from web.services.training_service import TrainingService


def test_output_root_and_repo_root_boundaries_do_not_overlap_secret(tmp_path: Path):
    repo = tmp_path / "repo"
    output = tmp_path / "output"
    secret = tmp_path / "secret" / "a.safetensors"
    repo.mkdir()
    output.mkdir()
    secret.parent.mkdir()
    secret.write_bytes(b"x")
    allow = path_safety.allowed_weight_dirs(root=repo, output_root=output)
    assert path_safety.is_under_allowed_dirs(secret, allow) is False
    inside = output / "run" / "a.safetensors"
    inside.parent.mkdir(parents=True)
    inside.write_bytes(b"x")
    assert path_safety.is_under_allowed_dirs(inside, allow) is True


def test_queue_runtime_delete_blocked_when_queue_references_run_dir(tmp_path: Path, monkeypatch):
    """History runtime delete plan must block dirs still referenced by queue items."""
    history_dir = tmp_path / "history"
    queue_dir = tmp_path / "queue"
    run_dir = tmp_path / "runs" / "demo-run"
    history_dir.mkdir()
    queue_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)

    task_id = "20260711-000001-training-gui-methods-lora"
    task_dir = history_dir / task_id
    task_dir.mkdir()
    (task_dir / "meta.toml").write_text(
        toml.dumps(
            {
                "id": task_id,
                "job": "training",
                "status": "done",
                "started_at": 1.0,
                "run_dir": str(run_dir),
                "training_output_dir": str(run_dir / "training_output"),
                "history_run_label": run_dir.name,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    monkeypatch.setattr(training_service, "QUEUE_DIR", queue_dir)
    monkeypatch.setattr(training_service, "QUEUE_FILE", queue_dir / "queue.json")

    svc = TrainingService(web.Application())
    # Inject a queued item that still points at the runtime dir.
    svc._queue = [  # type: ignore[attr-defined]
        {
            "id": "q1",
            "status": "queued",
            "run_dir": str(run_dir),
            "output_dir": str(run_dir / "training_output"),
            "history_task_ids": [task_id],
        }
    ]

    plan = svc._plan_history_delete([task_id], delete_runtime_dirs=True)  # type: ignore[attr-defined]
    blocked_ids = {item.get("id") for item in plan.get("blocked") or []}
    blocked_reasons = " ".join(str(item.get("reason") or "") for item in plan.get("blocked") or [])
    # Either the runtime path is blocked via queue refs, or blocked list is non-empty.
    assert plan["blocked"], f"expected queue/runtime blockers, got plan={plan}"
    assert task_id in blocked_ids or "队列" in blocked_reasons or "queue" in blocked_reasons.lower() or "运行" in blocked_reasons


def test_history_delete_running_task_raises_conflict(tmp_path: Path, monkeypatch):
    history_dir = tmp_path / "history"
    task_id = "20260711-000002-training-gui-methods-lora"
    task_dir = history_dir / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "meta.toml").write_text(
        toml.dumps({"id": task_id, "job": "training", "status": "running", "started_at": 1.0}),
        encoding="utf-8",
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    monkeypatch.setattr(training_service, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(training_service, "QUEUE_FILE", tmp_path / "queue" / "queue.json")
    (tmp_path / "queue").mkdir(parents=True, exist_ok=True)

    svc = TrainingService(web.Application())
    svc.status = "running"
    svc.current_task_id = task_id

    with pytest.raises(RuntimeError, match="不能删除"):
        svc.delete_history_task(task_id)


def test_preview_and_image_test_delete_reject_outside_current_dir(tmp_path: Path, monkeypatch):
    root = tmp_path / "repo"
    inference = root / "output" / "tests"
    other = root / "output" / "other"
    inference.mkdir(parents=True)
    other.mkdir(parents=True)
    target = other / "blocked.png"
    target.write_bytes(b"png")

    monkeypatch.setattr(preview_service, "ROOT", root)
    monkeypatch.setattr(
        preview_service,
        "get_preview_settings",
        lambda *args, **kwargs: {
            "effective_training_dir": "",
            "inference_dir": "output/tests",
            "custom_dir": "",
            "training_dir": "",
        },
    )
    # Keep resolve helpers on ROOT
    if hasattr(preview_service, "_resolve_preview_dir"):
        pass

    payload = preview_service.delete_preview_images(
        "inference",
        ["output/other/blocked.png"],
    )
    assert payload["ok"] is False
    assert payload["blocked_count"] >= 1
    assert target.exists()


def test_switch_output_root_updates_weight_allowlist_boundaries(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    old_output = tmp_path / "old_output"
    new_output = tmp_path / "new_output"
    secret = tmp_path / "secret" / "a.safetensors"
    for p in (repo, old_output, new_output, secret.parent):
        p.mkdir(parents=True, exist_ok=True)
    secret.write_bytes(b"x")
    old_weight = old_output / "run" / "old.safetensors"
    new_weight = new_output / "run" / "new.safetensors"
    old_weight.parent.mkdir(parents=True)
    new_weight.parent.mkdir(parents=True)
    old_weight.write_bytes(b"x")
    new_weight.write_bytes(b"x")

    settings_file = tmp_path / "configs" / "web-ui-settings.toml"
    settings_file.parent.mkdir(parents=True)
    settings_file.write_text(
        f'[global]\noutput_root = "{old_output.as_posix()}"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(settings_service, "ROOT", repo)
    monkeypatch.setattr(settings_service, "SETTINGS_FILE", settings_file)

    # Before switch: old allowed, new/secret not under old allowlist.
    allow_old = path_safety.allowed_weight_dirs(
        root=repo,
        output_root=settings_service.resolve_output_root(),
    )
    assert path_safety.is_under_allowed_dirs(old_weight, allow_old) is True
    assert path_safety.is_under_allowed_dirs(new_weight, allow_old) is False
    assert path_safety.is_under_allowed_dirs(secret, allow_old) is False

    saved = settings_service.save_global_settings({"output_root": str(new_output)})
    assert Path(saved["output_root"]).name == new_output.name or saved["output_root"].endswith("new_output")

    allow_new = path_safety.allowed_weight_dirs(
        root=repo,
        output_root=settings_service.resolve_output_root(),
    )
    assert path_safety.is_under_allowed_dirs(new_weight, allow_new) is True
    assert path_safety.is_under_allowed_dirs(old_weight, allow_new) is False
    assert path_safety.is_under_allowed_dirs(secret, allow_new) is False

    # resolve_allowed_file follows switched root
    resolved = path_safety.resolve_allowed_file(
        str(new_weight),
        root=repo,
        allowed_dirs=allow_new,
    )
    assert resolved == new_weight.resolve()
    with pytest.raises(ValueError):
        path_safety.resolve_allowed_file(
            str(old_weight),
            root=repo,
            allowed_dirs=allow_new,
        )
