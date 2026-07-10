"""History delete / batch operations tests."""

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

def test_delete_history_task_removes_directory_with_bad_files(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    task_dir = history_dir / "20260524-124851-training-imported-522"
    task_dir.mkdir(parents=True)
    (task_dir / "metrics.jsonl").write_text("{}", encoding="utf-8")
    (task_dir / "progress.jsonl").write_text("{}", encoding="utf-8")
    (task_dir / "system.jsonl").write_text("{}", encoding="utf-8")
    # 模拟一个损坏到无法正常读取/删除的残留文件。
    bad_file = task_dir / "metrics.jsonl"
    bad_file.unlink()
    bad_file.write_bytes(b"broken")
    try:
        os.chmod(bad_file, 0)
    except OSError:
        pass

    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    svc = TrainingService(web.Application())

    result = svc.delete_history_task("20260524-124851-training-imported-522")

    assert result["ok"] is True
    assert not task_dir.exists()
    assert svc.list_history_tasks() == []

def test_delete_history_task_hides_record_when_cleanup_fails(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    task_id = "20260524-124851-training-imported-522"
    _write_group_task(history_dir, task_id, started_at=1000.0)
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    def fail_rmtree(_path):
        raise OSError("无效的参数")

    monkeypatch.setattr(training_service.shutil, "rmtree", fail_rmtree)
    svc = TrainingService(web.Application())

    result = svc.delete_history_task(task_id)

    assert result["ok"] is True
    assert "cleanup_error" in result
    assert not (history_dir / task_id).exists()
    assert svc.list_history_tasks() == []
    tombstones = [path for path in history_dir.iterdir() if ".deleting-" in path.name]
    assert len(tombstones) == 1

def test_delete_training_history_task_removes_linked_preprocess_task(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    run_dir = tmp_path / "runs" / "524-20260524-225059"
    training_id = "20260524-225152-training-imported-524"
    preprocess_id = "20260524-225059-preprocess-imported-524"
    other_preprocess_id = "20260524-230000-preprocess-imported-524"
    history_meta = {
        "run_dir": str(run_dir),
        "training_output_dir": str(run_dir / "training_output"),
        "history_group_key": "source:configs/imported/524.toml",
        "history_group_label": "configs/imported/524.toml",
        "history_source_config_file": "configs/imported/524.toml",
        "history_run_label": run_dir.name,
    }
    _write_group_task(
        history_dir,
        training_id,
        job="training",
        started_at=1000.0,
        history_meta=history_meta,
    )
    _write_group_task(
        history_dir,
        preprocess_id,
        job="preprocess",
        started_at=990.0,
        archived=True,
        history_meta=history_meta,
    )
    _write_group_task(
        history_dir,
        other_preprocess_id,
        job="preprocess",
        started_at=980.0,
        archived=True,
        history_meta={
            **history_meta,
            "run_dir": str(tmp_path / "runs" / "524-20260524-230000"),
            "training_output_dir": str(tmp_path / "runs" / "524-20260524-230000" / "training_output"),
            "history_run_label": "524-20260524-230000",
        },
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    svc = TrainingService(web.Application())

    result = svc.delete_history_task(training_id)

    assert result["ok"] is True
    assert result["deleted_task_ids"] == [training_id, preprocess_id]
    assert result["linked_preprocess_deleted"] == 1
    assert not (history_dir / training_id).exists()
    assert not (history_dir / preprocess_id).exists()
    assert (history_dir / other_preprocess_id).exists()

def test_delete_preprocess_history_task_does_not_remove_training_task(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    run_dir = tmp_path / "runs" / "524-20260524-225059"
    training_id = "20260524-225152-training-imported-524"
    preprocess_id = "20260524-225059-preprocess-imported-524"
    history_meta = {
        "run_dir": str(run_dir),
        "training_output_dir": str(run_dir / "training_output"),
        "history_group_key": "source:configs/imported/524.toml",
        "history_run_label": run_dir.name,
    }
    _write_group_task(history_dir, training_id, job="training", history_meta=history_meta)
    _write_group_task(history_dir, preprocess_id, job="preprocess", archived=True, history_meta=history_meta)
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    svc = TrainingService(web.Application())

    result = svc.delete_history_task(preprocess_id)

    assert result["ok"] is True
    assert result["deleted_task_ids"] == [preprocess_id]
    assert result["linked_preprocess_deleted"] == 0
    assert (history_dir / training_id).exists()
    assert not (history_dir / preprocess_id).exists()

def test_history_batch_archive_unarchive_and_group(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    first = "20260524-225152-training-imported-a"
    second = "20260524-225153-training-imported-b"
    _write_group_task(history_dir, first, job="training", started_at=1000.0)
    _write_group_task(history_dir, second, job="training", started_at=1001.0)
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    svc = TrainingService(web.Application())

    archived = svc.batch_update_history_tasks({"action": "archive", "task_ids": [first, second]})
    assert archived["updated"] == 2
    assert all(task["archived"] for task in svc.list_history_tasks(include_archived=True))

    grouped = svc.batch_update_history_tasks({"action": "set_group", "task_ids": [first], "group": "正式训练"})
    assert grouped["tasks"][0]["group"] == "正式训练"

    unarchived = svc.batch_update_history_tasks({"action": "unarchive", "task_ids": [first]})
    assert unarchived["updated"] == 1
    tasks = {task["id"]: task for task in svc.list_history_tasks(include_archived=True)}
    assert tasks[first]["archived"] is False
    assert tasks[second]["archived"] is True

def test_history_batch_delete_dry_run_and_confirm_removes_runtime_dir(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    output_root = tmp_path / "runs"
    run_dir = _write_web_runtime_dir(output_root, "524-20260524-225059")
    training_id = "20260524-225152-training-imported-524"
    preprocess_id = "20260524-225059-preprocess-imported-524"
    history_meta = {
        "run_dir": str(run_dir),
        "training_output_dir": str(run_dir / "training_output"),
        "history_run_label": run_dir.name,
    }
    _write_group_task(history_dir, training_id, job="training", history_meta=history_meta)
    _write_group_task(history_dir, preprocess_id, job="preprocess", archived=True, history_meta=history_meta)
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    monkeypatch.setattr(training_service, "resolve_output_root", lambda: output_root)
    svc = TrainingService(web.Application())

    preview = svc.batch_update_history_tasks({
        "action": "delete",
        "task_ids": [training_id],
        "delete_runtime_dirs": True,
        "dry_run": True,
    })

    assert preview["dry_run"] is True
    assert preview["blocked"] == []
    assert {task["id"] for task in preview["tasks"]} == {training_id, preprocess_id}
    assert preview["runtime_dirs"][0]["path"] == str(run_dir)

    with pytest.raises(ValueError, match="二次按钮确认"):
        svc.batch_update_history_tasks({
            "action": "delete",
            "task_ids": [training_id],
            "delete_runtime_dirs": True,
        })

    deleted = svc.batch_update_history_tasks({
        "action": "delete",
        "task_ids": [training_id],
        "delete_runtime_dirs": True,
        "confirmed": True,
    })

    assert deleted["ok"] is True
    assert set(deleted["deleted_task_ids"]) == {training_id, preprocess_id}
    assert deleted["deleted_runtime_dirs"] == [str(run_dir)]
    assert not run_dir.exists()
    assert svc.list_history_tasks(include_archived=True) == []

def test_history_batch_delete_blocks_current_task_and_queue_references(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    output_root = tmp_path / "runs"
    run_dir = _write_web_runtime_dir(output_root, "blocked-run")
    task_id = "20260524-225152-training-imported-blocked"
    _write_group_task(
        history_dir,
        task_id,
        job="training",
        history_meta={"run_dir": str(run_dir), "training_output_dir": str(run_dir / "training_output")},
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    monkeypatch.setattr(training_service, "resolve_output_root", lambda: output_root)
    svc = TrainingService(web.Application())
    svc.status = "running"
    svc.current_task_id = task_id
    svc._queue = {
        "items": [
            {
                "id": "queue-a",
                "state": "queued",
                "runtime_info": {"run_dir": str(run_dir)},
            },
        ],
    }

    preview = svc.batch_update_history_tasks({
        "action": "delete",
        "task_ids": [task_id],
        "delete_runtime_dirs": True,
        "dry_run": True,
    })

    reasons = "\n".join(item["reason"] for item in preview["blocked"])
    assert "当前运行中的任务不能删除" in reasons
    assert "队列项引用" in reasons
    with pytest.raises(RuntimeError, match="不能删除"):
        svc.batch_update_history_tasks({
            "action": "delete",
            "task_ids": [task_id],
            "delete_runtime_dirs": True,
            "confirmed": True,
        })
    assert run_dir.exists()

def test_history_batch_route_calls_service():
    class FakeService:
        def __init__(self):
            self.payload = None

        def batch_update_history_tasks(self, payload):
            self.payload = payload
            return {"ok": True, "updated": len(payload["task_ids"])}

    svc = FakeService()
    req = _FakeJsonRequest(
        {"action": "archive", "task_ids": ["a", "b"]},
        {"training_service": svc},
    )

    response = asyncio.run(training_routes.handle_history_batch(req))
    payload = json.loads(response.text)

    assert response.status == 200
    assert payload["updated"] == 2
    assert svc.payload == {"action": "archive", "task_ids": ["a", "b"]}

