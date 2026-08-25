"""History list / summary / collection binding tests."""

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

def test_history_summary_includes_runtime_info(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    svc = TrainingService(web.Application())

    svc._start_history_task(
        job="preprocess",
        variant="522",
        preset="default",
        methods_subdir="imported",
        output_dir="output/runs/522-20260523-114514/training_output",
        sample_dir="output/runs/522-20260523-114514/training_output/sample",
        data_dirs={
            "source_image_dir": "image_dataset/a",
            "resized_image_dir": "output/runs/522-20260523-114514/dataset_cache/dataset-01/resized",
            "lora_cache_dir": "output/runs/522-20260523-114514/dataset_cache/dataset-01/lora",
        },
        sample_config={},
        command=["python", "tasks.py", "preprocess"],
        runtime_info={
            "run_dir": "output/runs/522-20260523-114514",
            "runtime_config_file": "output/runs/522-20260523-114514/config.runtime.toml",
            "original_config_file": "output/runs/522-20260523-114514/config.original.toml",
            "dataset_config_file": "output/runs/522-20260523-114514/dataset.runtime.toml",
            "model_cache_dir": "output/runs/522-20260523-114514/model_cache",
            "dataset_cache_dir": "output/runs/522-20260523-114514/dataset_cache",
            "training_output_dir": "output/runs/522-20260523-114514/training_output",
            "logs_dir": "output/runs/522-20260523-114514/model_cache/logs",
            "history_source_config_file": "configs/imported/522.toml",
        },
    )

    assert svc.list_history_tasks() == []
    task = svc.list_history_tasks(include_archived=True)[0]

    assert task["run_dir"] == "output/runs/522-20260523-114514"
    assert task["project_root_abs"] == str(training_service.ROOT.resolve())
    assert task["run_dir_abs"] == str((training_service.ROOT / "output/runs/522-20260523-114514").resolve())
    assert task["runtime_config_file"].endswith("config.runtime.toml")
    assert task["original_config_file"].endswith("config.original.toml")
    assert task["dataset_config_file"].endswith("dataset.runtime.toml")
    assert task["model_cache_dir"].endswith("model_cache")
    assert task["dataset_cache_dir"].endswith("dataset_cache")
    assert task["training_output_dir"].endswith("training_output")
    assert task["logs_dir"].endswith("model_cache/logs")
    assert task["history_source_config_file"] == "configs/imported/522.toml"
    assert task["history_group_key"] == "source:configs/imported/522.toml"
    assert task["history_group_label"] == "configs/imported/522.toml"
    assert task["history_run_label"] == "522-20260523-114514"
    assert task["archived"] is True
    assert task["name"] == "522-20260523-114514"

def test_history_store_keeps_direct_history_meta_helpers(tmp_path, monkeypatch):
    from web.services.training import history_store as history_store_impl

    history_dir = tmp_path / "history"
    task_id = "20260524-131153-training-imported-522"
    task_dir = _write_group_task(history_dir, task_id, job="training", started_at=1000.0)
    meta = json.loads((task_dir / "meta.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    def _fail_history_task_archived(_task):
        raise AssertionError("history_store should keep direct history_meta helper")

    monkeypatch.setattr(training_service, "_history_task_archived", _fail_history_task_archived)

    summary = history_store_impl._history_summary(meta, task_dir)

    assert summary["id"] == task_id
    assert summary["archived"] is False

def test_history_store_main_paths_do_not_depend_on_bind_legacy(tmp_path, monkeypatch):
    from web.services.training import history_store as history_store_impl

    history_dir = tmp_path / "history"
    task_id = "20260524-131153-training-imported-522"
    _write_group_task(history_dir, task_id, job="training", started_at=1000.0)
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    assert not hasattr(history_store_impl, "_bind_legacy")

    tasks = history_store_impl._list_history_tasks(include_archived=True, limit=0)

    assert [task["id"] for task in tasks] == [task_id]

def test_preprocess_history_summary_archives_legacy_placeholder_by_default(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    _write_group_task(
        history_dir,
        "20260524-131053-preprocess-imported-522",
        job="preprocess",
        started_at=1000.0,
        history_meta={
            "history_run_label": "522-20260524-131053",
            "run_dir": "output/runs/522-20260524-131053",
        },
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    svc = TrainingService(web.Application())
    assert svc.list_history_tasks() == []
    task = svc.list_history_tasks(include_archived=True)[0]

    assert task["archived"] is True
    assert task["name"] == "522-20260524-131053"

def test_preprocess_history_summary_respects_manual_unarchive(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    _write_group_task(
        history_dir,
        "20260524-131053-preprocess-imported-522",
        job="preprocess",
        started_at=1000.0,
        archived=False,
        history_meta={
            "updated_at": 1100.0,
            "history_run_label": "522-20260524-131053",
            "run_dir": "output/runs/522-20260524-131053",
        },
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    task = TrainingService(web.Application()).list_history_tasks()[0]

    assert task["archived"] is False
    assert task["name"] == "522-20260524-131053"

def test_history_list_repairs_legacy_preprocess_archived_flag(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    task_dir = _write_group_task(
        history_dir,
        "20260524-131053-preprocess-imported-522",
        job="preprocess",
        started_at=1000.0,
        archived=False,
        history_meta={
            "history_run_label": "522-20260524-131053",
            "run_dir": "output/runs/522-20260524-131053",
        },
    )
    meta_path = task_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.pop("updated_at", None)
    meta["archived"] = False
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    svc = TrainingService(web.Application())
    assert svc.list_history_tasks() == []
    task = svc.list_history_tasks(include_archived=True)[0]

    assert task["archived"] is True
    repaired = json.loads(meta_path.read_text(encoding="utf-8"))
    assert repaired["archived"] is True

def test_history_list_repairs_legacy_preprocess_name_and_group_meta(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    task_dir = _write_group_task(
        history_dir,
        "20260524-131053-preprocess-imported-522",
        job="preprocess",
        started_at=1000.0,
        archived=False,
        history_meta={
            "run_dir": "output/runs/522-20260524-131053",
            "history_source_config_file": "configs/imported/522.toml",
        },
    )
    meta_path = task_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for key in ("name", "history_group_key", "history_group_label", "history_run_label", "updated_at"):
        meta.pop(key, None)
    meta["archived"] = False
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    svc = TrainingService(web.Application())
    assert svc.list_history_tasks() == []
    task = svc.list_history_tasks(include_archived=True)[0]

    assert task["archived"] is True
    assert task["name"] == "522-20260524-131053"
    assert task["history_run_label"] == "522-20260524-131053"
    assert task["history_group_key"] == "source:configs/imported/522.toml"
    repaired = json.loads(meta_path.read_text(encoding="utf-8"))
    assert repaired["archived"] is True
    assert repaired["name"] == "522-20260524-131053"
    assert repaired["history_run_label"] == "522-20260524-131053"

def test_history_list_repairs_old_auto_prefixed_preprocess_name(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    task_dir = _write_group_task(
        history_dir,
        "20260524-131053-preprocess-imported-522",
        job="preprocess",
        started_at=1000.0,
        archived=True,
        history_meta={
            "name": "预处理 522-20260524-131053",
            "history_run_label": "522-20260524-131053",
            "run_dir": "output/runs/522-20260524-131053",
        },
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    task = TrainingService(web.Application()).list_history_tasks(include_archived=True)[0]

    assert task["name"] == "522-20260524-131053"
    repaired = json.loads((task_dir / "meta.json").read_text(encoding="utf-8"))
    assert repaired["name"] == "522-20260524-131053"

def test_history_list_skips_single_unreadable_summary(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    bad_dir = _write_group_task(
        history_dir,
        "20260524-131053-training-imported-bad",
        started_at=2000.0,
    )
    good_dir = _write_group_task(
        history_dir,
        "20260524-131054-training-imported-good",
        started_at=1000.0,
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    original_count_jsonl = training_service._count_jsonl

    def flaky_count_jsonl(path: Path) -> int:
        if path.parent == bad_dir:
            raise OSError("stale file handle")
        return original_count_jsonl(path)

    monkeypatch.setattr(training_service, "_count_jsonl", flaky_count_jsonl)

    tasks = TrainingService(web.Application()).list_history_tasks(include_archived=True)

    assert [task["id"] for task in tasks] == [good_dir.name]

def test_history_list_keeps_explicit_zero_jsonl_counts(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    task_id = "20260524-131153-training-imported-522"
    _write_group_task(
        history_dir,
        task_id,
        job="training",
        started_at=1000.0,
        history_meta={"log_count": 0, "metric_count": 0},
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    def fail_count_jsonl(path: Path) -> int:
        raise AssertionError(f"explicit zero count should not read {path.name}")

    monkeypatch.setattr(training_service, "_count_jsonl", fail_count_jsonl)

    task = TrainingService(web.Application()).list_history_tasks(include_archived=True)[0]

    assert task["log_count"] == 0
    assert task["metric_count"] == 0

def test_history_list_counts_running_jsonl_even_when_meta_has_zero(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    task_id = "20260524-131154-training-imported-running"
    task_dir = _write_group_task(
        history_dir,
        task_id,
        job="training",
        state="running",
        started_at=1000.0,
        history_meta={"log_count": 0, "metric_count": 0},
    )
    (task_dir / "logs.jsonl").write_text('{"line":"one"}\n{"line":"two"}\n', encoding="utf-8")
    (task_dir / "metrics.jsonl").write_text('{"loss":0.2}\n', encoding="utf-8")
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    task = TrainingService(web.Application()).list_history_tasks(include_archived=True)[0]

    assert task["log_count"] == 2
    assert task["metric_count"] == 1


def test_history_list_includes_bounded_task_row_metric_summary(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    _write_group_task(
        history_dir,
        "20260524-131155-training-imported-summary",
        job="training",
        started_at=1000.0,
        steps=[(step, 0.5 - step * 0.01) for step in range(1, 61)],
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    task = TrainingService(web.Application()).list_history_tasks(include_archived=True)[0]

    assert task["final_loss"] == pytest.approx(-0.1)
    assert task["last_step"] == 60
    assert len(task["loss_preview"]) <= 24
    assert task["loss_preview"][-1] == pytest.approx(-0.1)

def test_history_list_counts_progress_jsonl_when_metrics_missing(tmp_path, monkeypatch):
    """CLI/debug runs often only write progress.jsonl; list should not show 0 loss."""
    history_dir = tmp_path / "history"
    task_id = "20260725-debug-training-imported-progress-only"
    task_dir = _write_group_task(
        history_dir,
        task_id,
        job="training",
        state="interrupted",
        started_at=1000.0,
        # Explicit zero previously pinned metric_count to 0 even with progress data.
        history_meta={"log_count": 0, "metric_count": 0},
    )
    (task_dir / "progress.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ev": "step",
                        "ts": 1.0 + i,
                        "global_step": i + 1,
                        "loss": 0.2 - i * 0.01,
                        "lr": 1e-4,
                    }
                )
                for i in range(3)
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    task = TrainingService(web.Application()).list_history_tasks(include_archived=True)[0]

    assert task["metric_count"] == 3
    assert task["last_step"] == 3
    assert task["final_loss"] == pytest.approx(0.18)
    assert task["loss_preview"] == pytest.approx([0.2, 0.19, 0.18])


def test_history_list_metric_summary_falls_back_from_empty_metrics_to_progress(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    task_dir = _write_group_task(
        history_dir,
        "20260725-debug-training-imported-empty-metrics",
        job="training",
        state="interrupted",
        started_at=1000.0,
        history_meta={"log_count": 0, "metric_count": 0},
    )
    (task_dir / "metrics.jsonl").write_text("", encoding="utf-8")
    (task_dir / "progress.jsonl").write_text(
        json.dumps({"ev": "step", "global_step": 7, "loss": 0.125}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    task = TrainingService(web.Application()).list_history_tasks(include_archived=True)[0]

    assert task["last_step"] == 7
    assert task["final_loss"] == pytest.approx(0.125)

def test_history_list_binds_preprocess_collection_to_training_group(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    history_meta = {
        "history_group_key": "source:configs/imported/522.toml",
        "history_group_label": "configs/imported/522.toml",
        "history_source_config_file": "configs/imported/522.toml",
        "history_run_label": "522-20260524-131053",
    }
    preprocess_dir = _write_group_task(
        history_dir,
        "20260524-131053-preprocess-imported-522",
        job="preprocess",
        started_at=1000.0,
        archived=True,
        history_meta={**history_meta, "group": ""},
    )
    _write_group_task(
        history_dir,
        "20260524-131153-training-imported-522",
        job="training",
        started_at=1010.0,
        history_meta={**history_meta, "group": "骨女测试集合", "updated_at": 1200.0},
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    tasks = TrainingService(web.Application()).list_history_tasks(include_archived=True)

    assert {task["group"] for task in tasks} == {"骨女测试集合"}
    repaired = json.loads((preprocess_dir / "meta.json").read_text(encoding="utf-8"))
    assert repaired["group"] == "骨女测试集合"

def test_history_collection_binding_is_idempotent_and_uses_atomic_write(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    history_meta = {
        "history_group_key": "source:configs/imported/522.toml",
        "history_group_label": "configs/imported/522.toml",
        "history_source_config_file": "configs/imported/522.toml",
        "history_run_label": "522-20260524-131053",
    }
    preprocess_dir = _write_group_task(
        history_dir,
        "20260524-131053-preprocess-imported-522",
        job="preprocess",
        started_at=1000.0,
        archived=True,
        history_meta={**history_meta, "group": ""},
    )
    _write_group_task(
        history_dir,
        "20260524-131153-training-imported-522",
        job="training",
        started_at=1010.0,
        history_meta={**history_meta, "group": "正式集合", "updated_at": 1200.0},
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    writes = []
    original_atomic = training_service._write_json_atomic

    def record_atomic(path, payload):
        writes.append(Path(path))
        original_atomic(path, payload)

    monkeypatch.setattr(training_service, "_write_json_atomic", record_atomic)

    assert training_service._sync_bound_history_collection_groups() == 1
    assert writes == [preprocess_dir / "meta.json"]
    assert json.loads((preprocess_dir / "meta.json").read_text(encoding="utf-8"))["group"] == "正式集合"

    writes.clear()
    assert training_service._sync_bound_history_collection_groups() == 0
    assert writes == []

def test_setting_collection_expands_to_bound_preprocess_tasks(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    history_meta = {
        "history_group_key": "source:configs/imported/522.toml",
        "history_group_label": "configs/imported/522.toml",
        "history_source_config_file": "configs/imported/522.toml",
        "history_run_label": "522-20260524-131053",
    }
    preprocess_id = "20260524-131053-preprocess-imported-522"
    training_id = "20260524-131153-training-imported-522"
    preprocess_dir = _write_group_task(
        history_dir,
        preprocess_id,
        job="preprocess",
        started_at=1000.0,
        archived=True,
        history_meta=history_meta,
    )
    training_dir = _write_group_task(
        history_dir,
        training_id,
        job="training",
        started_at=1010.0,
        history_meta=history_meta,
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    result = TrainingService(web.Application()).batch_update_history_tasks({
        "action": "set_group",
        "task_ids": [training_id],
        "group": "同配置集合",
    })

    assert result["ok"] is True
    assert result["requested"] == 1
    assert result["updated"] == 2
    assert json.loads((training_dir / "meta.json").read_text(encoding="utf-8"))["group"] == "同配置集合"
    assert json.loads((preprocess_dir / "meta.json").read_text(encoding="utf-8"))["group"] == "同配置集合"

def test_history_detail_limits_logs_and_system_records(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    task_id = "20260524-131153-training-imported-522"
    task_dir = _write_group_task(history_dir, task_id, job="training", started_at=1000.0)
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    monkeypatch.setattr(training_service, "MAX_HISTORY_DETAIL_LOG_RECORDS", 3)
    monkeypatch.setattr(training_service, "MAX_HISTORY_DETAIL_SYSTEM_RECORDS", 2)
    (task_dir / "logs.jsonl").write_text(
        "\n".join(json.dumps({"id": idx, "line": f"log-{idx}"}) for idx in range(5)) + "\n",
        encoding="utf-8",
    )
    (task_dir / "system.jsonl").write_text(
        "\n".join(json.dumps({"ts": idx, "gpu_util": idx * 10}) for idx in range(4)) + "\n",
        encoding="utf-8",
    )

    payload = TrainingService(web.Application()).get_history_task(task_id)

    assert [item["line"] for item in payload["logs"]] == ["log-2", "log-3", "log-4"]
    assert [item["ts"] for item in payload["system"]] == [2, 3]
    assert payload["limits"]["logs_total"] == 5
    assert payload["limits"]["logs_returned"] == 3
    assert payload["limits"]["logs_truncated"] is True
    assert payload["limits"]["system_total"] == 4
    assert payload["limits"]["system_returned"] == 2
    assert payload["limits"]["system_truncated"] is True


def test_history_log_pages_cover_records_omitted_from_detail(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    task_id = "20260524-131153-training-imported-522"
    task_dir = _write_group_task(history_dir, task_id, job="training", started_at=1000.0)
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    monkeypatch.setattr(training_service, "MAX_HISTORY_DETAIL_LOG_RECORDS", 3)
    (task_dir / "logs.jsonl").write_text(
        "\n".join(json.dumps({"id": idx, "line": f"log-{idx}"}) for idx in range(8)) + "\n",
        encoding="utf-8",
    )
    service = TrainingService(web.Application())

    detail = service.get_history_task(task_id)
    first = service.get_history_log_page(task_id, offset=0, limit=3)
    middle = service.get_history_log_page(task_id, offset=3, limit=3)
    tail = service.get_history_log_page(task_id, limit=3)

    assert [item["line"] for item in detail["logs"]] == ["log-5", "log-6", "log-7"]
    assert [item["line"] for item in first["logs"]] == ["log-0", "log-1", "log-2"]
    assert [item["line"] for item in middle["logs"]] == ["log-3", "log-4", "log-5"]
    assert [item["line"] for item in tail["logs"]] == ["log-5", "log-6", "log-7"]
    assert first == {
        "ok": True,
        "logs": first["logs"],
        "offset": 0,
        "limit": 3,
        "returned": 3,
        "total": 8,
        "next_offset": 3,
        "has_more_before": False,
        "has_more_after": True,
    }
    assert tail["offset"] == 5
    assert tail["has_more_before"] is True
    assert tail["has_more_after"] is False
    middle_match = service.find_history_log_match(task_id, query="LOG-", cursor=4)
    wrapped_match = service.find_history_log_match(task_id, query="log-", cursor=8)
    previous_match = service.find_history_log_match(task_id, query="log-", cursor=2, direction="backward")
    assert (middle_match["match_index"], middle_match["match_ordinal"], middle_match["matches_total"]) == (4, 5, 8)
    assert (wrapped_match["match_index"], wrapped_match["match_ordinal"]) == (0, 1)
    assert (previous_match["match_index"], previous_match["match_ordinal"]) == (2, 3)
    (task_dir / "logs.jsonl").unlink()
    assert service.get_history_log_page(task_id, limit=3)["logs"] == []
    assert service.find_history_log_match(task_id, query="log")["match"] is None


def test_history_module_keeps_direct_history_meta_helpers(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    task_id = "20260524-131153-training-imported-522"
    task_dir = history_dir / task_id
    task_dir.mkdir(parents=True)
    snapshot = task_dir / "config.snapshot.toml"
    snapshot.write_text("max_train_epochs = 1\n", encoding="utf-8")
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    def _fail_history_artifact_path(*_args, **_kwargs):
        raise AssertionError("history module should keep direct history_meta helper")

    monkeypatch.setattr(training_service, "_history_artifact_path", _fail_history_artifact_path)

    svc = TrainingService(web.Application())

    assert svc.get_history_artifact_path(task_id, "config-snapshot") == snapshot.resolve()

def test_history_detail_exposes_linked_preprocess_task(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    run_dir = tmp_path / "runs" / "524-20260524-225059"
    other_run_dir = tmp_path / "runs" / "524-20260524-230000"
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
    _write_group_task(history_dir, training_id, job="training", started_at=1000.0, history_meta=history_meta)
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
        started_at=1200.0,
        archived=True,
        history_meta={
            **history_meta,
            "run_dir": str(other_run_dir),
            "training_output_dir": str(other_run_dir / "training_output"),
            "history_run_label": other_run_dir.name,
        },
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    svc = TrainingService(web.Application())

    payload = svc.get_history_task(training_id)
    linked = payload["task"]["linked_preprocess_task"]

    assert linked["id"] == preprocess_id
    assert linked["job"] == "preprocess"
    assert linked["archived"] is True
    assert linked["history_run_label"] == run_dir.name
    assert linked["id"] != other_preprocess_id
    assert "linked_preprocess_task" not in svc.get_history_task(preprocess_id)["task"]

def test_history_collection_settings_round_trip_and_normalize(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    monkeypatch.setattr(training_service, "HISTORY_COLLECTIONS_FILE", history_dir / "collections.json")
    svc = TrainingService(web.Application())

    empty = svc.get_history_collection_settings()
    assert empty["ok"] is True
    assert empty["collection_order"] == []
    assert empty["config_group_order"] == {}

    saved = svc.save_history_collection_settings({
        "collection_order": ["B", "", "A", "B", "  C  "],
        "config_group_order": {
            "A": ["g2", "g1", "g2", ""],
            "": ["bad"],
            "B": "not-list",
        },
    })

    assert saved["collection_order"] == ["B", "A", "C"]
    assert saved["config_group_order"] == {"A": ["g2", "g1"]}
    assert (history_dir / "collections.json").exists()
    loaded = svc.get_history_collection_settings()
    assert loaded["collection_order"] == ["B", "A", "C"]
    assert loaded["config_group_order"] == {"A": ["g2", "g1"]}

def test_history_collection_settings_routes(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    monkeypatch.setattr(training_service, "HISTORY_COLLECTIONS_FILE", history_dir / "collections.json")
    svc = TrainingService(web.Application())

    put_req = _FakeJsonRequest(
        {"collection_order": ["正式训练"], "config_group_order": {"正式训练": ["config-a"]}},
        {"training_service": svc},
    )
    put_response = asyncio.run(training_routes.handle_history_collection_settings_put(put_req))
    put_payload = json.loads(put_response.text)
    assert put_response.status == 200
    assert put_payload["collection_order"] == ["正式训练"]

    get_req = _FakeJsonRequest({}, {"training_service": svc})
    get_response = asyncio.run(training_routes.handle_history_collection_settings_get(get_req))
    get_payload = json.loads(get_response.text)
    assert get_response.status == 200
    assert get_payload["config_group_order"] == {"正式训练": ["config-a"]}


def test_training_service_root_points_at_repo_root() -> None:
    """Regression: ROOT must be project root so subprocess cwd can find tasks.py."""
    from web.services.training import constants as training_constants

    root = training_service.ROOT.resolve()
    constants_root = training_constants.ROOT.resolve()
    assert root == constants_root
    assert (root / "tasks.py").is_file()
    assert (root / "train.py").is_file()
    assert root.name != "web"
    assert root == Path(__file__).resolve().parents[1]


def test_history_summary_includes_config_chip_fields_from_snapshot(tmp_path, monkeypatch):
    from web.services.training import history_store as history_store_impl

    history_dir = tmp_path / "history"
    task_id = "20260727-chip-training-imported-demo"
    snapshot = "\n".join(
        [
            "network_module = \"networks.lora_anima\"",
            "use_lokr = true",
            'preprocess_precision_preference = "bf16"',
            'block_swap_transfer_dtype = "fp8_e4m3"',
            'mixed_precision = "bf16"',
            'base_compute = "w8a16_convrot"',
        ]
    ) + "\n"
    task_dir = _write_group_task(
        history_dir,
        task_id,
        job="training",
        variant="okkotsu_goddess_demo",
        started_at=2000.0,
        config_text=snapshot,
    )
    meta = json.loads((task_dir / "meta.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    summary = history_store_impl._history_summary(meta, task_dir)

    assert summary["training_variant"] == "lokr"
    assert summary["preprocess_precision"] == "bf16"
    assert summary["block_swap_precision"] == "fp8_e4m3"
    assert summary["base_compute"] == "w8a16_convrot"
    assert summary["precision_preference"] == "bf16"
    # 配置 stem 仍是 variant，不能被 chip 覆盖
    assert summary["variant"] == "okkotsu_goddess_demo"


def test_history_summary_config_chips_empty_without_snapshot(tmp_path, monkeypatch):
    from web.services.training import history_store as history_store_impl

    history_dir = tmp_path / "history"
    task_id = "20260727-chip-nosnap"
    task_dir = _write_group_task(history_dir, task_id, job="training", started_at=2100.0)
    (task_dir / "config.snapshot.toml").unlink()
    meta = json.loads((task_dir / "meta.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    summary = history_store_impl._history_summary(meta, task_dir)

    assert summary["training_variant"] == ""
    assert summary["preprocess_precision"] == ""
    assert summary["block_swap_precision"] == ""
    assert summary["base_compute"] == ""
    assert summary["precision_preference"] == ""


def test_history_config_chips_hydralora_and_tlora_from_text():
    from web.services.training.history_config_chips import history_config_chips_from_snapshot_text

    hydra = history_config_chips_from_snapshot_text(
        'use_moe_style = "shared_A"\nnetwork_module = "networks.lora_anima"\n',
        variant="whatever",
    )
    assert hydra["training_variant"] == "hydralora"

    tlora = history_config_chips_from_snapshot_text(
        "use_timestep_mask = true\nnetwork_module = \"networks.lora_anima\"\n",
        variant="tlora-8gb",
    )
    assert tlora["training_variant"] == "tlora"

    chimera = history_config_chips_from_snapshot_text(
        "use_chimera_hydra = true\nuse_moe_style = \"shared_A\"\n",
        variant="x",
    )
    assert chimera["training_variant"] == "chimera"


def test_history_config_chips_base_compute_and_precision_preference_from_text():
    from web.services.training.history_config_chips import history_config_chips_from_snapshot_text

    chips = history_config_chips_from_snapshot_text(
        "\n".join(
            [
                'base_compute = "w8a8_convrot"',
                'mixed_precision = "no"',
                'preprocess_precision_preference = "fp16"',
            ]
        )
        + "\n",
        variant="lora",
    )
    assert chips["base_compute"] == "w8a8_convrot"
    assert chips["precision_preference"] == "fp32"
    assert chips["preprocess_precision"] == "fp16"

    fp16 = history_config_chips_from_snapshot_text(
        'mixed_precision = "fp16"\nfull_fp16 = true\n',
        variant="lora",
    )
    assert fp16["precision_preference"] == "fp16"
    assert fp16["base_compute"] == ""


def test_service_startup_marks_orphaned_running_tasks_interrupted(
    tmp_path, monkeypatch
):
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

    metadata = json.loads((task_dir / "meta.json").read_text(encoding="utf-8"))
    assert metadata["state"] == "interrupted"
    assert "中断" in metadata["message"]
    assert metadata["finished_at"] == 1002.0
    assert metadata["log_count"] == 2
    assert metadata["metric_count"] == 2


def test_service_startup_keeps_history_available_when_orphan_repair_write_fails(
    tmp_path, monkeypatch
):
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

    monkeypatch.setattr(
        training_service, "_write_json_atomic", flaky_write_json_atomic
    )

    count = training_service._mark_orphaned_running_history_tasks()

    assert count == 1
    bad_metadata = json.loads((bad_dir / "meta.json").read_text(encoding="utf-8"))
    good_metadata = json.loads((good_dir / "meta.json").read_text(encoding="utf-8"))
    assert bad_metadata["state"] == "running"
    assert good_metadata["state"] == "interrupted"

    service = TrainingService(web.Application())
    payload = service.get_config_group_timeline("imported", "demo", "default")
    tasks = {task["id"]: task for task in payload["tasks"]}
    assert tasks[bad_dir.name]["state"] == "running"
    assert tasks[good_dir.name]["state"] == "interrupted"
