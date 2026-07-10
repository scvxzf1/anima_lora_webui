"""Config-group timeline tests."""

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

def test_config_group_timeline_can_select_history_group_key(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    source_meta = {
        "history_group_key": "source:configs/imported/demo.toml",
        "history_group_label": "configs/imported/demo.toml",
        "history_source_config_file": "configs/imported/demo.toml",
        "history_run_label": "demo-20260523-114514",
        "run_dir": "output/runs/demo-20260523-114514",
    }
    _write_group_task(
        history_dir,
        "20260517-000001-training-imported-demo",
        started_at=1000.0,
        steps=[(1, 0.3)],
        history_meta=source_meta,
    )
    _write_group_task(
        history_dir,
        "20260517-000002-training-imported-demo",
        started_at=2000.0,
        steps=[(1, 0.2)],
        history_meta={
            **source_meta,
            "history_run_label": "demo-20260523-120000",
            "run_dir": "output/runs/demo-20260523-120000",
        },
    )
    _write_group_task(
        history_dir,
        "20260517-000003-training-imported-demo",
        started_at=3000.0,
        steps=[(1, 0.9)],
        history_meta={
            "history_group_key": "source:configs/imported/other.toml",
            "history_group_label": "configs/imported/other.toml",
            "history_source_config_file": "configs/imported/other.toml",
            "history_run_label": "other-20260523-120000",
            "run_dir": "output/runs/other-20260523-120000",
        },
    )
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    svc = TrainingService(web.Application())
    payload = svc.get_config_group_timeline(
        "",
        "",
        "default",
        group_key="source:configs/imported/demo.toml",
    )

    assert payload["summary"]["task_count"] == 2
    assert payload["summary"]["group_count"] == 1
    assert payload["group"]["history_group_key"] == "source:configs/imported/demo.toml"
    assert payload["group"]["history_source_config_file"] == "configs/imported/demo.toml"
    assert [task["history_run_label"] for task in payload["tasks"]] == [
        "demo-20260523-114514",
        "demo-20260523-120000",
    ]

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

