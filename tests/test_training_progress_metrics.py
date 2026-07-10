"""Progress JSONL / rate / metrics tests split from test_training_resume.py."""

from __future__ import annotations

from tests import training_resume_test_support as _training_resume_support

globals().update(
    {
        name: value
        for name, value in vars(_training_resume_support).items()
        if name == "Path" or not name.startswith("__")
    }
)

# Split: progress_metrics

def test_training_service_ingests_progress_jsonl(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    task_dir = history_dir / "task"
    task_dir.mkdir(parents=True)
    (task_dir / "meta.json").write_text(
        json.dumps({"id": "task", "started_at": 1000.0}),
        encoding="utf-8",
    )
    progress_path = task_dir / "progress.jsonl"
    events = [
        {"ev": "run_start", "ts": 0.0, "total_steps": 10, "total_epochs": 1, "pid": 1},
        {"ev": "step", "ts": 1.0, "global_step": 1, "epoch": 0, "loss": 0.5, "lr": 1e-4},
        {"ev": "step", "ts": 1.5, "global_step": 2, "epoch": 0, "loss/average": 0.4, "lr/unet": 2e-4},
        {"ev": "val", "ts": 2.0, "global_step": 1, "epoch": 0, "cmmd": 0.03},
        {"ev": "ckpt", "ts": 3.0, "global_step": 1, "path": "output/demo.safetensors"},
        {"ev": "run_end", "ts": 4.0, "status": "ok", "final_step": 1},
    ]
    progress_path.write_text(
        "\n".join(json.dumps(item) for item in events) + "\n",
        encoding="utf-8",
    )

    svc = TrainingService(web.Application())
    svc.current_task_dir = task_dir
    svc.current_task_id = "task"
    svc._progress_jsonl_path = progress_path

    async def ingest():
        svc._progress_jsonl_lock = asyncio.Lock()
        await svc._ingest_progress_jsonl(final=True)

    asyncio.run(ingest())

    metrics = [
        json.loads(line)
        for line in (task_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    logs = [
        json.loads(line)
        for line in (task_dir / "logs.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert metrics[0]["step"] == 1
    assert metrics[0]["loss"] == 0.5
    assert metrics[0]["ts"] == 1001.0
    assert metrics[1]["step"] == 2
    assert metrics[1]["loss"] == 0.4
    assert metrics[1]["lr"] == 2e-4
    assert metrics[2]["kind"] == "val"
    assert metrics[2]["cmmd"] == 0.03
    assert any("结构化训练进度已开始" in item["line"] for item in logs)
    assert any("已保存检查点" in item["line"] for item in logs)
    assert any("结构化训练进度结束" in item["line"] for item in logs)
    snapshot = svc.get_status_snapshot()
    assert snapshot["latest_progress"]["current"] == 2
    assert snapshot["latest_progress"]["total"] == 10
    assert snapshot["latest_progress"]["label"] == "Training"
    assert snapshot["latest_metric"]["kind"] == "val"
    assert snapshot["latest_metric"]["cmmd"] == 0.03

def test_history_detail_recovers_lr_from_progress_jsonl(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    task_dir = history_dir / "task"
    task_dir.mkdir(parents=True)
    (task_dir / "meta.json").write_text(
        json.dumps({
            "id": "task",
            "job": "training",
            "state": "idle",
            "started_at": 1000.0,
        }),
        encoding="utf-8",
    )
    (task_dir / "config.snapshot.toml").write_text("learning_rate = 0.001\n", encoding="utf-8")
    (task_dir / "metrics.jsonl").write_text(
        "\n".join([
            json.dumps({"step": 2, "ts": 1001.0, "loss": 0.5}),
            json.dumps({"step": 3, "ts": 1002.0, "loss": 0.4}),
        ]) + "\n",
        encoding="utf-8",
    )
    (task_dir / "progress.jsonl").write_text(
        "\n".join([
            json.dumps({"ev": "run_start", "ts": 0.0, "total_steps": 2, "total_epochs": 1, "pid": 1}),
            json.dumps({"ev": "step", "ts": 1.0, "global_step": 1, "epoch": 0, "loss/average": 0.5, "lr/unet": 1e-4}),
            json.dumps({"ev": "step", "ts": 2.0, "global_step": 2, "epoch": 0, "loss/average": 0.4, "lr/group0": 2e-4}),
        ]) + "\n",
        encoding="utf-8",
    )

    payload = training_service._load_history_task("task")

    assert [item["step"] for item in payload["metrics"]] == [1, 2]
    assert [item["loss"] for item in payload["metrics"]] == [0.5, 0.4]
    assert [item["lr"] for item in payload["metrics"]] == [1e-4, 2e-4]

def test_training_service_persists_learning_rate_change_logs(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    task_dir = history_dir / "task"
    task_dir.mkdir(parents=True)

    svc = TrainingService(web.Application())
    svc.current_task_dir = task_dir

    async def record_metrics():
        await svc._record_metric({"step": 1, "lr": 1e-4, "ts": 1001.0})
        await svc._record_metric({"step": 2, "lr": 1.00001e-4, "ts": 1002.0})
        await svc._record_metric({"step": 3, "lr": 8.5e-5, "ts": 1003.0})
        await svc._record_metric({"step": 4, "loss": 0.4, "ts": 1004.0})

    asyncio.run(record_metrics())
    assert svc.get_status_snapshot()["latest_metric"]["step"] == 4

    logs = [
        json.loads(line)
        for line in (task_dir / "logs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    metric_logs = [item for item in logs if item.get("kind") == "metric"]

    assert [item["line"] for item in metric_logs] == [
        "[学习率] step 1: 1.00e-04",
        "[学习率] step 3: 1.00e-04 → 8.50e-05",
    ]
    assert [item["ts"] for item in metric_logs] == [1001.0, 1003.0]

def test_training_service_metric_runtime_reset_clears_learning_rate_log_state():
    svc = TrainingService(web.Application())
    svc._metrics_history = [{"step": 1, "lr": 1e-4}]
    svc._metric_seen_keys = {("demo",)}
    svc._last_lr_log_text = "1.00e-04"

    svc._reset_metric_runtime_state()

    assert svc._metrics_history == []
    assert svc._metric_seen_keys == set()
    assert svc._last_lr_log_text == ""

def test_training_service_rate_uses_recent_step_median(monkeypatch):
    svc = TrainingService(web.Application())
    ticks = iter([0.0, 8.0, 16.0, 24.0, 146.0, 154.0, 162.0])
    monkeypatch.setattr(training_service.time, "monotonic", lambda: next(ticks))

    assert svc._compute_rate(1, 100) == ""
    assert svc._compute_rate(2, 100) == "8.00s/step"
    assert svc._compute_rate(3, 100) == "8.00s/step"
    assert svc._compute_rate(4, 100) == "8.00s/step"
    # 单次长暂停不应把“当前速度”拖成全局平均值。
    assert svc._compute_rate(5, 100) == "8.00s/step"
    assert svc._compute_rate(6, 100) == "8.00s/step"
    assert svc._compute_rate(7, 100) == "8.00s/step"

def test_progress_parser_handles_structured_step_and_val_events():
    step_metric = progress_parser.metric_from_progress_jsonl_event(
        {
            "ev": "step",
            "global_step": "12",
            "epoch": "2",
            "loss/average": "0.123",
            "lr/group0": "1e-4",
        },
        1234.5,
        rate="1.50s/step",
    )
    val_metric = progress_parser.metric_from_progress_jsonl_event(
        {"ev": "val", "global_step": 12, "val_step": "3", "cmmd": "0.42"},
        1236.0,
    )

    assert step_metric == {
        "ts": 1234.5,
        "step": 12,
        "epoch": 2,
        "rate": "1.50s/step",
        "loss": 0.123,
        "lr": 1e-4,
    }
    assert val_metric == {
        "ts": 1236.0,
        "step": 12,
        "kind": "val",
        "cmmd": 0.42,
        "loss": 0.42,
        "val_step": 3,
    }
    assert training_service._metric_from_progress_jsonl_event(
        {"ev": "step", "global_step": "12", "loss/current": "0.2"},
        1.0,
    ) == {"ts": 1.0, "step": 12, "loss": 0.2}

def test_progress_jsonl_metrics_derive_recent_step_rate(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    task_dir = history_dir / "task"
    task_dir.mkdir(parents=True)
    (task_dir / "meta.json").write_text(json.dumps({"started_at": 1000.0}), encoding="utf-8")
    (task_dir / "progress.jsonl").write_text(
        "\n".join([
            json.dumps({"ev": "step", "ts": 1.0, "global_step": 1, "loss": 0.5, "lr": 1e-4}),
            json.dumps({"ev": "step", "ts": 9.0, "global_step": 2, "loss": 0.4, "lr": 1e-4}),
            json.dumps({"ev": "step", "ts": 17.0, "global_step": 3, "loss": 0.3, "lr": 1e-4}),
            json.dumps({"ev": "step", "ts": 139.0, "global_step": 4, "loss": 0.2, "lr": 1e-4}),
            json.dumps({"ev": "step", "ts": 147.0, "global_step": 5, "loss": 0.1, "lr": 1e-4}),
        ]) + "\n",
        encoding="utf-8",
    )

    metrics = training_service._metrics_from_progress_jsonl(task_dir / "progress.jsonl", task_dir)

    assert [item.get("rate") for item in metrics] == [
        None,
        "8.00s/step",
        "8.00s/step",
        "8.00s/step",
        "8.00s/step",
    ]

def test_history_detail_persists_average_speed_from_logs_once(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    task_dir = history_dir / "task"
    task_dir.mkdir(parents=True)
    (task_dir / "meta.json").write_text(
        json.dumps({
            "id": "task",
            "job": "training",
            "state": "idle",
            "started_at": 1000.0,
            "finished_at": 1040.0,
        }),
        encoding="utf-8",
    )
    (task_dir / "logs.jsonl").write_text(
        "\n".join([
            json.dumps({"id": 1, "kind": "progress", "line": "steps:   0%| | 0/100 [00:00<?, ?it/s]", "ts": 1000.0}),
            json.dumps({"id": 2, "kind": "progress", "line": "steps:   1%| | 1/100 [00:10<00:00, 10.00s/it, avr_loss=0.5]", "ts": 1010.0}),
            json.dumps({"id": 3, "kind": "progress", "line": "steps:   2%| | 2/100 [00:20<00:00, 10.00s/it, avr_loss=0.4]", "ts": 1020.0}),
            json.dumps({"id": 4, "kind": "progress", "line": "steps:   3%| | 3/100 [00:30<00:00, 10.00s/it, avr_loss=0.3]", "ts": 1030.0}),
        ]) + "\n",
        encoding="utf-8",
    )

    payload = training_service._load_history_task("task")
    task = payload["task"]

    assert task["average_step_rate"] == "10.00s/step"
    assert task["average_step_seconds"] == 10.0
    assert task["average_step_source"] == "logs.jsonl"
    assert task["average_step_sample_count"] == 4
    assert task["average_step_start_step"] == 0
    assert task["average_step_end_step"] == 3

    persisted = json.loads((task_dir / "meta.json").read_text(encoding="utf-8"))
    assert persisted["average_step_rate"] == "10.00s/step"
    assert persisted["average_step_speed_version"] == training_service.HISTORY_AVERAGE_SPEED_VERSION

    (task_dir / "logs.jsonl").write_text(
        "\n".join([
            json.dumps({"id": 1, "kind": "progress", "line": "steps:   0%| | 0/100 [00:00<?, ?it/s]", "ts": 1000.0}),
            json.dumps({"id": 2, "kind": "progress", "line": "steps:   3%| | 3/100 [01:00<00:00, 20.00s/it, avr_loss=0.3]", "ts": 1060.0}),
        ]) + "\n",
        encoding="utf-8",
    )

    payload_after_log_change = training_service._load_history_task("task")

    assert payload_after_log_change["task"]["average_step_rate"] == "10.00s/step"

def test_progress_jsonl_oom_event_records_hint():
    svc = TrainingService(web.Application())

    asyncio.run(
        svc._handle_progress_jsonl_event({
            "ev": "run_end",
            "status": "error",
            "final_step": 0,
            "error": "OutOfMemoryError: CUDA out of memory.",
        })
    )

    lines = [item["line"] for item in svc.get_log_records()]
    assert "大概率爆显存" in lines
    assert any(
        "结构化训练进度结束" in line and "大概率爆显存" in line
        for line in lines
    )
    assert svc.get_status_snapshot()["error_hint"] == "大概率爆显存"

def test_extract_metrics_from_log_accepts_nan_and_infinity_loss():
    svc = TrainingService(web.Application())

    metric = svc._extract_metrics_from_log("step=0 loss=NaN learning_rate=Infinity")

    assert metric is not None
    assert metric["step"] == 0
    assert metric["loss"] != metric["loss"]
    assert metric["lr"] == float("inf")

