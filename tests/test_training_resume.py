from __future__ import annotations

import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace

import toml
import torch
from aiohttp import web
from PIL import Image
from safetensors.torch import save_file

from library.training.checkpoints import CheckpointSaver, save_checkpoint_state
from web.routes import training as training_routes
from web.services import config_service, settings_service, training_service
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
    history_meta=None,
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
    if history_meta:
        meta.update(history_meta)
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


def _write_runtime_config_tree(root):
    configs = root / "configs"
    imported = configs / "imported"
    datasets = configs / "datasets"
    imported.mkdir(parents=True)
    datasets.mkdir(parents=True)
    (configs / "base.toml").write_text(
        "\n".join(
            [
                'source_image_dir = "image_dataset/fallback"',
                'resized_image_dir = "post_image_dataset/resized"',
                'lora_cache_dir = "post_image_dataset/lora"',
                'output_dir = "legacy/output"',
                'logging_dir = "legacy/logs"',
                'output_name = "demo"',
                'dataset_config = "configs/datasets/522.toml"',
            ]
        ),
        encoding="utf-8",
    )
    (configs / "presets.toml").write_text("[default]\n", encoding="utf-8")
    (imported / "522.toml").write_text(
        "\n".join(
            [
                'dataset_config = "configs/datasets/522.toml"',
                'output_dir = "legacy/from-toml"',
                'logging_dir = "legacy/logs"',
                'output_name = "522-demo"',
            ]
        ),
        encoding="utf-8",
    )
    (datasets / "522.toml").write_text(
        "\n".join(
            [
                "[[datasets]]",
                "resolution = 768",
                "",
                "[[datasets.subsets]]",
                'image_dir = "old/a_resized"',
                'cache_dir = "old/a_lora"',
                'custom_attributes = {source_dir = "image_dataset/a"}',
                "num_repeats = 2",
                "",
                "[[datasets]]",
                "resolution = 1024",
                "",
                "[[datasets.subsets]]",
                'image_dir = "old/b_resized"',
                'cache_dir = "old/b_lora"',
                'custom_attributes = {source_dir = "image_dataset/b"}',
                "num_repeats = 1",
            ]
        ),
        encoding="utf-8",
    )


def _patch_runtime_service_paths(monkeypatch, root):
    configs = root / "configs"
    monkeypatch.setattr(config_service, "ROOT", root)
    monkeypatch.setattr(config_service, "CONFIGS_DIR", configs)
    monkeypatch.setattr(config_service, "GUI_METHODS_DIR", configs / "gui-methods")
    monkeypatch.setattr(config_service, "IMPORTED_CONFIGS_DIR", configs / "imported")
    monkeypatch.setattr(config_service, "PRESETS_FILE", configs / "presets.toml")
    monkeypatch.setattr(config_service, "DATASET_PRESETS_DIR", configs / "datasets")
    monkeypatch.setattr(training_service, "ROOT", root)
    monkeypatch.setattr(settings_service, "ROOT", root)
    monkeypatch.setattr(settings_service, "SETTINGS_FILE", configs / "web-ui-settings.toml")


class _FakeJsonRequest:
    def __init__(self, data, app=None):
        self._data = data
        self.app = app or {}

    async def json(self):
        return self._data


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
    assert captured["use_runtime_dir"] is False
    assert captured["resume_info"]["checkpoint"] == str(state_dir)
    assert captured["resume_info"]["history_group_key"] == "legacy:imported\u0001demo\u0001default"
    assert captured["resume_info"]["history_group_label"] == "imported / demo / default"


def test_web_runtime_config_creates_run_directory_and_overrides_paths(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)

    class FixedDatetime:
        @classmethod
        def now(cls):
            return datetime(2026, 5, 23, 11, 45, 14)

        @classmethod
        def fromtimestamp(cls, value):
            return datetime.fromtimestamp(value)

    monkeypatch.setattr(training_service, "datetime", FixedDatetime)

    runtime = training_service._prepare_web_runtime_config(
        "522",
        "default",
        "imported",
        source_config_file="configs/imported/522.toml",
    )

    run_dir = tmp_path / "output" / "runs" / "522-20260523-114514"
    assert runtime["run_dir"] == "output/runs/522-20260523-114514"
    assert (run_dir / "config.original.toml").read_text(encoding="utf-8") == (
        tmp_path / "configs" / "imported" / "522.toml"
    ).read_text(encoding="utf-8")
    assert (run_dir / "model_cache" / "logs").is_dir()
    assert (run_dir / "training_output" / "sample").is_dir()
    assert (run_dir / "dataset_cache" / "dataset-01" / "resized").is_dir()
    assert (run_dir / "dataset_cache" / "dataset-02" / "lora").is_dir()
    run_meta = json.loads((run_dir / "run.meta.json").read_text(encoding="utf-8"))
    assert run_meta["history_source_config_file"] == "configs/imported/522.toml"
    assert run_meta["runtime_config_file"] == "output/runs/522-20260523-114514/config.runtime.toml"

    runtime_cfg = toml.loads((run_dir / "config.runtime.toml").read_text(encoding="utf-8"))
    assert runtime_cfg["output_dir"] == "output/runs/522-20260523-114514/training_output"
    assert runtime_cfg["logging_dir"] == "output/runs/522-20260523-114514/model_cache/logs"
    assert runtime_cfg["dataset_config"] == "output/runs/522-20260523-114514/dataset.runtime.toml"
    assert runtime_cfg["source_image_dir"] == "image_dataset/a"
    assert runtime_cfg["resized_image_dir"] == "output/runs/522-20260523-114514/dataset_cache/dataset-01/resized"
    assert runtime_cfg["lora_cache_dir"] == "output/runs/522-20260523-114514/dataset_cache/dataset-01/lora"

    dataset_cfg = toml.loads((run_dir / "dataset.runtime.toml").read_text(encoding="utf-8"))
    first_subset = dataset_cfg["datasets"][0]["subsets"][0]
    second_subset = dataset_cfg["datasets"][1]["subsets"][0]
    assert first_subset["custom_attributes"]["source_dir"] == "image_dataset/a"
    assert first_subset["image_dir"].endswith("dataset-01/resized")
    assert first_subset["cache_dir"].endswith("dataset-01/lora")
    assert second_subset["custom_attributes"]["source_dir"] == "image_dataset/b"
    assert second_subset["image_dir"].endswith("dataset-02/resized")
    assert second_subset["cache_dir"].endswith("dataset-02/lora")

    env = {}
    training_service._apply_runtime_env(env, runtime)
    assert env["ANIMA_RUNTIME_CONFIG"] == "output/runs/522-20260523-114514/config.runtime.toml"
    assert env["TORCHINDUCTOR_CACHE_DIR"].endswith("model_cache/torchinductor")
    assert env["TRITON_CACHE_DIR"].endswith("model_cache/triton")


def test_runtime_config_recovers_source_group_from_run_meta(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)

    class FixedDatetime:
        @classmethod
        def now(cls):
            return datetime(2026, 5, 23, 11, 45, 14)

        @classmethod
        def fromtimestamp(cls, value):
            return datetime.fromtimestamp(value)

    monkeypatch.setattr(training_service, "datetime", FixedDatetime)
    runtime = training_service._prepare_web_runtime_config(
        "522",
        "default",
        "imported",
        source_config_file="configs/imported/522.toml",
    )

    recovered = training_service._runtime_from_config_file(runtime["runtime_config_file"])
    assert recovered is not None
    assert recovered["history_source_config_file"] == "configs/imported/522.toml"

    history_dir = tmp_path / "history"
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    svc = TrainingService(web.Application())
    svc._start_history_task(
        job="training",
        variant="522",
        preset="default",
        methods_subdir="imported",
        output_dir=recovered["output_dir"],
        sample_dir=recovered["sample_dir"],
        data_dirs=recovered["data_dirs"],
        sample_config={},
        command=["python", "train.py"],
        config_file=recovered["runtime_config_file"],
        runtime_info=recovered,
    )

    task = svc.list_history_tasks()[0]
    assert task["history_group_key"] == "source:configs/imported/522.toml"
    assert task["history_source_config_file"] == "configs/imported/522.toml"
    assert task["history_run_label"] == "522-20260523-114514"


def test_absolute_output_root_runtime_config_allowed_in_preflight(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    output_root = tmp_path.parent / "absolute-output-root"
    monkeypatch.setattr(settings_service, "resolve_output_root", lambda value=None: output_root.resolve())
    monkeypatch.setattr(training_service, "resolve_output_root", lambda: output_root.resolve())
    monkeypatch.setattr(config_service, "resolve_output_root", lambda: output_root.resolve())
    monkeypatch.setattr(
        training_service,
        "_display_settings_path",
        lambda path: str(Path(path).resolve()),
    )

    for rel in ("image_dataset/a", "image_dataset/b"):
        image_dir = tmp_path / rel
        image_dir.mkdir(parents=True)
        Image.new("RGB", (8, 8), color=(10, 20, 30)).save(image_dir / "sample.png")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "anima.safetensors").write_bytes(b"model")
    (tmp_path / "models" / "qwen.safetensors").write_bytes(b"qwen")
    (tmp_path / "models" / "vae.safetensors").write_bytes(b"vae")
    (tmp_path / "configs" / "imported" / "522.toml").write_text(
        "\n".join(
            [
                'pretrained_model_name_or_path = "models/anima.safetensors"',
                'qwen3 = "models/qwen.safetensors"',
                'vae = "models/vae.safetensors"',
                'dataset_config = "configs/datasets/522.toml"',
            ]
        ),
        encoding="utf-8",
    )

    runtime = training_service._prepare_web_runtime_config(
        "522",
        "default",
        "imported",
        source_config_file="configs/imported/522.toml",
    )

    runtime_config = runtime["runtime_config_file"]
    assert Path(runtime_config).is_absolute()
    assert config_service.is_web_runtime_config(runtime_config) is True
    result = config_service.preflight_training_config(
        "522",
        "default",
        "imported",
        config_file=runtime_config,
    )
    assert not any("项目目录内" in item["message"] for item in result["errors"])


def test_handle_start_converts_plain_config_to_preprocess_train_after(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    image_a = tmp_path / "image_dataset" / "a"
    image_b = tmp_path / "image_dataset" / "b"
    image_a.mkdir(parents=True)
    image_b.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(image_a / "a.png")
    Image.new("RGB", (8, 8), color=(30, 20, 10)).save(image_b / "b.png")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "anima.safetensors").write_bytes(b"model")
    (tmp_path / "models" / "qwen.safetensors").write_bytes(b"qwen")
    (tmp_path / "models" / "vae.safetensors").write_bytes(b"vae")
    (tmp_path / "configs" / "base.toml").write_text(
        "\n".join(
            [
                'pretrained_model_name_or_path = "models/anima.safetensors"',
                'qwen3 = "models/qwen.safetensors"',
                'vae = "models/vae.safetensors"',
                'dataset_config = "configs/datasets/522.toml"',
                'source_image_dir = "image_dataset/a"',
            ]
        ),
        encoding="utf-8",
    )

    class FakeService:
        def __init__(self):
            self.preprocess_calls = []
            self.start_calls = []

        async def start_preprocess(self, *args, **kwargs):
            self.preprocess_calls.append((args, kwargs))

        async def start(self, *args, **kwargs):
            self.start_calls.append((args, kwargs))

    svc = FakeService()
    req = _FakeJsonRequest(
        {
            "variant": "522",
            "preset": "default",
            "methods_subdir": "imported",
            "config_file": "configs/imported/522.toml",
            "extra_args": ["--foo"],
            "gpu_whitelist": [0],
        },
        {"training_service": svc},
    )

    response = asyncio.run(training_routes.handle_start(req))

    assert response.status == 200
    payload = json.loads(response.text)
    assert payload["job"] == "preprocess"
    assert payload["train_after"] is True
    assert "自动开始训练" in payload["message"]
    assert len(svc.preprocess_calls) == 1
    assert svc.start_calls == []
    args, kwargs = svc.preprocess_calls[0]
    assert args[:5] == ("522", "default", "imported", ["--foo"], True)
    assert kwargs["config_file"] == "configs/imported/522.toml"
    assert kwargs["gpu_whitelist"] == [0]


def test_start_preprocess_preserves_extra_args_for_pending_training(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        training_service,
        "_prepare_web_runtime_config",
        lambda *args, **kwargs: {
            "runtime_config_file": "output/runs/522-20260523-114514/config.runtime.toml",
            "output_dir": "output/runs/522-20260523-114514/training_output",
            "sample_dir": "output/runs/522-20260523-114514/training_output/sample",
            "sample_config": {},
            "data_dirs": {},
            "run_dir": "output/runs/522-20260523-114514",
        },
    )

    svc = TrainingService(web.Application())

    async def fake_launch(*args, **kwargs):
        return None

    svc._launch_job = fake_launch
    asyncio.run(
        svc.start_preprocess(
            "522",
            "default",
            "imported",
            ["--sample_every_n_steps", "5"],
            train_after=True,
        )
    )

    assert svc._pending_train_after_preprocess["extra_args"] == ["--sample_every_n_steps", "5"]


def test_handle_start_returns_400_for_missing_config_file(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)

    class FakeService:
        async def start_preprocess(self, *args, **kwargs):
            raise AssertionError("不应启动预处理")

        async def start(self, *args, **kwargs):
            raise AssertionError("不应启动训练")

    req = _FakeJsonRequest(
        {
            "variant": "522",
            "preset": "default",
            "methods_subdir": "imported",
            "config_file": "configs/imported/missing.toml",
        },
        {"training_service": FakeService()},
    )

    response = asyncio.run(training_routes.handle_start(req))

    assert response.status == 400


def test_handle_start_blocks_spd_cli_only_variant(monkeypatch):
    class FakeService:
        async def start_preprocess(self, *args, **kwargs):
            raise AssertionError("不应启动预处理")

        async def start(self, *args, **kwargs):
            raise AssertionError("不应启动训练")

    monkeypatch.setattr(
        training_routes,
        "preflight_training_config",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("不应预检测 SPD")),
    )
    req = _FakeJsonRequest(
        {
            "variant": "spd",
            "preset": "default",
            "methods_subdir": "methods",
            "config_file": "configs/methods/spd.toml",
        },
        {"training_service": FakeService()},
    )

    response = asyncio.run(training_routes.handle_start(req))

    assert response.status == 400
    payload = json.loads(response.text)
    assert payload["ok"] is False
    assert "CLI" in payload["error"]


def test_handle_start_uses_runtime_config_for_direct_training(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    run_dir = tmp_path / "output" / "runs" / "522-20260523-114514"
    run_dir.mkdir(parents=True)
    (run_dir / "model_cache").mkdir()
    (run_dir / "dataset_cache").mkdir()
    (run_dir / "training_output").mkdir()
    (run_dir / "config.runtime.toml").write_text(
        'output_dir = "output/runs/522-20260523-114514/training_output"\n',
        encoding="utf-8",
    )

    class FakeService:
        def __init__(self):
            self.preprocess_calls = []
            self.start_calls = []

        async def start_preprocess(self, *args, **kwargs):
            self.preprocess_calls.append((args, kwargs))

        async def start(self, *args, **kwargs):
            self.start_calls.append((args, kwargs))

    svc = FakeService()
    monkeypatch.setattr(training_routes, "preflight_training_config", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(training_routes, "is_web_runtime_config", lambda value: value.endswith("config.runtime.toml"))
    req = _FakeJsonRequest(
        {
            "variant": "522",
            "preset": "default",
            "methods_subdir": "imported",
            "config_file": "output/runs/522-20260523-114514/config.runtime.toml",
            "extra_args": ["--foo"],
        },
        {"training_service": svc},
    )

    response = asyncio.run(training_routes.handle_start(req))

    assert response.status == 200
    payload = json.loads(response.text)
    assert payload["job"] == "training"
    assert payload["train_after"] is False
    assert svc.preprocess_calls == []
    assert len(svc.start_calls) == 1
    args, kwargs = svc.start_calls[0]
    assert args[:4] == ("522", "default", ["--foo"], "imported")
    assert kwargs["config_file"] == "output/runs/522-20260523-114514/config.runtime.toml"
    assert kwargs["use_runtime_dir"] is False


def test_handle_start_uses_runtime_config_from_absolute_output_root(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    output_root = tmp_path / "external-runs"
    run_dir = output_root / "522-20260523-114514"
    model_cache = run_dir / "model_cache"
    dataset_cache = run_dir / "dataset_cache" / "dataset-01"
    training_output = run_dir / "training_output"
    source_dir = tmp_path / "image_dataset" / "a"
    for path in (model_cache, dataset_cache / "resized", dataset_cache / "lora", training_output, source_dir):
        path.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(dataset_cache / "resized" / "a.png")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "anima.safetensors").write_bytes(b"model")
    (tmp_path / "models" / "qwen.safetensors").write_bytes(b"qwen")
    (tmp_path / "models" / "vae.safetensors").write_bytes(b"vae")
    runtime_config = run_dir / "config.runtime.toml"
    dataset_config = run_dir / "dataset.runtime.toml"
    runtime_config.write_text(
        "\n".join(
            [
                f'dataset_config = "{dataset_config.as_posix()}"',
                f'output_dir = "{training_output.as_posix()}"',
                f'logging_dir = "{(model_cache / "logs").as_posix()}"',
                'source_image_dir = "image_dataset/a"',
                f'resized_image_dir = "{(dataset_cache / "resized").as_posix()}"',
                f'lora_cache_dir = "{(dataset_cache / "lora").as_posix()}"',
                'pretrained_model_name_or_path = "models/anima.safetensors"',
                'qwen3 = "models/qwen.safetensors"',
                'vae = "models/vae.safetensors"',
            ]
        ),
        encoding="utf-8",
    )
    dataset_config.write_text(
        "\n".join(
            [
                "[[datasets]]",
                "",
                "[[datasets.subsets]]",
                f'image_dir = "{(dataset_cache / "resized").as_posix()}"',
                f'cache_dir = "{(dataset_cache / "lora").as_posix()}"',
                'custom_attributes = {source_dir = "image_dataset/a"}',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config_service, "resolve_output_root", lambda: output_root.resolve())

    class FakeService:
        def __init__(self):
            self.preprocess_calls = []
            self.start_calls = []

        async def start_preprocess(self, *args, **kwargs):
            self.preprocess_calls.append((args, kwargs))

        async def start(self, *args, **kwargs):
            self.start_calls.append((args, kwargs))

    svc = FakeService()
    req = _FakeJsonRequest(
        {
            "variant": "522",
            "preset": "default",
            "methods_subdir": "imported",
            "config_file": str(runtime_config),
        },
        {"training_service": svc},
    )

    response = asyncio.run(training_routes.handle_start(req))

    assert response.status == 200
    payload = json.loads(response.text)
    assert payload["job"] == "training"
    assert svc.preprocess_calls == []
    assert len(svc.start_calls) == 1
    assert svc.start_calls[0][1]["config_file"] == str(runtime_config)
    assert svc.start_calls[0][1]["use_runtime_dir"] is False


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


def test_start_after_preprocess_uses_runtime_config_for_preflight(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    captured = {}

    def fake_preflight(*args, **kwargs):
        captured["preflight_args"] = args
        captured["preflight_kwargs"] = kwargs
        return {"ok": True}

    svc = TrainingService(web.Application())
    monkeypatch.setattr(training_service, "preflight_training_config", fake_preflight)

    async def fake_start(*args, **kwargs):
        captured["start_args"] = args
        captured["start_kwargs"] = kwargs

    svc.start = fake_start

    asyncio.run(
        svc._start_pending_training(
            {
                "variant": "522",
                "preset": "default",
                "methods_subdir": "imported",
                "extra_args": [],
                "config_file": "output/runs/522-20260523-114514/config.runtime.toml",
                "source_config_file": "configs/imported/522.toml",
                "gpu_whitelist": [0],
            }
        )
    )

    assert captured["preflight_kwargs"]["config_file"] == "output/runs/522-20260523-114514/config.runtime.toml"
    assert captured["start_kwargs"]["config_file"] == "output/runs/522-20260523-114514/config.runtime.toml"
    assert captured["start_kwargs"]["source_config_file"] == "configs/imported/522.toml"
    assert captured["start_kwargs"]["use_runtime_dir"] is False


def test_status_snapshot_includes_runtime_info():
    svc = TrainingService(web.Application())
    svc.current_runtime_info = {
        "run_dir": "output/runs/522-20260523-114514",
        "runtime_config_file": "output/runs/522-20260523-114514/config.runtime.toml",
        "original_config_file": "output/runs/522-20260523-114514/config.original.toml",
        "dataset_config_file": "output/runs/522-20260523-114514/dataset.runtime.toml",
        "model_cache_dir": "output/runs/522-20260523-114514/model_cache",
        "dataset_cache_dir": "output/runs/522-20260523-114514/dataset_cache",
        "training_output_dir": "output/runs/522-20260523-114514/training_output",
        "logs_dir": "output/runs/522-20260523-114514/model_cache/logs",
    }

    snapshot = svc.get_status_snapshot()

    assert snapshot["run_dir"] == "output/runs/522-20260523-114514"
    assert snapshot["runtime_config_file"].endswith("config.runtime.toml")
    assert snapshot["dataset_cache_dir"].endswith("dataset_cache")


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
    assert metrics[1]["kind"] == "val"
    assert metrics[1]["cmmd"] == 0.03
    assert any("结构化训练进度已开始" in item["line"] for item in logs)
    assert any("已保存检查点" in item["line"] for item in logs)
    assert any("结构化训练进度结束" in item["line"] for item in logs)


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

    task = svc.list_history_tasks()[0]

    assert task["run_dir"] == "output/runs/522-20260523-114514"
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

    def unwrap_model(self, model):
        return model


class _TinyResumeNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


def _resume_saver(args):
    return CheckpointSaver(
        args=args,
        accelerator=_FakeAccelerator(),
        save_dtype=None,
        metadata={},
        minimum_metadata={},
        get_sai_model_spec_fn=lambda _args: {},
        current_epoch=SimpleNamespace(value=0),
        current_step=SimpleNamespace(value=0),
    )


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


def test_auto_resume_skips_incompatible_network_state(tmp_path):
    state_dir = tmp_path / "demo-checkpoint-state"
    state_dir.mkdir()
    (state_dir / "train_state.json").write_text(
        json.dumps({"current_epoch": 1, "current_step": 3}),
        encoding="utf-8",
    )
    save_file({"lora_down.weight": torch.zeros(1)}, str(state_dir / "model.safetensors"))
    args = SimpleNamespace(
        output_dir=str(tmp_path),
        output_name="demo",
        checkpointing_epochs=1,
        resume=None,
        max_train_steps=10,
        skip_until_initial_step=False,
    )

    _resume_saver(args).auto_resume(_TinyResumeNetwork())

    assert args.resume is None
    assert args.skip_until_initial_step is False


def test_auto_resume_uses_compatible_network_state(tmp_path):
    state_dir = tmp_path / "demo-checkpoint-state"
    state_dir.mkdir()
    (state_dir / "train_state.json").write_text(
        json.dumps({"current_epoch": 1, "current_step": 3}),
        encoding="utf-8",
    )
    save_file({"weight": torch.zeros(1)}, str(state_dir / "model.safetensors"))
    args = SimpleNamespace(
        output_dir=str(tmp_path),
        output_name="demo",
        checkpointing_epochs=1,
        resume=None,
        max_train_steps=10,
        skip_until_initial_step=False,
    )

    _resume_saver(args).auto_resume(_TinyResumeNetwork())

    assert args.resume == str(state_dir)
    assert args.skip_until_initial_step is True


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
