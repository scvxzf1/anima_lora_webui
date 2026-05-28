from __future__ import annotations

import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import toml
import torch
from aiohttp import web
from PIL import Image
from safetensors.torch import save_file

from library.training.checkpoints import (
    CheckpointSaver,
    plan_resume_start,
    save_checkpoint_state,
)
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
    gui_methods = configs / "gui-methods"
    datasets = configs / "datasets"
    imported.mkdir(parents=True)
    gui_methods.mkdir(parents=True)
    datasets.mkdir(parents=True)
    (root / "tasks.py").write_text("print('tasks')\n", encoding="utf-8")
    (root / "library" / "preprocess").mkdir(parents=True)
    (root / "library" / "__init__.py").write_text("", encoding="utf-8")
    (root / "library" / "preprocess" / "__init__.py").write_text("", encoding="utf-8")
    preprocess_dir = root / "scripts" / "preprocess"
    preprocess_dir.mkdir(parents=True)
    (root / "scripts" / "__init__.py").write_text("", encoding="utf-8")
    (root / "scripts" / "tasks").mkdir(parents=True)
    (root / "scripts" / "tasks" / "__init__.py").write_text("", encoding="utf-8")
    (root / "scripts" / "tasks" / "preprocess.py").write_text("", encoding="utf-8")
    for path in (
        preprocess_dir / "resize_images.py",
        preprocess_dir / "cache_latents.py",
        preprocess_dir / "cache_text_embeddings.py",
    ):
        path.write_text("from library.preprocess import resize_to_buckets\n", encoding="utf-8")
    (configs / "base.toml").write_text(
        "\n".join(
            [
                'source_image_dir = "image_dataset/fallback"',
                'resized_image_dir = "post_image_dataset/resized"',
                'lora_cache_dir = "post_image_dataset/lora"',
                'output_dir = "legacy/output"',
                'logging_dir = "legacy/logs"',
                'output_name = "demo"',
                "train_batch_size = 2",
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
                "train_batch_size = 2",
            ]
        ),
        encoding="utf-8",
    )
    (gui_methods / "lora.toml").write_text(
        "\n".join(
            [
                'network_module = "networks.lora_anima"',
                'output_name = "lora-demo"',
            ]
        ),
        encoding="utf-8",
    )
    (gui_methods / "lokr.toml").write_text(
        "\n".join(
            [
                'network_module = "networks.lora_anima"',
                'use_lokr = true',
                'output_name = "lokr-demo"',
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


def _write_continue_lora_weight(
    path: Path,
    *,
    kind: str = "LoRA",
    tensors=None,
    metadata=None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tensors is None:
        if kind == "LoKr":
            tensors = {
                "lora_unet_blocks_0_self_attn_q_proj.lokr_w1": torch.randn(2, 2),
                "lora_unet_blocks_0_self_attn_q_proj.lokr_w2": torch.randn(4, 4),
                "lora_unet_blocks_0_self_attn_q_proj.alpha": torch.tensor(32.0),
            }
            metadata = {"ss_network_spec": "lokr", "ss_network_dim": "32"}
        else:
            tensors = {
                "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight": torch.randn(4, 8),
                "lora_unet_blocks_0_self_attn_q_proj.lora_up.weight": torch.randn(12, 4),
                "lora_unet_blocks_0_self_attn_q_proj.alpha": torch.tensor(4.0),
            }
            metadata = {"ss_network_spec": "lora"}
    save_file(tensors, str(path), metadata=metadata)
    return path


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
    assert dataset_cfg["datasets"][0]["batch_size"] == 2
    assert dataset_cfg["datasets"][1]["batch_size"] == 2
    assert "resolution" not in dataset_cfg["datasets"][0]
    assert "bucket_reso_steps" not in dataset_cfg["datasets"][0]
    first_subset = dataset_cfg["datasets"][0]["subsets"][0]
    second_subset = dataset_cfg["datasets"][1]["subsets"][0]
    assert first_subset["custom_attributes"]["source_dir"] == "image_dataset/a"
    assert first_subset["custom_attributes"]["preprocess"]["resolution"] == 768
    assert first_subset["image_dir"].endswith("dataset-01/resized")
    assert first_subset["cache_dir"].endswith("dataset-01/lora")
    assert second_subset["custom_attributes"]["source_dir"] == "image_dataset/b"
    assert second_subset["custom_attributes"]["preprocess"]["resolution"] == 1024
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

    task = svc.list_history_tasks(include_archived=True)[0]
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
            "confirmed": True,
            "confirm_preprocess": True,
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


def test_handle_start_requires_explicit_confirmation_before_preprocess_train_after(tmp_path, monkeypatch):
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
        },
        {"training_service": svc},
    )

    response = asyncio.run(training_routes.handle_start(req))

    assert response.status == 409
    payload = json.loads(response.text)
    assert payload["requires_confirmation"] is True
    assert payload["requires_preprocess_confirmation"] is True
    assert payload["preflight"]["ok"] is True
    assert svc.preprocess_calls == []
    assert svc.start_calls == []


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


def test_inspect_continue_lora_weight_detects_lora_and_lokr(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    lora_path = _write_continue_lora_weight(tmp_path / "weights" / "demo.safetensors", kind="LoRA")
    lokr_path = _write_continue_lora_weight(tmp_path / "weights" / "demo_lokr.safetensors", kind="LoKr")

    lora_payload = training_service.inspect_continue_lora_weight(
        str(lora_path),
        variant="lora",
        preset="default",
        methods_subdir="gui-methods",
    )
    lokr_payload = training_service.inspect_continue_lora_weight(
        str(lokr_path),
        variant="lokr",
        preset="default",
        methods_subdir="gui-methods",
    )
    lokr_blocked = training_service.inspect_continue_lora_weight(
        str(lokr_path),
        variant="lora",
        preset="default",
        methods_subdir="gui-methods",
    )

    assert lora_payload["kind"] == "LoRA"
    assert lora_payload["compatible"] is True
    assert lokr_payload["kind"] == "LoKr"
    assert lokr_payload["compatible"] is True
    assert lokr_blocked["compatible"] is False
    assert "lokr" in lokr_blocked["message"].lower()


def test_inspect_continue_lora_weight_rejects_complex_lora_like_weights(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    plain_lora_tensors = {
        "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight": torch.randn(4, 8),
        "lora_unet_blocks_0_self_attn_q_proj.lora_up.weight": torch.randn(12, 4),
        "lora_unet_blocks_0_self_attn_q_proj.alpha": torch.tensor(4.0),
    }
    cases = [
        (
            "hydra_keys",
            {
                "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight": torch.randn(4, 8),
                "lora_unet_blocks_0_self_attn_q_proj.lora_ups.0.weight": torch.randn(12, 4),
                "lora_unet_blocks_0_self_attn_q_proj.router.weight": torch.randn(2, 4),
            },
            None,
        ),
        (
            "stacked_keys",
            {
                "lora_unet_blocks_0_self_attn_q_proj.lora_down_weight": torch.randn(2, 4, 8),
                "lora_unet_blocks_0_self_attn_q_proj.lora_up_weight": torch.randn(2, 12, 4),
            },
            None,
        ),
        ("hydra_spec", plain_lora_tensors, {"ss_network_spec": "hydra"}),
        ("stacked_spec", plain_lora_tensors, {"ss_network_spec": "stacked_experts_global_fei"}),
        ("chimera_spec", plain_lora_tensors, {"ss_network_spec": "chimera_hydra"}),
        (
            "reft_key",
            {"reft_unet_blocks_0.rotate_layer.weight": torch.randn(4, 4)},
            {"ss_network_spec": "reft"},
        ),
    ]

    for name, tensors, metadata in cases:
        path = _write_continue_lora_weight(
            tmp_path / "weights" / f"{name}.safetensors",
            tensors=tensors,
            metadata=metadata,
        )
        with pytest.raises(ValueError, match="未识别为 LoRA 或 LoKr"):
            training_service.inspect_continue_lora_weight(
                str(path),
                variant="lora",
                preset="default",
                methods_subdir="gui-methods",
            )


def test_inspect_continue_lora_weight_reports_path_errors(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)

    with pytest.raises(FileNotFoundError, match="权重文件不存在"):
        training_service.inspect_continue_lora_weight(
            str(tmp_path / "weights" / "missing.safetensors"),
            variant="lora",
            preset="default",
            methods_subdir="gui-methods",
        )

    txt_path = tmp_path / "weights" / "demo.txt"
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("not a safetensors file", encoding="utf-8")
    with pytest.raises(ValueError, match="只支持 .safetensors"):
        training_service.inspect_continue_lora_weight(
            str(txt_path),
            variant="lora",
            preset="default",
            methods_subdir="gui-methods",
        )

    directory_path = tmp_path / "weights" / "directory.safetensors"
    directory_path.mkdir()
    with pytest.raises(ValueError, match="权重路径不是文件"):
        training_service.inspect_continue_lora_weight(
            str(directory_path),
            variant="lora",
            preset="default",
            methods_subdir="gui-methods",
        )

    unreadable_path = _write_continue_lora_weight(tmp_path / "weights" / "unreadable.safetensors")
    real_access = os.access

    def fake_access(path, mode):
        if Path(path) == unreadable_path and mode == os.R_OK:
            return False
        return real_access(path, mode)

    monkeypatch.setattr(training_service.os, "access", fake_access)
    with pytest.raises(ValueError, match="权重文件不可读取"):
        training_service.inspect_continue_lora_weight(
            str(unreadable_path),
            variant="lora",
            preset="default",
            methods_subdir="gui-methods",
        )


def test_start_training_appends_network_weights_and_history_meta(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(training_service, "HISTORY_DIR", tmp_path / "history")
    weight = _write_continue_lora_weight(tmp_path / "weights" / "demo.safetensors", kind="LoRA")

    captured = {}
    svc = TrainingService(web.Application())

    async def fake_launch(cmd, env, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        svc.current_task_dir = tmp_path / "history" / "fake-task"
        svc.current_task_dir.mkdir(parents=True)
        svc.current_task_id = "fake-task"
        history_kwargs = {
            key: kwargs[key]
            for key in (
                "job",
                "variant",
                "preset",
                "methods_subdir",
                "output_dir",
                "sample_dir",
                "data_dirs",
                "sample_config",
                "config_file",
                "resume_info",
                "continue_info",
                "gpu_whitelist",
                "runtime_info",
            )
            if key in kwargs
        }
        svc._start_history_task(command=cmd, **history_kwargs)

    svc._launch_job = fake_launch
    asyncio.run(
        svc.start(
            "lora",
            "default",
            [],
            "gui-methods",
            config_file="configs/gui-methods/lora.toml",
            use_runtime_dir=False,
            continue_info={"continue_from_weight_abs_path": str(weight)},
        )
    )

    meta = json.loads((tmp_path / "history" / "fake-task" / "meta.json").read_text(encoding="utf-8"))
    assert captured["cmd"][1] == str(tmp_path / "train.py")
    assert "accelerate.commands.accelerate_cli" not in captured["cmd"]
    assert "--network_weights" in captured["cmd"]
    assert str(weight.resolve()) in captured["cmd"]
    assert "--dim_from_weights" in captured["cmd"]
    assert meta["training_mode"] == "continue_lora"
    assert meta["continue_from_weight_abs_path"] == str(weight.resolve())
    assert meta["continue_from_weight_name"] == "demo.safetensors"
    assert meta["continue_from_weight_kind"] == "LoRA"
    snapshot = (tmp_path / "history" / "fake-task" / "config.snapshot.toml").read_text(encoding="utf-8")
    assert '# training_mode = "continue_lora"' in snapshot
    assert str(weight.resolve()) in snapshot


def test_start_preprocess_keeps_continue_info_for_pending_training(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    runtime_config = tmp_path / "output" / "runs" / "522-20260523-114514" / "config.runtime.toml"
    runtime_config.parent.mkdir(parents=True)
    runtime_config.write_text('network_module = "networks.lora_anima"\n', encoding="utf-8")
    weight = _write_continue_lora_weight(tmp_path / "weights" / "demo.safetensors", kind="LoRA")
    monkeypatch.setattr(
        training_service,
        "_prepare_web_runtime_config",
        lambda *args, **kwargs: {
            "runtime_config_file": str(runtime_config),
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
            "lora",
            "default",
            "gui-methods",
            train_after=True,
            continue_info={"continue_from_weight_abs_path": str(weight)},
        )
    )

    assert svc._pending_train_after_preprocess["continue_info"]["continue_from_weight_abs_path"] == str(weight.resolve())


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


def test_handle_start_blocks_preprocess_environment_error(monkeypatch):
    class FakeService:
        def __init__(self):
            self.preprocess_calls = []
            self.start_calls = []

        async def start_preprocess(self, *args, **kwargs):
            self.preprocess_calls.append((args, kwargs))

        async def start(self, *args, **kwargs):
            self.start_calls.append((args, kwargs))

    failure = {
        "ok": False,
        "summary": {"errors": 1, "warnings": 0, "checks": 1},
        "checks": [{
            "level": "error",
            "key": "preprocess_environment",
            "message": "预处理启动环境异常: ModuleNotFoundError",
        }],
        "errors": [{
            "level": "error",
            "key": "preprocess_environment",
            "message": "预处理启动环境异常: ModuleNotFoundError",
        }],
        "warnings": [],
    }
    monkeypatch.setattr(training_routes, "preflight_training_config", lambda *args, **kwargs: failure)
    svc = FakeService()
    req = _FakeJsonRequest(
        {
            "variant": "522",
            "preset": "default",
            "methods_subdir": "imported",
            "config_file": "configs/imported/522.toml",
        },
        {"training_service": svc},
    )

    response = asyncio.run(training_routes.handle_start(req))

    assert response.status == 400
    payload = json.loads(response.text)
    assert payload["preflight"]["errors"][0]["key"] == "preprocess_environment"
    assert svc.preprocess_calls == []
    assert svc.start_calls == []


def test_handle_preprocess_blocks_preprocess_environment_error(monkeypatch):
    class FakeService:
        def __init__(self):
            self.preprocess_calls = []

        async def start_preprocess(self, *args, **kwargs):
            self.preprocess_calls.append((args, kwargs))

    failure = {
        "ok": False,
        "summary": {"errors": 1, "warnings": 0, "checks": 2},
        "checks": [
            {"level": "ok", "key": "source_image_dir", "message": "源图像目录 存在"},
            {
                "level": "error",
                "key": "preprocess_environment",
                "message": "预处理启动环境异常: ModuleNotFoundError",
            },
        ],
        "errors": [{
            "level": "error",
            "key": "preprocess_environment",
            "message": "预处理启动环境异常: ModuleNotFoundError",
        }],
        "warnings": [],
    }
    monkeypatch.setattr(training_routes, "preflight_training_config", lambda *args, **kwargs: failure)
    svc = FakeService()
    req = _FakeJsonRequest(
        {
            "variant": "522",
            "preset": "default",
            "methods_subdir": "imported",
            "config_file": "configs/imported/522.toml",
        },
        {"training_service": svc},
    )

    response = asyncio.run(training_routes.handle_preprocess(req))

    assert response.status == 400
    payload = json.loads(response.text)
    assert payload["preflight"]["errors"][0]["key"] == "preprocess_environment"
    assert svc.preprocess_calls == []


def test_handle_preprocess_requires_confirmation_before_train_after(monkeypatch):
    class FakeService:
        def __init__(self):
            self.preprocess_calls = []

        async def start_preprocess(self, *args, **kwargs):
            self.preprocess_calls.append((args, kwargs))

    preflight = {
        "ok": True,
        "summary": {"errors": 0, "warnings": 0, "checks": 1},
        "checks": [{"level": "ok", "key": "source_image_dir", "message": "源图像目录存在"}],
        "errors": [],
        "warnings": [],
    }
    monkeypatch.setattr(training_routes, "preflight_training_config", lambda *args, **kwargs: preflight)
    svc = FakeService()
    req = _FakeJsonRequest(
        {
            "variant": "522",
            "preset": "default",
            "methods_subdir": "imported",
            "config_file": "configs/imported/522.toml",
            "train_after": True,
        },
        {"training_service": svc},
    )

    response = asyncio.run(training_routes.handle_preprocess(req))

    assert response.status == 409
    payload = json.loads(response.text)
    assert payload["requires_confirmation"] is True
    assert payload["requires_train_after_confirmation"] is True
    assert svc.preprocess_calls == []


def test_handle_preprocess_allows_confirmed_train_after(monkeypatch):
    class FakeService:
        def __init__(self):
            self.preprocess_calls = []

        async def start_preprocess(self, *args, **kwargs):
            self.preprocess_calls.append((args, kwargs))

    preflight = {
        "ok": True,
        "summary": {"errors": 0, "warnings": 0, "checks": 1},
        "checks": [{"level": "ok", "key": "source_image_dir", "message": "源图像目录存在"}],
        "errors": [],
        "warnings": [],
    }
    monkeypatch.setattr(training_routes, "preflight_training_config", lambda *args, **kwargs: preflight)
    svc = FakeService()
    req = _FakeJsonRequest(
        {
            "variant": "522",
            "preset": "default",
            "methods_subdir": "imported",
            "config_file": "configs/imported/522.toml",
            "extra_args": ["--foo"],
            "gpu_whitelist": [0],
            "train_after": True,
            "confirmed": True,
            "confirm_train_after": True,
        },
        {"training_service": svc},
    )

    response = asyncio.run(training_routes.handle_preprocess(req))

    assert response.status == 200
    payload = json.loads(response.text)
    assert "自动开始训练" in payload["message"]
    assert len(svc.preprocess_calls) == 1
    args, kwargs = svc.preprocess_calls[0]
    assert args[:5] == ("522", "default", "imported", ["--foo"], True)
    assert kwargs["config_file"] == "configs/imported/522.toml"
    assert kwargs["gpu_whitelist"] == [0]


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
            "confirmed": True,
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


def test_handle_start_requires_explicit_confirmation_for_runtime_config(monkeypatch):
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
        },
        {"training_service": svc},
    )

    response = asyncio.run(training_routes.handle_start(req))

    assert response.status == 409
    payload = json.loads(response.text)
    assert payload["requires_confirmation"] is True
    assert payload["requires_preprocess_confirmation"] is False
    assert svc.preprocess_calls == []
    assert svc.start_calls == []


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
            "confirmed": True,
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

    assert svc.list_history_tasks() == []
    task = svc.list_history_tasks(include_archived=True)[0]

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
    assert task["archived"] is True
    assert task["name"] == "522-20260523-114514"


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


def test_plan_resume_start_uses_steps_from_state():
    args = SimpleNamespace(
        initial_epoch=None,
        initial_step=None,
        gradient_accumulation_steps=2,
        max_train_steps=100,
        skip_until_initial_step=True,
        resume="state-dir",
    )

    plan = plan_resume_start(
        args,
        steps_from_state=8,
        batches_per_epoch=10,
        num_processes=1,
    )

    assert plan.initial_step == 16
    assert plan.epoch_to_start == 3
    assert plan.steps_from_state is None


def test_plan_resume_start_initial_step_overrides_state():
    args = SimpleNamespace(
        initial_epoch=None,
        initial_step=6,
        gradient_accumulation_steps=2,
        max_train_steps=100,
        skip_until_initial_step=False,
        resume="state-dir",
    )

    plan = plan_resume_start(
        args,
        steps_from_state=42,
        batches_per_epoch=10,
        num_processes=1,
    )

    assert plan.initial_step == 0
    assert plan.epoch_to_start == 1
    assert plan.steps_from_state == 42


def test_plan_resume_start_skip_until_initial_step_scales_by_grad_accum():
    args = SimpleNamespace(
        initial_epoch=3,
        initial_step=None,
        gradient_accumulation_steps=3,
        max_train_steps=100,
        skip_until_initial_step=True,
        resume=None,
    )

    plan = plan_resume_start(
        args,
        steps_from_state=None,
        batches_per_epoch=12,
        num_processes=2,
    )

    assert plan.initial_step == 12
    assert plan.epoch_to_start == 3


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


def test_training_error_classifier_detects_cuda_oom():
    text = (
        "torch.OutOfMemoryError: CUDA out of memory. "
        "Tried to allocate 64.00 MiB."
    )

    assert training_service.classify_training_error(text) == "大概率爆显存"


def test_training_error_hint_is_added_once():
    assert (
        training_service._message_with_error_hint("训练异常退出 (code=1)", "大概率爆显存")
        == "训练异常退出 (code=1)：大概率爆显存"
    )
    assert (
        training_service._message_with_error_hint(
            "训练异常退出 (code=1)：大概率爆显存",
            "大概率爆显存",
        )
        == "训练异常退出 (code=1)：大概率爆显存"
    )


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
