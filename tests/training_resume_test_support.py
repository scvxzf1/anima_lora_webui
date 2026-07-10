"""Shared helpers for split training resume / history / runtime tests."""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
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
from library.runtime.launch import (
    ACCELERATE_LAUNCH_ENV,
    ACCELERATE_MIXED_PRECISION_ENV,
)
from web.routes import training as training_routes
from web.services import config_service, settings_service, training_service
from web.services.training import progress_parser
from web.services.training import runtime_config as training_runtime_config
from web.services.training_service import TrainingService


def _write_fake_accelerate_state_files(state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "model.safetensors").write_bytes(b"model")
    (state_dir / "optimizer.bin").write_bytes(b"optimizer")
    (state_dir / "scheduler.bin").write_bytes(b"scheduler")
    (state_dir / "random_states_0.pkl").write_bytes(b"rng")

def _write_resume_history(tmp_path):
    task_id = "20260517-000000-training-imported-demo"
    history_dir = tmp_path / "history"
    task_dir = history_dir / task_id
    output_dir = tmp_path / "output"
    state_dir = output_dir / "demo-checkpoint-state"
    source_dir = tmp_path / "image_dataset" / "demo"
    resized_dir = tmp_path / "post_image_dataset" / "resized"
    cache_dir = tmp_path / "post_image_dataset" / "lora"
    task_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)
    source_dir.mkdir(parents=True)
    resized_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)

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
        "\n".join(
            [
                f'output_dir = "{output_dir.as_posix()}"',
                'output_name = "demo"',
                f'source_image_dir = "{source_dir.as_posix()}"',
                f'resized_image_dir = "{resized_dir.as_posix()}"',
                f'lora_cache_dir = "{cache_dir.as_posix()}"',
                "train_batch_size = 1",
                "max_train_steps = 100",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (state_dir / "train_state.json").write_text(
        json.dumps({"current_epoch": 3, "current_step": 42}),
        encoding="utf-8",
    )
    _write_fake_accelerate_state_files(state_dir)
    (output_dir / "demo-checkpoint.safetensors").write_bytes(b"stub")
    os.utime(state_dir / "train_state.json", (1500.0, 1500.0))
    return history_dir, task_id, state_dir

def _patch_resume_runtime_output_root(monkeypatch, tmp_path):
    output_root = tmp_path / "runtime-runs"
    monkeypatch.setattr(training_service, "resolve_output_root", lambda: output_root)
    return output_root

def _patch_queue_storage(monkeypatch, tmp_path):
    queue_dir = tmp_path / "queue"
    monkeypatch.setattr(training_service, "QUEUE_DIR", queue_dir)
    monkeypatch.setattr(training_service, "QUEUE_FILE", queue_dir / "queue.json")
    return queue_dir

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
    (gui_methods / "dora.toml").write_text(
        "\n".join(
            [
                'network_module = "networks.lora_anima"',
                'dora_wd = true',
                'output_name = "dora-demo"',
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
    (gui_methods / "glora.toml").write_text(
        "\n".join(
            [
                'network_module = "networks.lora_anima"',
                'use_glora = true',
                'output_name = "glora-demo"',
            ]
        ),
        encoding="utf-8",
    )
    (gui_methods / "loha.toml").write_text(
        "\n".join(
            [
                'network_module = "networks.lora_anima"',
                'use_loha = true',
                'output_name = "loha-demo"',
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
        if kind == "LoHa":
            tensors = {
                "lora_unet_blocks_0_self_attn_q_proj.hada_w1_a": torch.randn(12, 4),
                "lora_unet_blocks_0_self_attn_q_proj.hada_w1_b": torch.randn(4, 8),
                "lora_unet_blocks_0_self_attn_q_proj.hada_w2_a": torch.randn(12, 4),
                "lora_unet_blocks_0_self_attn_q_proj.hada_w2_b": torch.randn(4, 8),
                "lora_unet_blocks_0_self_attn_q_proj.alpha": torch.tensor(4.0),
            }
            metadata = {"ss_network_spec": "loha", "ss_network_dim": "4"}
        elif kind == "LoKr":
            tensors = {
                "lora_unet_blocks_0_self_attn_q_proj.lokr_w1": torch.randn(2, 2),
                "lora_unet_blocks_0_self_attn_q_proj.lokr_w2": torch.randn(4, 4),
                "lora_unet_blocks_0_self_attn_q_proj.alpha": torch.tensor(32.0),
            }
            metadata = {"ss_network_spec": "lokr", "ss_network_dim": "32"}
        elif kind == "GLoRA":
            tensors = {
                "lora_unet_blocks_0_self_attn_q_proj.a1.weight": torch.randn(8, 4),
                "lora_unet_blocks_0_self_attn_q_proj.a2.weight": torch.randn(4, 8),
                "lora_unet_blocks_0_self_attn_q_proj.b1.weight": torch.randn(12, 4),
                "lora_unet_blocks_0_self_attn_q_proj.b2.weight": torch.randn(4, 8),
                "lora_unet_blocks_0_self_attn_q_proj.alpha": torch.tensor(4.0),
            }
            metadata = {"ss_network_spec": "glora", "ss_network_dim": "4"}
        elif kind == "DoRA":
            tensors = {
                "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight": (
                    torch.randn(4, 8)
                ),
                "lora_unet_blocks_0_self_attn_q_proj.lora_up.weight": (
                    torch.randn(12, 4)
                ),
                "lora_unet_blocks_0_self_attn_q_proj.alpha": torch.tensor(4.0),
                "lora_unet_blocks_0_self_attn_q_proj.dora_scale": (
                    torch.rand(12) + 0.5
                ),
            }
            metadata = {
                "ss_network_spec": "dora",
                "ss_adapter_variant": "dora",
                "ss_dora_compatible_export": "true",
            }
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
    monkeypatch.setattr(config_service, "DEFAULT_SAMPLE_PROMPTS_FILE", str(configs / "sample_prompts.txt"))
    monkeypatch.setattr(config_service, "DATASET_PRESETS_DIR", configs / "datasets")
    monkeypatch.setattr(training_service, "ROOT", root)
    monkeypatch.setattr(settings_service, "ROOT", root)
    monkeypatch.setattr(settings_service, "SETTINGS_FILE", configs / "web-ui-settings.toml")

class _FakeJsonRequest:
    def __init__(self, data, app=None, query=None, match_info=None):
        self._data = data
        self.app = app or {}
        self.query = query or {}
        self.match_info = match_info or {}

    async def json(self):
        return self._data

def _write_web_runtime_dir(output_root: Path, name: str) -> Path:
    run_dir = output_root / name
    (run_dir / "model_cache").mkdir(parents=True)
    (run_dir / "dataset_cache").mkdir(parents=True)
    (run_dir / "training_output" / "sample").mkdir(parents=True)
    (run_dir / "config.runtime.toml").write_text("output_name = 'demo'\n", encoding="utf-8")
    (run_dir / "training_output" / "demo.safetensors").write_bytes(b"weight")
    return run_dir

class _FakeAccelerator:
    def __init__(self, *, step=2, fail=False, is_main_process=True):
        self.step = step
        self.fail = fail
        self.is_main_process = is_main_process

    def print(self, *_args, **_kwargs):
        return None

    def wait_for_everyone(self):
        return None

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

class _TinySaveNetwork(_TinyResumeNetwork):
    def save_weights(self, path, _save_dtype, _metadata):
        Path(path).write_bytes(b"weights")

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

def _checkpointing_args(tmp_path, *, keep_last=2):
    return SimpleNamespace(
        output_dir=str(tmp_path),
        output_name="demo",
        save_model_as="safetensors",
        no_metadata=False,
        checkpointing_epochs=1,
        checkpointing_last_n_epochs=keep_last,
        resume=None,
        max_train_steps=100,
        skip_until_initial_step=False,
    )

def _weight_checkpoint_args(tmp_path, *, keep_last):
    return SimpleNamespace(
        output_dir=str(tmp_path),
        output_name="demo",
        save_model_as="safetensors",
        no_metadata=False,
        save_every_n_epochs=1,
        save_last_n_epochs=keep_last,
        save_state=False,
    )

def _weight_checkpoint_saver(args):
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

