from argparse import Namespace
import json
from types import SimpleNamespace

import pytest
import torch

from library.training.auto_block_swap import capacity
from library.training.adaptive_runtime import supervisor
from library.training.adaptive_runtime.training_runtime import TrainingRuntime


@pytest.mark.parametrize("count,limit", [(28, 26), (40, 38)])
def test_anima_header_capacity(monkeypatch, count, limit):
    from library.anima import checkpoint

    monkeypatch.setattr(checkpoint, "inspect_anima_checkpoint",
                        lambda path: SimpleNamespace(num_blocks=count))
    result = capacity.preload_model_capacity(
        Namespace(model_family="anima", pretrained_model_name_or_path="local.safetensors")
    )
    assert result == {"model_family": "anima", "block_count": count, "model_limit": limit}


def test_krea_and_z_image_local_capacity(tmp_path):
    checkpoint = tmp_path / "krea.safetensors"
    checkpoint.touch()
    krea = capacity.preload_model_capacity(
        Namespace(model_family="krea2_raw", pretrained_model_name_or_path=str(checkpoint))
    )
    assert krea["block_count"] == 28 and krea["model_limit"] == 26
    transformer = tmp_path / "transformer"
    transformer.mkdir()
    (transformer / "config.json").write_text(json.dumps({"n_layers": 30}))
    z_image = capacity.preload_model_capacity(
        Namespace(model_family="z_image", pretrained_model_name_or_path=str(tmp_path))
    )
    assert z_image["block_count"] == 30 and z_image["model_limit"] == 28


@pytest.mark.parametrize("value", [None, True, 2, -1, "30"])
def test_invalid_z_image_structure_fails_closed(tmp_path, value):
    (tmp_path / "config.json").write_text(json.dumps({"n_layers": value}))
    with pytest.raises(ValueError):
        capacity.preload_model_capacity(
            Namespace(model_family="z_image", pretrained_model_name_or_path=str(tmp_path))
        )


def test_unknown_or_missing_checkpoint_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="does not support"):
        capacity.preload_model_capacity(Namespace(model_family="qwen_image_2_1"))
    with pytest.raises(ValueError, match="Local Krea-2"):
        capacity.preload_model_capacity(
            Namespace(model_family="krea2_raw", pretrained_model_name_or_path=str(tmp_path / "missing"))
        )


def test_z_image_single_file_uses_existing_header_validation(tmp_path, monkeypatch):
    from library.models.z_image import weights

    checkpoint = tmp_path / "z.safetensors"
    checkpoint.touch()
    calls = []
    monkeypatch.setattr(weights, "validate_z_image_single_file",
                        lambda path, component: calls.append((path, component)))
    assert capacity.preload_model_capacity(
        Namespace(model_family="z_image", pretrained_model_name_or_path=str(checkpoint))
    )["model_limit"] == 28
    assert calls == [(str(checkpoint), "transformer")]


def test_loaded_structure_and_true_dtype_host_budget(monkeypatch):
    from library.training.auto_block_swap.probe import model_block_bytes
    from library.training.auto_block_swap import resources

    model = SimpleNamespace(blocks=torch.nn.ModuleList([torch.nn.Linear(2, 2).float()
                                                         for _ in range(4)]))
    expected = {"model_family": "anima", "block_count": 4, "model_limit": 2,
                "requested_cap": 0, "effective_cap": 2}
    model.blocks[0].half()
    assert model_block_bytes(model, "anima") == [8, 16, 16, 16]
    monkeypatch.setattr(resources, "host_memory", lambda: SimpleNamespace(
        available=140, reserve=0, to_dict=lambda: {"available": 140}))
    measured = capacity.loaded_host_capacity(None, model, expected)
    assert measured["host_limit"] == 2
    assert measured["host_budget_reason"] == "measured_cpu_model"
    assert capacity.loaded_model_limit(model, "anima") == 2
    expected["block_count"] = 5
    with pytest.raises(ValueError, match="structure drifted"):
        capacity.loaded_host_capacity(None, model, expected)


def test_gpu_resident_model_has_unknown_host_budget(monkeypatch):
    model = SimpleNamespace(blocks=torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(4)]))
    monkeypatch.setattr(capacity, "loaded_blocks", lambda model, family: [
        SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device=SimpleNamespace(type="cuda"))]),
                        buffers=lambda: iter(()))
        for _ in range(4)
    ])
    expected = {"model_family": "anima", "block_count": 4, "model_limit": 2}
    result = capacity.loaded_host_capacity(None, model, expected)
    assert result == {"host_limit": None, "host_budget_reason": "model_not_cpu_resident"}


def test_worker_host_limit_is_not_cuda_oom(monkeypatch, tmp_path):
    runtime = TrainingRuntime(tmp_path, args=Namespace(
        blocks_to_swap=2, _adaptive_swap_capacity={"model_family": "anima",
        "block_count": 4, "model_limit": 2, "requested_cap": 0, "effective_cap": 2}))
    monkeypatch.setattr(capacity, "loaded_host_capacity", lambda *args: {
        "host_limit": 1, "host_budget_reason": "measured_cpu_model"})
    with pytest.raises(capacity.HostLimitReached):
        runtime.validate_swap_capacity(object(), "anima")
    assert runtime.result("host_limit")["host_limit"] == 1


def test_supervisor_rejects_user_cap_before_recovery_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "preload_model_capacity", lambda args: {
        "model_family": "anima", "block_count": 28, "model_limit": 26})
    with pytest.raises(ValueError, match="exceeds model limit"):
        supervisor.run_supervised(Namespace(
            output_dir=str(tmp_path), model_family="anima", adaptive_precision="fp16_fp32",
            adaptive_oom_retry=True, adaptive_oom_retry_max_swap=28,
            adaptive_oom_retry_max_attempts=4, adaptive_oom_retry_swap_increment=2,
            blocks_to_swap=0, seed=42))
    assert not (tmp_path / ".adaptive-recovery").exists()


def test_z_image_initial_28_is_legal_with_auto_cap(tmp_path, monkeypatch):
    from library.training.adaptive_runtime.training_config import require_training_contract

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"n_layers": 30}))
    args = Namespace(output_dir=str(tmp_path), pretrained_model_name_or_path=str(model),
                     model_family="z_image", adaptive_precision="fp16_fp32",
                     adaptive_oom_retry=True, adaptive_oom_retry_max_swap=0,
                     adaptive_oom_retry_max_attempts=1, adaptive_oom_retry_swap_increment=2,
                     adaptive_oom_retry_timeout=10, blocks_to_swap=28, seed=42, attn_mode="torch")
    require_training_contract(args)
    plans = []

    class Runner:
        def __init__(self, directory, command, **kwargs):
            pass

        def __call__(self, plan, **kwargs):
            plans.append(plan.blocks_to_swap)
            return {"status": "error"}

    monkeypatch.setattr(supervisor, "IsolatedRunner", Runner)
    report = supervisor.run_supervised(args)
    assert plans == [28]
    assert (report["model_limit"], report["requested_cap"], report["effective_cap"]) == (28, 0, 28)
