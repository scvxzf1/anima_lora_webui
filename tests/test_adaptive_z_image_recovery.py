import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bench.adaptive_runtime import probe_z_image_train, recovery, z_image_checkpoint


def test_z_image_bf16_rejects_turing_before_loading(monkeypatch):
    monkeypatch.setattr(probe_z_image_train.torch.cuda, "get_device_capability", lambda: (7, 5))
    with pytest.raises(ValueError, match="native BF16"):
        probe_z_image_train.build(SimpleNamespace(precision="bf16"), "cuda")


def test_reference_promotes_base_to_fp32_before_attention(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(4, 2)).to(torch.bfloat16)
    monkeypatch.setattr(probe_z_image_train, "load_z_image_transformer", lambda *a, **k: model)
    monkeypatch.setattr(probe_z_image_train.torch.cuda, "get_device_capability", lambda: (7, 5))

    def inspect(model, mode, *, dtype):
        assert dtype == torch.float32
        assert all(p.dtype == torch.float32 for p in model.parameters())
        raise RuntimeError("reference dtype inspected")

    monkeypatch.setattr(probe_z_image_train, "prepare_z_image_attention", inspect)
    with pytest.raises(RuntimeError, match="reference dtype inspected"):
        probe_z_image_train.build(SimpleNamespace(precision="fp32-reference", weights="unused"), "cuda")


def test_weight_inventory_tracks_shards_and_not_other_components(tmp_path):
    root = tmp_path / "transformer"
    root.mkdir()
    (root / "config.json").write_text("{}")
    (root / "weights.safetensors").write_bytes(b"original")
    before = z_image_checkpoint.weight_inventory(tmp_path)
    assert len(before) == 2
    (tmp_path / "text_encoder").mkdir()
    (tmp_path / "text_encoder" / "other.bin").write_bytes(b"unrelated")
    assert z_image_checkpoint.weight_inventory(tmp_path) == before
    (root / "weights.safetensors").write_bytes(b"changed weight")
    assert z_image_checkpoint.weight_inventory(tmp_path) != before
    (root / "weights.safetensors").unlink()
    with pytest.raises(ValueError, match="requires local"):
        z_image_checkpoint.weight_inventory(tmp_path)


def test_signature_requires_immutable_inputs():
    with pytest.raises(ValueError, match="immutable"):
        z_image_checkpoint.signature_for(SimpleNamespace(inputs=None), {})


def test_z_image_resume_passes_scaler_to_restore(tmp_path, monkeypatch):
    sentinel = object()
    payload = {"step": 1}
    monkeypatch.setattr(z_image_checkpoint, "read_checkpoint", lambda *a, **k: payload)
    monkeypatch.setattr(z_image_checkpoint, "write_result", lambda *a, **k: None)

    def restore(value, **kwargs):
        assert value is payload
        assert kwargs["scaler"] is sentinel
        return ["initial"], [{"step": 1}]

    monkeypatch.setattr(z_image_checkpoint, "restore_checkpoint", restore)
    args = SimpleNamespace(resume=tmp_path / "step.pt", steps=3, output=tmp_path / "result.json")
    report = {}
    initial = z_image_checkpoint.resume_if_requested(args, report, {}, None, None, [], [], scaler=sentinel)
    assert initial == ["initial"]
    assert report["updates"] == [{"step": 1}]


@pytest.mark.parametrize("family,module", [("krea2", "probe_krea_train"),
                                           ("z_image", "probe_z_image_train")])
def test_recovery_command_preserves_plan_resume_and_precision(tmp_path, family, module):
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"plan": {"blocks_to_swap": 24}, "resume": "/saved/step-1.pt"}))
    args = SimpleNamespace(weights=Path("weights"), inputs=Path("inputs"),
                           memory_limit_gib=9.05, fp32_pattern=["*.attention.to_q"])
    command = recovery.worker_command(args, "fp16-islands", request, model_family=family)
    assert command[1:3] == ["-m", f"bench.adaptive_runtime.{module}"]
    for flag, value in {"--swap": "24", "--resume": "/saved/step-1.pt",
                        "--precision": "fp16-islands", "--memory-limit-gib": "9.05",
                        "--fp32-pattern": "*.attention.to_q"}.items():
        assert command[command.index(flag) + 1] == value
    assert "--checkpoint-every-step" in command


def test_unknown_family_is_rejected():
    with pytest.raises(KeyError):
        recovery.worker_command(None, "bf16", None, model_family="unknown")


@pytest.mark.parametrize("resume", [None, "/saved/scaled-step-1.pt"])
@pytest.mark.parametrize("family", ["krea2", "z_image"])
def test_scaled_recovery_command_keeps_scaler_contract(tmp_path, resume, family):
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"plan": {"blocks_to_swap": 26}, "resume": resume}))
    args = SimpleNamespace(weights="weights", inputs="inputs", memory_limit_gib=5.5,
                           fp32_pattern=["*.attention.*"], loss_scale=1024, scaled_checkpoint=True)
    argv = recovery.worker_command(args, "fp16-islands", request, model_family=family)
    assert "--scaled-checkpoint" in argv
    assert argv[argv.index("--loss-scale") + 1] == "1024"
    assert argv[argv.index("--swap") + 1] == "26"
    assert ("--resume" in argv) == (resume is not None)


@pytest.mark.parametrize("family,precision,scale,enabled", [
    ("unknown", "fp16-islands", 1024, True), ("z_image", "bf16", 1024, True),
    ("z_image", "fp16-islands", 1024, False), ("z_image", "fp16-islands", 1, True),
    ("z_image", "fp16-islands", float("nan"), True),
])
def test_scaled_recovery_rejects_unsupported_contract(family, precision, scale, enabled):
    with pytest.raises(ValueError):
        recovery.validate_recovery_scaling(SimpleNamespace(loss_scale=scale, scaled_checkpoint=enabled),
                                            precision, family)


def test_resolved_profile_is_recorded_before_gpu_placement(tmp_path, monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(4, 2))
    monkeypatch.setattr(probe_z_image_train, "load_z_image_transformer", lambda *a, **k: model)

    def fail_after_precision(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("placement failure")

    monkeypatch.setattr(probe_z_image_train, "prepare_z_image_attention", fail_after_precision)
    args = SimpleNamespace(precision="fp16-islands", weights="unused", fp32_module=[],
                           fp32_pattern=["0"], output=tmp_path / "result.json")
    report = {"precision_profile_resolved": False}
    with pytest.raises(torch.cuda.OutOfMemoryError):
        probe_z_image_train.build(args, "cuda", report)
    written = json.loads(args.output.read_text())
    assert written["precision_profile_resolved"] is True
    assert written["fp32_modules"] == ["0"]
    assert written["island_count"] == 1
