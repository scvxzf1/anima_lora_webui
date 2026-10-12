"""CPU-only contracts for disposable Anima precision probes."""

from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.torch import save_file
import torch

from bench.adaptive_runtime.anima_inputs import load_anima_inputs, prepare_anima_inputs
from bench.adaptive_runtime import probe_anima_train as probe


def source_files(tmp_path, *, two_keys=False, metadata=None):
    latent = tmp_path / "latent.npz"
    arrays = {"latents_8x8": np.ones((16, 8, 8), dtype=np.float32)}
    if two_keys:
        arrays["latents_9x9"] = np.ones((16, 8, 8), dtype=np.float32)
    np.savez(latent, **arrays)
    text = tmp_path / "text.safetensors"
    save_file({"crossattn_emb": torch.ones(512, 1024)}, str(text),
              metadata=metadata or {"model_family": "anima", "cache_schema": "anima_te_v1"})
    return latent, text


def test_prepare_and_load_preserve_full_prompt_and_hash(tmp_path):
    latent, text = source_files(tmp_path)
    output = tmp_path / "inputs.safetensors"
    manifest = prepare_anima_inputs(latent, text, output)
    cached, digest = load_anima_inputs(output, device="cpu", dtype=torch.float32)
    assert manifest["inputs_sha256"] == digest
    assert cached["latents"].shape == (1, 16, 1, 8, 8)
    assert cached["prompt"].shape == (1, 512, 1024)
    assert cached["noise"].shape == cached["latents"].shape
    with pytest.raises(FileExistsError):
        prepare_anima_inputs(latent, text, output)
    output.with_suffix(".json").write_text("{}")
    with pytest.raises(ValueError, match="manifest/hash"):
        load_anima_inputs(output, device="cpu", dtype=torch.float32)


def test_prepare_refuses_ambiguity_shape_and_foreign_text(tmp_path):
    latent, text = source_files(tmp_path, two_keys=True)
    with pytest.raises(ValueError, match="exactly one"):
        prepare_anima_inputs(latent, text, tmp_path / "out.safetensors")
    with pytest.raises(ValueError, match="disagree"):
        prepare_anima_inputs(latent, text, tmp_path / "out.safetensors", latent_key="latents_9x9")
    bad = tmp_path / "foreign.safetensors"
    save_file({"crossattn_emb": torch.ones(512, 1024)}, str(bad),
              metadata={"model_family": "krea2_raw", "cache_schema": "krea2_te_v1"})
    with pytest.raises(ValueError, match="family/schema"):
        prepare_anima_inputs(latent, bad, tmp_path / "out.safetensors", latent_key="latents_8x8")


def test_reference_requires_bf16_source_without_silently_rounding_fp32(tmp_path):
    path = tmp_path / "base.safetensors"
    save_file({"weight": torch.ones(2, dtype=torch.bfloat16)}, str(path))
    assert probe.checkpoint_source_dtypes(path) == {"BF16": 1}
    save_file({"weight": torch.tensor([1 + 2**-20])}, str(path))
    with pytest.raises(ValueError, match="BF16 source"):
        probe.checkpoint_source_dtypes(path)


def args(tmp_path, **changes):
    weights = tmp_path / "weights.safetensors"
    weights.touch()
    inputs = tmp_path / "inputs.safetensors"
    inputs.touch()
    return SimpleNamespace(**{**dict(weights=weights, inputs=inputs,
                                    output=tmp_path / "worker" / "result.json",
                                    precision="fp16-islands", swap=2, steps=3,
                                    loss_scale=1024.0, fp32_pattern=["block"],
                                    record_replay=False, replay_reference=None), **changes})


def test_build_installs_precision_before_adapter_and_swap(monkeypatch, tmp_path):
    events = []

    class Model(torch.nn.Module):
        num_blocks = 6

        def __init__(self):
            super().__init__()
            self.block = torch.nn.Linear(2, 2)

        def enable_gradient_checkpointing(self):
            events.append("checkpoint")

        def enable_fp32_residual(self):
            events.append("residual")

        def enable_block_swap(self, *a, **kw):
            events.append("swap")
            assert kw["transfer_dtype"] is None

        def move_to_device_except_swap_blocks(self, _):
            events.append("move")

        def switch_block_swap_for_training(self):
            events.append("prepare")

    class Network(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            events.append("network")
            self.unet_loras = [1]
            self.adapter = torch.nn.Parameter(torch.ones(1))

        def apply_to(self, **kwargs):
            events.append("apply")

        def to(self, **kwargs):
            events.append("adapter_fp32")
            assert kwargs["dtype"] == torch.float32
            return self

    monkeypatch.setattr(probe, "load_anima_model", lambda **kw: Model())
    monkeypatch.setattr(probe, "install_training_precision",
                        lambda model, config, **kw: events.append("precision") or
                        {"assignments": {"block": "fp32"}})
    monkeypatch.setattr(probe.LoRANetworkCfg, "from_kwargs", lambda *a, **kw: object())
    monkeypatch.setattr(probe, "LoRANetwork", Network)
    report = {"stage": "model_load"}
    arg = args(tmp_path)
    arg.output.parent.mkdir()
    probe.build(arg, report)
    assert events == ["precision", "network", "apply", "adapter_fp32",
                      "checkpoint", "residual", "swap", "move", "prepare"]
    assert report["fp32_modules"] == ["block"]


def test_signature_and_rejections(tmp_path):
    arg = args(tmp_path)
    sig = probe.signature_for(arg, {"weights_sha256": "w", "inputs_sha256": "i",
                                    "fp32_modules": ["block"]})
    assert sig["family"] == "anima" and sig["sigmas"] == [0.2, 0.5, 0.8]
    assert sig["loss_scale"] == 1024.0 and sig["checkpoint"] == "full"
    probe.validate_args(arg)
    with pytest.raises(ValueError, match="steps"):
        probe.validate_args(args(tmp_path, steps=6))
    with pytest.raises(ValueError, match="FP32 patterns/scaling"):
        probe.validate_args(args(tmp_path, precision="bf16"))
    with pytest.raises(ValueError, match="recording"):
        probe.validate_args(args(tmp_path, precision="fp16-islands", record_replay=True))


def test_failure_writes_terminal_result_without_gpu(tmp_path):
    arg = args(tmp_path, steps=6)
    result = probe.run(arg)
    assert result["status"] == "error" and result["stage"] == "validate"
    assert "steps" in arg.output.read_text()
    with pytest.raises(FileExistsError):
        probe.run(arg)


def test_adamw_step_count_rejects_skip():
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([param])
    scaler = SimpleNamespace(get_scale=lambda: 1.0, step=lambda opt: None, update=lambda: None)
    with pytest.raises(probe.NonfiniteTrainingError, match="skipped"):
        probe._optimizer_step(optimizer, scaler, [param])
    param.grad = torch.tensor(1.0)
    scaler.step = lambda opt: opt.step()
    assert probe._optimizer_step(optimizer, scaler, [param]) == (1.0, 1.0)


@pytest.mark.parametrize("prefix", ["", "GPU-"])
def test_torch_uuid_matches_nvidia_smi_identity(prefix):
    raw = "806debd9-d254-da57-d515-ccae15b4ac6a"
    assert probe.device_uuid(prefix + raw) == "GPU-" + raw


def test_bf16_boundary_preserves_replay_leaf_and_casts_timestep():
    from library.anima.models import TimestepEmbedding

    timestep = TimestepEmbedding(8, 8, use_adaln_lora=True).bfloat16()
    modulation = torch.nn.Linear(8, 8).bfloat16()
    noisy = torch.ones(1, 16, 1, 8, 8, requires_grad=True)
    sigma = torch.tensor([0.2])

    def model(value, model_sigma, prompt, padding_mask):
        assert value is noisy and model_sigma.dtype == torch.bfloat16
        embedding, _ = timestep(model_sigma[:, None, None].expand(1, 1, 8))
        with torch.autocast("cpu", enabled=False):
            result = modulation(embedding)
        return value * result.float().mean()

    prediction = probe.forward_prediction(model, noisy, sigma, torch.ones(1), "bf16")
    prediction.sum().backward()
    assert sigma.dtype == torch.float32 and noisy.grad is not None


def test_cpu_three_step_capture_and_optimizer_updates(tmp_path):
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = torch.nn.Parameter(torch.tensor(0.5))

    network = Network()

    def model(noisy, sigma, prompt, padding_mask):
        assert noisy.ndim == 5 and prompt.shape == (1, 512, 1024)
        assert padding_mask.shape == (1, 1, 8, 8)
        assert padding_mask.dtype == noisy.dtype and torch.count_nonzero(padding_mask) == 0
        return noisy * network.adapter

    arg = args(tmp_path, precision="fp32-reference", swap=0,
               fp32_pattern=[], loss_scale=1.0)
    arg.output.parent.mkdir()
    inputs = {"latents": torch.ones(1, 16, 1, 8, 8),
              "noise": torch.zeros(1, 16, 1, 8, 8),
              "prompt": torch.ones(1, 512, 1024)}
    report = {"uuid": "cpu-fixture", "weights_sha256": "w", "inputs_sha256": "i",
              "fp32_modules": [], "updates": []}
    resources = []
    probe.train_steps(arg, model, network, inputs, report, resources)
    probe.close_resources(resources)
    assert [case["step"] for case in report["updates"]] == [1, 2, 3]
    assert report["adapter_max_delta"] > 0
    assert (arg.output.parent / "training-capture" / "manifest.json").is_file()
