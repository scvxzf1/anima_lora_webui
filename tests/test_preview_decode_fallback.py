"""CPU-only negative proofs for independent preview fallback and telemetry."""

from functools import partial
import json
from types import SimpleNamespace

import pytest
import torch

from library.training import preview_decode
from library.training.preview_decode_budget import (
    DecodeEstimate, PreviewDecodeBudgetError, cpu_decode_budget, estimate_cpu_decode,
)
from library.training.preview_decode_events import decode_event_sink
from library.training.preview_decode_policy import PreviewDecodeRestoreError, decode_with_policy


class VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.use_tiling = False
        self.moves = []

    def to(self, device=None, *, dtype=None):
        if dtype is not None:
            self.dtype = dtype
        else:
            self.device = torch.device(device)
            self.moves.append(str(device))
        return self


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)


@pytest.mark.parametrize("error", [
    torch.OutOfMemoryError("CPU out of memory"),
    torch.cuda.OutOfMemoryError("out of memory"),
    torch.cuda.OutOfMemoryError("CUDA host out of memory"),
    RuntimeError("CUDA illegal memory access"), RuntimeError("CUDA timeout"),
    ValueError("NaN in preview"),
])
def test_typed_cpu_ambiguous_and_non_oom_fail_closed(error):
    calls, events, vae = [], [], VAE()
    def decode(*args):
        calls.append(1)
        raise error
    with pytest.raises(type(error)):
        decode_with_policy(vae, torch.zeros(1), "cuda", "auto", decode, event_sink=events.append)
    assert calls == [1]
    expected = "oom" if isinstance(error, torch.OutOfMemoryError) else "reject"
    assert [event["event"] for event in events] == ["attempt", expected]
    assert not events[-1]["cuda_oom"]
    assert vae.device.type == "cpu" and not vae.use_tiling


def memory(available):
    return SimpleNamespace(available=available, reserve=20)


def budget(available):
    return partial(cpu_decode_budget, estimator=lambda *args: DecodeEstimate(10, 10, 10),
                   memory_reader=lambda: memory(available))


def test_two_gpu_ooms_then_cpu_success_and_structured_events(tmp_path):
    vae, calls = VAE(), []
    def decode(model, latent):
        calls.append((model.device.type, model.use_tiling))
        if model.device.type == "cuda":
            raise RuntimeError("CUDA out of memory")
        return "image"
    latent = torch.ones(1)
    assert decode_with_policy(vae, latent, "cuda", "auto", decode,
                              budget_service=budget(100),
                              event_sink=decode_event_sink(tmp_path, "kept.pt")) == "image"
    rows = [json.loads(row) for row in (tmp_path / "sample" / "vae-decode-events.jsonl").read_text().splitlines()]
    assert calls == [("cuda", False), ("cuda", True), ("cpu", True)]
    assert [row["event"] for row in rows] == ["attempt", "oom", "fallback", "attempt", "oom", "fallback",
                                             "attempt", "cpu_budget", "success"]
    assert rows[2]["next_stage"] == "tiled" and rows[5]["next_stage"] == "cpu"
    assert rows[7]["estimated_bytes"] == 30 and rows[7]["available_bytes"] == 100
    assert rows[7]["reserve_bytes"] == 20
    assert all(row["latent_filename"] == "kept.pt" for row in rows)
    assert torch.equal(latent, torch.ones(1))
    assert vae.device.type == "cpu" and not vae.use_tiling


def test_budget_rejects_before_cpu_move_or_decode():
    vae, events = VAE(), []
    with pytest.raises(PreviewDecodeBudgetError, match="host RAM"):
        decode_with_policy(vae, torch.zeros(1), "cuda", "cpu",
                           lambda *args: pytest.fail("must not decode"),
                           budget_service=budget(49), event_sink=events.append)
    assert vae.moves == ["cpu"]  # restoration only
    assert [event["event"] for event in events] == ["attempt", "cpu_budget", "reject"]
    assert not events[1]["accepted"]


@pytest.mark.parametrize("cpu_error", [PreviewDecodeBudgetError("RAM budget"), torch.OutOfMemoryError("CPU out of memory")])
def test_failed_cpu_decode_keeps_latent_and_training_args(tmp_path, monkeypatch, cpu_error):
    staged = tmp_path / "sample" / "latents"
    staged.mkdir(parents=True)
    path = staged / "pending.pt"
    torch.save({"latents": torch.ones(1)}, path)
    args = SimpleNamespace(output_dir=str(tmp_path), sample_vae_decode="cpu", mixed_precision="bf16",
                           train_batch_size=2, adaptive_oom_retry=True, optimizer_type="adamw")
    before = vars(args).copy()
    def fail(model, latent):
        if isinstance(cpu_error, PreviewDecodeBudgetError):
            pytest.fail("budget rejection must happen before decode")
        raise cpu_error
    monkeypatch.setattr(preview_decode, "clean_memory_on_device", lambda *args: None)
    available = 49 if isinstance(cpu_error, PreviewDecodeBudgetError) else 100
    monkeypatch.setattr(preview_decode, "decode_with_policy", partial(decode_with_policy, budget_service=budget(available)))
    preview_decode.decode_pending_samples(SimpleNamespace(device="cuda"), args, VAE(), decode_latent=fail)
    assert path.exists() and vars(args) == before
    rows = [json.loads(row) for row in (tmp_path / "sample" / "vae-decode-events.jsonl").read_text().splitlines()]
    expected = "oom" if isinstance(cpu_error, torch.OutOfMemoryError) else "reject"
    assert rows[-1]["event"] == expected and rows[-1]["stage"] == "cpu"
    assert not rows[-1]["cuda_oom"]


def test_budget_accounts_parameters_buffers_and_latent_without_clone():
    class TensorLike:
        device = SimpleNamespace(type="cuda")
        def numel(self):
            return 10
        def element_size(self):
            return 2
    parameter, buffer = TensorLike(), TensorLike()
    model = SimpleNamespace(parameters=lambda: iter([parameter, parameter]), buffers=lambda: iter([buffer]))
    estimate = estimate_cpu_decode(model, torch.ones(1, 16, 4, 4))
    assert estimate.model_copy_bytes == 40
    assert estimate.latent_bytes == 1 * 16 * 4 * 4 * 4 * 3
    assert estimate.tile_workspace_bytes > 0
    with pytest.raises(PreviewDecodeBudgetError, match="inspectable"):
        estimate_cpu_decode(object(), torch.zeros(1))


def test_event_io_failure_does_not_destroy_success(caplog):
    def broken(event):
        raise OSError("read-only event directory")
    assert decode_with_policy(VAE(), torch.zeros(1), "cpu", "cpu", lambda *args: "image",
                              budget_service=budget(100), event_sink=broken) == "image"
    assert "Unable to write" in caplog.text


def test_restoration_failure_emits_fatal():
    vae, events = VAE(), []
    original_to = vae.to
    def move(device=None, **kwargs):
        if str(device) == "cpu":
            raise RuntimeError("broken restore")
        return original_to(device, **kwargs)
    vae.to = move
    with pytest.raises(PreviewDecodeRestoreError):
        decode_with_policy(vae, torch.zeros(1), "cuda", "auto", lambda *args: "image", event_sink=events.append)
    assert events[-1]["event"] == "restoration_fatal"


def test_cpu_dtype_migration_preserves_channels_last_and_strides():
    tensor = torch.ones(1, 4, 8, 8).to(memory_format=torch.channels_last)
    original = tensor.stride()
    moved = tensor.to(device="cpu", dtype=torch.bfloat16)
    restored = moved.to(device="cpu", dtype=torch.float32)
    assert restored.stride() == original and restored.is_contiguous(memory_format=torch.channels_last)
    transposed = torch.ones(2, 3, 4).transpose(1, 2)
    assert transposed.to(dtype=torch.bfloat16).stride() == transposed.stride()
