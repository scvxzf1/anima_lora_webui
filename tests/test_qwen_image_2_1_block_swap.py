"""CPU contracts for Qwen Image 2.1 block swap and block compilation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from library.models.qwen_image_2_1 import block_swap as swap_module
from library.models.qwen_image_2_1 import compile as compile_module
from library.models.qwen_image_2_1.family import forward_for_loss
from library.training import model_loading


class RecordingOffloader:
    def __init__(self, blocks, blocks_to_swap, device, **_kwargs):
        self.blocks = blocks
        self.blocks_to_swap = blocks_to_swap
        self.device = torch.device(device)
        self.cuda_available = False
        self.futures = {}
        self.events = []

    def wait_for_block(self, index):
        self.events.append(("wait", index))

    def submit_move_blocks(self, _blocks, index):
        self.events.append(("submit", index))

    def prepare_block_devices_before_forward(self, _blocks, free_cache=True):
        self.events.append(("prepare", free_cache))

    def set_forward_only(self, value):
        self.events.append(("forward_only", value))

    def restore_blocks_to_device(self, _blocks, device):
        self.events.append(("restore", torch.device(device).type))

    def flush_profile_events(self, blocking=False):
        self.events.append(("flush", blocking))


def _tiny_model():
    from diffusers import QwenImage21Transformer2DModel

    return QwenImage21Transformer2DModel(
        num_layers=3,
        num_attention_heads=1,
        attention_head_dim=128,
        context_in_dim=4096,
        mlp_ratio=1,
    )


def _forward(model):
    return forward_for_loss(
        model,
        torch.randn(1, 64, 2, 2),
        torch.randn(1, 3, 4096),
        torch.tensor([[True, True, False]]),
        torch.tensor([0.5]),
    )


def test_checkpoint_recompute_does_not_resubmit_swap(monkeypatch):
    monkeypatch.setattr(swap_module, "ModelOffloader", RecordingOffloader)
    model = _tiny_model()
    model.enable_gradient_checkpointing()
    adapter = swap_module.enable_qwen_image_2_1_block_swap(model, 1, torch.device("cpu"))
    # The training bootstrap repeats this call after applying LoRA.
    model.enable_gradient_checkpointing()

    _forward(model).square().mean().backward()

    assert adapter.offloader.events == [
        event for index in range(3) for event in (("wait", index), ("submit", index))
    ]
    assert model.transformer_blocks[0].attn.to_q.weight.grad is not None


def test_external_adapter_keeps_its_gradient_with_swap(monkeypatch):
    monkeypatch.setattr(swap_module, "ModelOffloader", RecordingOffloader)
    model = _tiny_model().requires_grad_(False)
    projection = model.transformer_blocks[0].attn.to_q
    adapter = torch.nn.Linear(projection.in_features, projection.out_features, bias=False)
    original_forward = projection.forward
    projection.forward = lambda x: original_forward(x) + adapter(x)
    model.enable_gradient_checkpointing()
    swap_module.enable_qwen_image_2_1_block_swap(model, 1, torch.device("cpu"))

    _forward(model).square().mean().backward()

    assert adapter.weight.grad is not None
    assert adapter.weight.grad.abs().sum() > 0
    assert projection.weight.grad is None


def test_no_grad_forward_and_placement_protocol(monkeypatch):
    monkeypatch.setattr(swap_module, "ModelOffloader", RecordingOffloader)
    model = _tiny_model()
    keys = set(model.state_dict())
    adapter = swap_module.enable_qwen_image_2_1_block_swap(model, 1, torch.device("cpu"))
    model.move_to_device_except_swap_blocks(torch.device("cpu"))
    model.prepare_block_swap_before_forward(free_cache=False)
    adapter.offloader.events.clear()

    with torch.no_grad():
        _forward(model)

    assert set(model.state_dict()) == keys
    assert adapter.offloader.events == [
        event for index in range(3) for event in (("wait", index), ("submit", index))
    ]
    assert model.pause_block_swap() is True
    assert model.blocks_to_swap == 0
    assert model.resume_block_swap() is True
    assert model.blocks_to_swap == 1


@pytest.mark.parametrize("count", [0, 2])
def test_swap_rejects_invalid_count(monkeypatch, count):
    monkeypatch.setattr(swap_module, "ModelOffloader", RecordingOffloader)
    with pytest.raises(ValueError, match="between 1 and 1"):
        swap_module.enable_qwen_image_2_1_block_swap(
            _tiny_model(), count, torch.device("cpu")
        )


def test_compile_targets_inner_forward_after_swap(monkeypatch):
    monkeypatch.setattr(swap_module, "ModelOffloader", RecordingOffloader)
    model = _tiny_model()
    adapter = swap_module.enable_qwen_image_2_1_block_swap(model, 1, torch.device("cpu"))
    original = list(adapter.inner_forwards)
    wrappers = [block.forward for block in model.transformer_blocks]
    calls = []

    def fake_compile(forward, **kwargs):
        calls.append((forward, kwargs))
        return lambda *args, **named: forward(*args, **named)

    monkeypatch.setattr(compile_module.torch, "compile", fake_compile)
    assert compile_module.compile_qwen_image_2_1_blocks(
        model, backend="inductor", dynamic_seq=True, scope="all"
    ) == 3
    assert [forward for forward, _ in calls] == original
    assert all(kwargs["dynamic"] is True for _, kwargs in calls)
    assert [block.forward for block in model.transformer_blocks] == wrappers
    with torch.no_grad():
        _forward(model)
    assert ("wait", 0) in adapter.offloader.events
    with pytest.raises(RuntimeError, match="already compiled"):
        compile_module.compile_qwen_image_2_1_blocks(model)


def test_compile_resident_scope_and_cudagraph_rejection(monkeypatch):
    monkeypatch.setattr(swap_module, "ModelOffloader", RecordingOffloader)
    model = _tiny_model()
    swap_module.enable_qwen_image_2_1_block_swap(model, 1, torch.device("cpu"))
    calls = []
    monkeypatch.setattr(
        compile_module.torch,
        "compile",
        lambda forward, **kwargs: calls.append((forward, kwargs)) or forward,
    )
    assert compile_module.compile_qwen_image_2_1_blocks(model, scope="resident") == 2
    assert len(calls) == 2
    with pytest.raises(ValueError, match="non-CUDAGraph"):
        compile_module.compile_qwen_image_2_1_blocks(_tiny_model(), mode="reduce-overhead")


@pytest.mark.parametrize(("swapped", "scope", "compiled_count"), [
    (24, "all", 32),
    (24, "resident", 8),
    (30, "all", 32),
])
def test_full_block_count_swap_compile_boundary(monkeypatch, swapped, scope, compiled_count):
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList(
                torch.nn.Linear(1, 1) for _ in range(32)
            )

        def enable_gradient_checkpointing(self):
            self.gradient_checkpointing = True
            self._gradient_checkpointing_func = lambda function, *args: function(*args)

    monkeypatch.setattr(swap_module, "ModelOffloader", RecordingOffloader)
    monkeypatch.setattr(compile_module.torch, "compile", lambda forward, **_kw: forward)
    model = FakeModel()
    adapter = swap_module.enable_qwen_image_2_1_block_swap(
        model, swapped, torch.device("cpu")
    )

    assert len(adapter.inner_forwards) == 32
    assert compile_module.compile_qwen_image_2_1_blocks(model, scope=scope) == compiled_count
    assert model.blocks_to_swap == swapped


def test_loader_stages_swapped_transformer_on_cpu(monkeypatch):
    captured = {}

    class FakeModel(torch.nn.Module):
        def enable_gradient_checkpointing(self):
            captured["checkpointing"] = True

    def fake_load(path, *, dtype, device):
        captured["load"] = (path, dtype, torch.device(device))
        return FakeModel()

    def fake_enable(model, count, device, **kwargs):
        captured["swap"] = (model, count, torch.device(device), kwargs)

    monkeypatch.setattr(
        "library.models.qwen_image_2_1.weights.load_qwen_image_2_1_transformer", fake_load
    )
    monkeypatch.setattr(swap_module, "enable_qwen_image_2_1_block_swap", fake_enable)
    monkeypatch.setattr(model_loading, "resolve_block_swap_profile_jsonl", lambda _args: None)
    args = SimpleNamespace(
        pretrained_model_name_or_path="qwen.safetensors",
        gradient_checkpointing=True,
        blocks_to_swap=24,
        block_swap_transfer_dtype="bf16",
        block_swap_restore_mode="slab",
    )
    trainer = SimpleNamespace(is_swapping_blocks=False)
    accelerator = SimpleNamespace(device=torch.device("cpu"))

    model, text_encoders = model_loading._load_qwen_image_2_1_dit(
        trainer, args, torch.bfloat16, accelerator, [None]
    )

    assert trainer.is_swapping_blocks is True
    assert captured["load"] == ("qwen.safetensors", torch.bfloat16, torch.device("cpu"))
    assert captured["swap"] == (
        model,
        24,
        torch.device("cpu"),
        {"profile_jsonl": None, "transfer_dtype": "bf16", "restore_mode": "slab"},
    )
    assert text_encoders == [None]
    assert captured["checkpointing"] is True
