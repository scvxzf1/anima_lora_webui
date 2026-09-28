"""CPU contracts for budgeted Qwen projection retention."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TransformerBlock
from library.models.qwen_image_2_1 import block_swap as swap_module
from library.models.qwen_image_2_1 import compile as compile_module
from library.models.qwen_image_2_1.training_blocks import install_training_block_projections
from networks.lora_modules.lora import LoRAModule


def _fixture(*, saved_blocks=0, budget_mib=0, block_count=2, swap=False):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = nn.ModuleList(
                QwenImage21TransformerBlock(8, 2, 4, mlp_ratio=2).to(torch.bfloat16)
                for _ in range(block_count)
            )
            self.gradient_checkpointing = True
            self._gradient_checkpointing_func = lambda fn, *args: checkpoint(
                fn, *args, use_reentrant=False,
            )

        def enable_gradient_checkpointing(self):
            self.gradient_checkpointing = True

    model = Model()
    if swap:
        swap_module.enable_qwen_image_2_1_block_swap(model, 1, torch.device("cpu"))
    loras = []
    for block in model.transformer_blocks:
        for name in ("to_q", "to_k", "to_v"):
            lora = LoRAModule(name, getattr(block.attn, name), lora_dim=2, alpha=2)
            lora.to(torch.bfloat16)
            lora.apply_to()
            with torch.no_grad():
                lora.lora_down.weight.normal_(std=0.05)
                lora.lora_up.weight.normal_(std=0.05)
            loras.append(lora)
    args = SimpleNamespace(
        model_family="qwen_image_2_1", qwen_fused_projections="all",
        qwen_saved_projection_blocks=saved_blocks,
        qwen_projection_budget_mib=budget_mib,
        mixed_precision="bf16", base_compute="bf16",
        block_swap_transfer_dtype="bf16", gradient_checkpointing=True,
        selective_checkpoint="off", network_module="networks.lora_anima",
    )
    install_training_block_projections(
        args, model, SimpleNamespace(unet_loras=loras, text_encoder_loras=[]),
    )
    return model, loras


def _run(model, loras, x, modulation):
    input_x = x.detach().clone().requires_grad_(True)
    x = input_x
    for block in model.transformer_blocks:
        x = model._gradient_checkpointing_func(block, x, modulation)
    result = x.float().square().mean()
    result.backward()
    grads = [
        (lora.lora_down.weight.grad.clone(), lora.lora_up.weight.grad.clone())
        for lora in loras
    ]
    return x.detach(), input_x.grad.clone(), grads


def test_segmented_retains_projection_results_and_matches_full_checkpoint():
    torch.manual_seed(17)
    full, full_loras = _fixture()
    torch.manual_seed(17)
    segmented, segmented_loras = _fixture(saved_blocks=1, budget_mib=1)
    x = torch.randn(1, 4, 8, dtype=torch.bfloat16)
    modulation = torch.randn(1, 32, dtype=torch.bfloat16)
    counts = {"full": [0, 0], "segmented": [0, 0]}
    handles = []
    for name, model in (("full", full), ("segmented", segmented)):
        tail = model.transformer_blocks[-1]
        handles.append(tail.attn.packed_qkv.register_forward_hook(
            lambda *_args, key=name: counts[key].__setitem__(0, counts[key][0] + 1)
        ))
        handles.append(tail.img_mlp.gate_up.register_forward_hook(
            lambda *_args, key=name: counts[key].__setitem__(1, counts[key][1] + 1)
        ))
    try:
        full_out, full_dx, full_grads = _run(full, full_loras, x, modulation)
        segmented_out, segmented_dx, segmented_grads = _run(segmented, segmented_loras, x, modulation)
    finally:
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(segmented_out, full_out, rtol=0, atol=0)
    torch.testing.assert_close(segmented_dx, full_dx, rtol=0, atol=0)
    for got_pair, expected_pair in zip(segmented_grads, full_grads):
        for got, expected in zip(got_pair, expected_pair):
            torch.testing.assert_close(got, expected, rtol=0, atol=0)
    assert counts["full"] == [2, 2]
    assert counts["segmented"] == [1, 1]


def test_budget_checked_before_projection_and_changes_with_shape():
    model, _ = _fixture(saved_blocks=2, budget_mib=1)
    block = model.transformer_blocks[-1]
    modulation = torch.zeros(1, 32, dtype=torch.bfloat16)
    with torch.no_grad():
        block(torch.zeros(1, 4, 8, dtype=torch.bfloat16), modulation)
    assert block(torch.zeros(1, 4, 8, dtype=torch.bfloat16), modulation).shape == (1, 4, 8)
    with pytest.raises(RuntimeError, match="payload"):
        block(torch.zeros(1, 10000, 8, dtype=torch.bfloat16), modulation)


def test_segmented_compile_targets_four_segments_not_whole_block(monkeypatch):
    model, _ = _fixture(saved_blocks=1, budget_mib=1)
    calls = []
    monkeypatch.setattr(
        compile_module.torch, "compile",
        lambda fn, **_kwargs: calls.append(fn) or fn,
    )
    assert compile_module.compile_qwen_image_2_1_blocks(model) == 2
    assert len(calls) == 5  # One ordinary full block, four selected segments.
    assert model.transformer_blocks[-1].forward not in calls


def test_selected_block_preserves_swap_wait_submit_and_backward_hook(monkeypatch):
    class RecordingOffloader:
        def __init__(self, blocks, _count, _device, **_kwargs):
            self.events = []
            self._cpu_weight_masters = None
            self._block_module_maps = [None] * len(blocks)
            for index, block in enumerate(blocks):
                block.register_full_backward_hook(
                    lambda _module, _grad_in, _grad_out, i=index:
                    self.events.append(("backward", i))
                )

        def wait_for_block(self, index):
            self.events.append(("wait", index))

        def submit_move_blocks(self, _blocks, index):
            self.events.append(("submit", index))

    monkeypatch.setattr(swap_module, "ModelOffloader", RecordingOffloader)
    model, loras = _fixture(saved_blocks=1, budget_mib=1, block_count=3, swap=True)
    _run(
        model, loras, torch.randn(1, 4, 8, dtype=torch.bfloat16),
        torch.randn(1, 32, dtype=torch.bfloat16),
    )
    events = model._qwen_image_2_1_block_swap_adapter.offloader.events
    assert events[:4] == [("wait", 0), ("submit", 0), ("wait", 1), ("submit", 1)]
    assert ("backward", 2) in events[4:]
    assert ("backward", 0) in events[4:]


@pytest.mark.parametrize("count,budget", [(1, 0), (0, 1), (3, 1)])
def test_invalid_selection_rejected(count, budget):
    with pytest.raises(ValueError):
        _fixture(saved_blocks=count, budget_mib=budget)
