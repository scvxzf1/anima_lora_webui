from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import nn

from library.anima.models import Block
from library.runtime.offloading import ModelOffloader
from networks.attention_dispatch import AttentionParams
from networks.lora_modules.lora import LoRAModule


@dataclass(frozen=True)
class _HotScenario:
    name: str
    compile_blocks: bool = False
    full_checkpoint: bool = False
    block_swap: bool = False


@dataclass
class _RunResult:
    losses: list[torch.Tensor]
    trainable_params: dict[str, torch.Tensor]
    trainable_grads: dict[str, torch.Tensor]
    profile_events: list[dict]


_SCENARIOS = (
    _HotScenario("no_compile_no_checkpoint_no_swap"),
    _HotScenario("compile_only", compile_blocks=True),
    _HotScenario("full_checkpoint_only", full_checkpoint=True),
    _HotScenario("block_swap_only", block_swap=True),
    _HotScenario("compile_full_checkpoint", compile_blocks=True, full_checkpoint=True),
    _HotScenario("compile_block_swap", compile_blocks=True, block_swap=True),
    _HotScenario("full_checkpoint_block_swap", full_checkpoint=True, block_swap=True),
    _HotScenario(
        "compile_full_checkpoint_block_swap",
        compile_blocks=True,
        full_checkpoint=True,
        block_swap=True,
    ),
)


class _TinyBlockStack(nn.Module):
    def __init__(self, *, num_blocks: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Block(x_dim=8, context_dim=8, num_heads=2, mlp_ratio=2.0)
                for _ in range(num_blocks)
            ]
        )

    def enable_full_checkpointing(self) -> None:
        for block in self.blocks:
            block.enable_gradient_checkpointing()

    def compile_block_forwards(self, backend: str) -> None:
        for block in self.blocks:
            block._forward = torch.compile(
                block._forward,
                backend=backend,
                dynamic=False,
            )

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        crossattn_emb: torch.Tensor,
        offloader: ModelOffloader | None = None,
    ) -> torch.Tensor:
        attn_params = AttentionParams.create_attention_params("torch")
        if offloader is not None:
            offloader.prepare_block_devices_before_forward(
                self.blocks,
                free_cache=False,
            )
        for block_idx, block in enumerate(self.blocks):
            if offloader is not None:
                offloader.wait_for_block(block_idx)
            x = block(x, emb, crossattn_emb, attn_params)
            if offloader is not None:
                offloader.submit_move_blocks(self.blocks, block_idx)
        return x


def _attach_lora(block: Block, block_idx: int) -> None:
    for name, linear in (
        ("self_qkv", block.self_attn.qkv_proj),
        ("cross_q", block.cross_attn.q_proj),
        ("mlp1", block.mlp.layer1),
    ):
        module = LoRAModule(
            f"hot_b{block_idx}_{name}",
            linear,
            lora_dim=2,
            alpha=2,
        )
        module.apply_to()
        block.add_module(f"_hot_lora_{name}", module)


def _make_base_stack() -> _TinyBlockStack:
    torch.manual_seed(20260628)
    stack = _TinyBlockStack()
    for param in stack.parameters():
        param.requires_grad_(False)
    for block_idx, block in enumerate(stack.blocks):
        _attach_lora(block, block_idx)
    stack.train()
    return stack


def _make_step_inputs(step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(8800 + step)
    x = torch.randn(2, 1, 3, 1, 8, requires_grad=True)
    emb = torch.randn(2, 1, 8, requires_grad=True)
    crossattn_emb = torch.randn(2, 5, 8, requires_grad=True)
    return x, emb, crossattn_emb


def _trainable_snapshot(stack: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in stack.named_parameters()
        if param.requires_grad
    }


def _grad_snapshot(stack: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.grad.detach().cpu().clone()
        for name, param in stack.named_parameters()
        if param.requires_grad and param.grad is not None
    }


def _drain_offloader(offloader: ModelOffloader | None) -> None:
    if offloader is None:
        return
    for block_idx in list(offloader.futures.keys()):
        offloader._wait_blocks_move(block_idx, phase="drain")


def _read_profile(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _shutdown_offloader(offloader: ModelOffloader | None) -> None:
    if offloader is None:
        return
    _drain_offloader(offloader)
    for handle in getattr(offloader, "remove_handles", []):
        handle.remove()
    offloader.thread_pool.shutdown(wait=True)


def _run_training_steps(
    base_stack: _TinyBlockStack,
    scenario: _HotScenario,
    tmp_path: Path,
    *,
    compile_backend: str,
) -> _RunResult:
    stack = copy.deepcopy(base_stack)
    stack.train()
    if scenario.full_checkpoint:
        stack.enable_full_checkpointing()
    if scenario.compile_blocks:
        stack.compile_block_forwards(compile_backend)

    profile_path = tmp_path / f"{scenario.name}.block_swap.jsonl"
    offloader = (
        ModelOffloader(
            stack.blocks,
            blocks_to_swap=2,
            device=torch.device("cpu"),
            supports_backward=True,
            profile_jsonl=str(profile_path),
        )
        if scenario.block_swap
        else None
    )
    optimizer = torch.optim.SGD(
        [param for param in stack.parameters() if param.requires_grad],
        lr=1e-2,
    )
    losses: list[torch.Tensor] = []
    try:
        for step in range(2):
            x, emb, crossattn_emb = _make_step_inputs(step)
            out = stack(x, emb, crossattn_emb, offloader)
            loss = out.square().mean() + 0.0625 * out.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert x.grad is not None
            assert emb.grad is not None
            assert crossattn_emb.grad is not None
            assert torch.isfinite(x.grad).all()
            assert torch.isfinite(emb.grad).all()
            assert torch.isfinite(crossattn_emb.grad).all()

            optimizer.step()
            _drain_offloader(offloader)
            losses.append(loss.detach().cpu())
    finally:
        _shutdown_offloader(offloader)

    return _RunResult(
        losses=losses,
        trainable_params=_trainable_snapshot(stack),
        trainable_grads=_grad_snapshot(stack),
        profile_events=_read_profile(profile_path if scenario.block_swap else None),
    )


def _assert_block_swap_profile(events: list[dict]) -> None:
    assert any(event.get("ev") == "block_swap_config" for event in events)
    assert any(
        event.get("ev") == "block_swap"
        and event.get("phase") == "forward_wait"
        and event.get("submit_phase") == "forward_prefetch"
        for event in events
    )
    assert any(
        event.get("ev") == "block_swap"
        and event.get("phase") == "backward_wait"
        and event.get("submit_phase") == "backward_prefetch"
        for event in events
    )


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
@pytest.mark.parametrize("scenario", _SCENARIOS, ids=lambda scenario: scenario.name)
def test_compile_full_checkpoint_block_swap_hot_matrix(
    tmp_path: Path,
    scenario: _HotScenario,
) -> None:
    compile_backend = os.environ.get("ANIMA_HOT_COMPILE_BACKEND", "eager")
    base_stack = _make_base_stack()
    baseline = _run_training_steps(
        base_stack,
        _SCENARIOS[0],
        tmp_path,
        compile_backend=compile_backend,
    )
    actual = _run_training_steps(
        base_stack,
        scenario,
        tmp_path,
        compile_backend=compile_backend,
    )

    torch.testing.assert_close(torch.stack(actual.losses), torch.stack(baseline.losses))
    assert actual.trainable_params.keys() == baseline.trainable_params.keys()
    for name, actual_param in actual.trainable_params.items():
        torch.testing.assert_close(actual_param, baseline.trainable_params[name])
    assert actual.trainable_grads.keys() == baseline.trainable_grads.keys()
    for name, actual_grad in actual.trainable_grads.items():
        torch.testing.assert_close(actual_grad, baseline.trainable_grads[name])

    if scenario.block_swap:
        _assert_block_swap_profile(actual.profile_events)
    else:
        assert actual.profile_events == []
