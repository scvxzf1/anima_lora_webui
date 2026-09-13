from types import SimpleNamespace

import pytest
import torch

from library.training.anima_block_freeze import (
    apply_anima_block_freeze,
    parse_anima_freeze_blocks,
)
from networks.lora_modules.lora import LoRAModule


class _Accelerator:
    is_main_process = False

    @staticmethod
    def unwrap_model(model):
        return model


class _Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unet_loras = []

    def add_lora(self, block: int, linear: torch.nn.Linear) -> LoRAModule:
        lora = LoRAModule(f"lora_unet_blocks_{block}_proj", linear, lora_dim=2, alpha=2)
        lora.original_name = f"blocks.{block}.proj"
        lora.apply_to()
        self.add_module(lora.lora_name, lora)
        self.unet_loras.append(lora)
        with torch.no_grad():
            lora.lora_up.weight.fill_(0.125 * (block + 1))
        return lora

    @staticmethod
    def is_mergeable() -> bool:
        return True


def _args(**overrides):
    values = {
        "model_family": "anima",
        "network_module": "networks.lora_anima",
        "torch_compile": False,
        "blocks_to_swap": 0,
        "anima_freeze_blocks": "2",
        "anima_freeze_fuse": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_parse_anima_freeze_blocks_normalizes_and_rejects_invalid_values() -> None:
    assert parse_anima_freeze_blocks("17, 2  5,2") == (2, 5, 17)
    assert parse_anima_freeze_blocks("off") == ()
    with pytest.raises(ValueError, match="negative"):
        parse_anima_freeze_blocks("2,-1")


def test_freeze_and_fuse_preserves_forward_and_input_gradient() -> None:
    torch.manual_seed(7)
    network = _Network()
    target_linear = torch.nn.Linear(4, 4, bias=False)
    other_linear = torch.nn.Linear(4, 4, bias=False)
    target = network.add_lora(2, target_linear)
    other = network.add_lora(3, other_linear)
    x = torch.randn(3, 4)
    before = target_linear(x).detach()

    summary = apply_anima_block_freeze(
        _args(), _Accelerator(), network, resume_step=1000
    )

    assert summary is not None
    assert summary.blocks == (2,)
    assert summary.module_count == 1
    assert summary.fused is True
    assert target._fused is True
    assert all(not parameter.requires_grad for parameter in target.parameters())
    assert all(parameter.requires_grad for parameter in other.parameters())
    torch.testing.assert_close(target_linear(x), before, rtol=1e-5, atol=1e-6)

    x_with_grad = x.detach().requires_grad_(True)
    target_linear(x_with_grad).sum().backward()
    assert x_with_grad.grad is not None
    assert torch.count_nonzero(x_with_grad.grad) > 0
    assert all(parameter.grad is None for parameter in target.parameters())


def test_freeze_refuses_uncontrolled_or_unmatched_runs() -> None:
    network = _Network()
    network.add_lora(2, torch.nn.Linear(4, 4, bias=False))
    accelerator = _Accelerator()

    with pytest.raises(ValueError, match="resume-only"):
        apply_anima_block_freeze(_args(), accelerator, network, resume_step=0)
    with pytest.raises(ValueError, match="torch_compile=false"):
        apply_anima_block_freeze(
            _args(torch_compile=True), accelerator, network, resume_step=1000
        )
    with pytest.raises(ValueError, match="did not match"):
        apply_anima_block_freeze(
            _args(anima_freeze_blocks="17"), accelerator, network, resume_step=1000
        )
