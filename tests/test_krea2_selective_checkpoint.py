from __future__ import annotations

import pytest
import torch

from library.models.krea2_raw.dit import SingleStreamDiT


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False


def _tiny_dit(n_blocks: int = 6) -> SingleStreamDiT:
    model = SingleStreamDiT.__new__(SingleStreamDiT)
    torch.nn.Module.__init__(model)
    model.blocks = torch.nn.ModuleList([_Block() for _ in range(n_blocks)])
    return model


def test_krea2_every_other_checkpoints_even_blocks() -> None:
    model = _tiny_dit()

    model.enable_selective_checkpointing("every_other")

    assert model.selective_checkpoint == "every_other"
    assert [block.gradient_checkpointing for block in model.blocks] == [
        True,
        False,
        True,
        False,
        True,
        False,
    ]


def test_krea2_selective_off_clears_existing_flags() -> None:
    model = _tiny_dit()
    model.enable_gradient_checkpointing()

    model.enable_selective_checkpointing("off")

    assert not any(block.gradient_checkpointing for block in model.blocks)


@pytest.mark.parametrize("mode", ["adapter_aware", "mlp_only", "peak_blocks_mlp"])
def test_krea2_rejects_selective_modes_without_matching_semantics(mode: str) -> None:
    model = _tiny_dit()

    with pytest.raises(ValueError, match="supports only"):
        model.enable_selective_checkpointing(mode, blocks="4-5")
