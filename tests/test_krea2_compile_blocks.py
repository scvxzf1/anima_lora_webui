from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import torch

from library.models.krea2_raw.dit import SingleStreamDiT


class _Block(torch.nn.Module):
    def _forward(self, value):
        return value + 1


def _tiny_dit(n_blocks: int = 4, blocks_to_swap: int = 0) -> SingleStreamDiT:
    model = SingleStreamDiT.__new__(SingleStreamDiT)
    torch.nn.Module.__init__(model)
    model.blocks = torch.nn.ModuleList([_Block() for _ in range(n_blocks)])
    model.blocks_to_swap = blocks_to_swap
    return model


def test_krea2_compile_blocks_compiles_only_resident_blocks() -> None:
    model = _tiny_dit(blocks_to_swap=2)
    compiled = Mock(side_effect=lambda fn, **kwargs: Mock(wraps=fn))

    with patch("torch.compile", compiled):
        model.compile_blocks(backend="eager", compile_block_scope="resident")

    assert compiled.call_count == 2
    assert hasattr(model.blocks[0], "_krea_compile_base_forward")
    assert not hasattr(model.blocks[2], "_krea_compile_base_forward")


def test_krea2_compile_blocks_all_includes_swapped_tail() -> None:
    model = _tiny_dit(blocks_to_swap=2)
    compiled = Mock(side_effect=lambda fn, **kwargs: Mock(wraps=fn))

    with patch("torch.compile", compiled):
        model.compile_blocks(backend="eager", compile_block_scope="all")

    assert compiled.call_count == 4


def test_krea2_compile_blocks_reuses_unwrapped_base() -> None:
    model = _tiny_dit()
    compiled = Mock(side_effect=lambda fn, **kwargs: Mock(wraps=fn))

    with patch("torch.compile", compiled):
        model.compile_blocks(backend="eager")
        first_bases = [block._krea_compile_base_forward for block in model.blocks]
        model.compile_blocks(backend="eager")

    assert compiled.call_count == 8
    assert [block._krea_compile_base_forward for block in model.blocks] == first_bases


def test_krea2_compile_blocks_rejects_dynamic_sequence() -> None:
    with pytest.raises(ValueError, match="fixed padded"):
        _tiny_dit().compile_blocks(backend="eager", dynamic_seq=True)
