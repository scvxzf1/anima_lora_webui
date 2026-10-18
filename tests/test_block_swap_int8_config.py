from __future__ import annotations

import torch

from library.runtime import block_swap_config as config
from library.runtime.offloading import ModelOffloader


def test_staging_ring_resolution_prefers_config_value(monkeypatch) -> None:
    monkeypatch.setenv("ANIMA_BLOCK_SWAP_INT8_STAGING_RING", "0")

    assert config.resolve_block_swap_int8_staging_ring() == 0
    assert config.resolve_block_swap_int8_staging_ring(1) == 1
    # 0 is a meaningful value, not "unset".
    assert config.resolve_block_swap_int8_staging_ring(0) == 0
    monkeypatch.delenv("ANIMA_BLOCK_SWAP_INT8_STAGING_RING")
    assert config.resolve_block_swap_int8_staging_ring() == 2


def test_staging_ring_rejects_negative_values() -> None:
    import pytest

    with pytest.raises(ValueError):
        config.normalize_block_swap_int8_staging_ring(-1)
    with pytest.raises(ValueError):
        config.normalize_block_swap_int8_staging_ring("two")


def test_dual_stream_resolution_prefers_config_value(monkeypatch) -> None:
    monkeypatch.setenv("ANIMA_BLOCK_SWAP_INT8_DUAL_STREAM", "1")

    assert config.resolve_block_swap_int8_dual_stream() is True
    assert config.resolve_block_swap_int8_dual_stream(False) is False
    monkeypatch.delenv("ANIMA_BLOCK_SWAP_INT8_DUAL_STREAM")
    assert config.resolve_block_swap_int8_dual_stream() is False
    assert config.resolve_block_swap_int8_dual_stream("true") is True


def test_restore_chunk_rows_resolution(monkeypatch) -> None:
    monkeypatch.setenv("ANIMA_BLOCK_SWAP_INT8_RESTORE_CHUNK_ROWS", "64")
    assert config.resolve_block_swap_int8_restore_chunk_rows() == 64
    assert config.resolve_block_swap_int8_restore_chunk_rows(128) == 128


def _offloader(**kwargs) -> ModelOffloader:
    return ModelOffloader([], kwargs.pop("blocks_to_swap", 7), torch.device("cpu"), **kwargs)


def test_offloader_defaults_to_fused_restore() -> None:
    offloader = _offloader(transfer_dtype="int8")
    try:
        assert offloader.int8_restore_mode == "fused"
        assert offloader.int8_staging_ring == 2
        assert offloader.int8_dual_stream is False
    finally:
        offloader.thread_pool.shutdown()


def test_describe_block_swap_config_reports_effective_combo() -> None:
    offloader = _offloader(
        transfer_dtype="int8",
        int8_restore_mode="fused",
        int8_staging_ring=2,
        int8_dual_stream=True,
    )
    try:
        line = offloader.describe_block_swap_config()
    finally:
        offloader.thread_pool.shutdown()

    assert line.startswith("[BlockSwap] ")
    assert "swap=7" in line
    assert "transfer_dtype=int8" in line
    assert "int8_restore_mode=fused" in line
    assert "staging_ring=2 (2 x 48 MiB)" in line
    assert "dual_stream=True" in line


def test_describe_block_swap_config_marks_per_slot_ring() -> None:
    offloader = _offloader(
        blocks_to_swap=6,
        transfer_dtype="int8",
        int8_restore_mode="fused",
        int8_staging_ring=0,
    )
    try:
        line = offloader.describe_block_swap_config()
    finally:
        offloader.thread_pool.shutdown()

    assert "staging_ring=0 (6 x 48 MiB (per slot))" in line


def test_describe_block_swap_config_is_short_for_bf16() -> None:
    offloader = _offloader(transfer_dtype="bf16")
    try:
        line = offloader.describe_block_swap_config()
    finally:
        offloader.thread_pool.shutdown()

    assert line == "[BlockSwap] swap=7, transfer_dtype=bf16, restore_mode=slab"
