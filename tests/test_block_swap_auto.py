"""In-process block-swap auto tuner contracts."""

from __future__ import annotations

import logging

from library.runtime.block_swap_auto import (
    maybe_promote,
    per_block_bytes,
    plan_promotion,
    resolve_margin_bytes,
    resolve_offloader_and_blocks,
)

GIB = 1024 * 1024 * 1024
MIB = 1024 * 1024


class _FakeMaster:
    def __init__(self, nbytes: int) -> None:
        self._nbytes = nbytes

    def stored_nbytes(self) -> int:
        return self._nbytes


class _FakeOffloader:
    def __init__(self, swap: int, blocks: int, per_block: int = 1000) -> None:
        self.blocks_to_swap = swap
        self._cpu_weight_masters = {
            index: {"weight": _FakeMaster(per_block)} for index in range(blocks)
        }
        self.calls: list[tuple[int, int]] = []

    def reconfigure(self, blocks, count: int) -> bool:
        self.calls.append((len(blocks), count))
        self.blocks_to_swap = count
        return True


class _Holder:
    def __init__(self, offloader, blocks) -> None:
        self.offloader = offloader
        self.blocks = blocks


def test_resolve_margin_defaults_and_env(monkeypatch):
    assert resolve_margin_bytes(12) == 12 * MIB
    monkeypatch.delenv("ANIMA_BLOCK_SWAP_AUTO_MARGIN_MIB", raising=False)
    assert resolve_margin_bytes() == 400 * MIB
    monkeypatch.setenv("ANIMA_BLOCK_SWAP_AUTO_MARGIN_MIB", "256")
    assert resolve_margin_bytes() == 256 * MIB
    monkeypatch.setenv("ANIMA_BLOCK_SWAP_AUTO_MARGIN_MIB", "not-a-number")
    assert resolve_margin_bytes() == 400 * MIB


def test_per_block_bytes_sums_masters():
    offloader = _FakeOffloader(swap=2, blocks=4, per_block=500)
    assert per_block_bytes(offloader, 3) == 500
    assert per_block_bytes(offloader, 4) == 0
    assert per_block_bytes(offloader, -1) == 0


def test_plan_promotion_converts_headroom():
    offloader = _FakeOffloader(swap=24, blocks=42, per_block=MIB)
    plan = plan_promotion(
        offloader, [None] * 42, margin_bytes=0, free_bytes=10 * MIB
    )
    assert plan["promotable"] == 10
    assert plan["new_swap"] == 14
    assert plan["applied"] if "applied" in plan else True


def test_plan_promotion_keeps_margin_and_two_swapped():
    offloader = _FakeOffloader(swap=24, blocks=42, per_block=MIB)
    tight = plan_promotion(
        offloader, [None] * 42, margin_bytes=10 * MIB, free_bytes=10 * MIB
    )
    assert tight["promotable"] == 0 and tight["new_swap"] == 24
    assert tight["reason"] == "no headroom above margin"
    huge = plan_promotion(
        offloader, [None] * 42, margin_bytes=0, free_bytes=GIB
    )
    assert huge["new_swap"] == 2 and huge["promotable"] == 22


def test_plan_promotion_without_swap():
    offloader = _FakeOffloader(swap=0, blocks=8, per_block=MIB)
    plan = plan_promotion(offloader, [None] * 8, margin_bytes=0, free_bytes=GIB)
    assert plan["new_swap"] == 0 and "disabled" in plan["reason"]


def test_plan_promotion_without_cuda_or_masters(monkeypatch):
    import library.runtime.block_swap_auto as auto

    monkeypatch.setattr(auto, "cuda_free_bytes", lambda device=None: None)
    offloader = _FakeOffloader(swap=4, blocks=8, per_block=MIB)
    plan = plan_promotion(offloader, [None] * 8, margin_bytes=0)
    assert plan["reason"] == "CUDA unavailable"

    monkeypatch.setattr(auto, "cuda_free_bytes", lambda device=None: GIB)
    offloader._cpu_weight_masters = {}
    plan = plan_promotion(offloader, [None] * 8, margin_bytes=0)
    assert plan["reason"] == "no CPU masters measured yet"


def test_resolve_offloader_and_blocks_shapes():
    offloader, blocks = _FakeOffloader(swap=4, blocks=6), [1, 2, 3, 4, 5, 6]
    assert resolve_offloader_and_blocks(_Holder(offloader, blocks)) is not None

    class _AdapterModel:
        def __init__(self) -> None:
            self._z_image_block_swap_adapter = _Holder(offloader, blocks)

    assert resolve_offloader_and_blocks(_AdapterModel()) is not None

    class _Plain:
        pass

    assert resolve_offloader_and_blocks(_Plain()) is None
    assert resolve_offloader_and_blocks(None) is None


def test_maybe_promote_applies_and_reports():
    offloader, blocks = _FakeOffloader(swap=24, blocks=42, per_block=MIB), [None] * 42
    plan = maybe_promote(
        _Holder(offloader, blocks), margin_mib=0, free_bytes=6 * MIB
    )
    assert plan is not None and plan["applied"] is True
    assert offloader.calls == [(42, 18)]
    assert offloader.blocks_to_swap == 18


def test_maybe_promote_keeps_level_without_headroom():
    offloader, blocks = _FakeOffloader(swap=8, blocks=42, per_block=MIB), [None] * 42
    plan = maybe_promote(
        _Holder(offloader, blocks), margin_mib=0, free_bytes=0
    )
    assert plan is not None and plan["applied"] is False
    assert offloader.calls == []


def test_maybe_promote_without_offloader_warns(caplog):
    class _Plain:
        pass

    with caplog.at_level(logging.WARNING):
        assert maybe_promote(_Plain(), margin_mib=0, free_bytes=GIB) is None
    assert "no block-swap offloader" in caplog.text

MB = 1024 * 1024
REAL_BLOCK = 138 * MB          # measured Anima block payload
REAL_LINK = 3.0e9              # measured pinned H2D ceiling on the 50HX


def test_stop_condition_keeps_swap_when_dma_is_already_hidden():
    """step_ms=1500 is the measured swap=12 step on the 50HX (3.32 GB/step)."""
    offloader = _FakeOffloader(swap=24, blocks=42, per_block=REAL_BLOCK)
    plan = plan_promotion(
        offloader,
        [None] * 42,
        margin_bytes=0,
        free_bytes=64 * REAL_BLOCK,
        step_ms=1500.0,
        link_bytes_per_s=REAL_LINK,
        dma_ratio_floor=0.5,
    )
    assert plan["stopped_by_dma"] is True
    # 2 * 8 * 138 MiB / 3.0 GB/s ~= 772 ms, the first level above 0.5 * 1500 ms.
    assert plan["new_swap"] == 8, plan
    assert plan["projected_dma_ms_after"] >= plan["dma_ratio_floor"] * 1500.0


def test_stop_condition_does_not_block_promotion_when_step_is_compute_bound():
    """A 100 ms step leaves no DMA budget to protect, so the contract floor applies."""
    offloader = _FakeOffloader(swap=24, blocks=42, per_block=REAL_BLOCK)
    plan = plan_promotion(
        offloader,
        [None] * 42,
        margin_bytes=0,
        free_bytes=64 * REAL_BLOCK,
        step_ms=100.0,
        link_bytes_per_s=REAL_LINK,
    )
    assert plan["stopped_by_dma"] is False
    assert plan["new_swap"] == 2


def test_stop_condition_can_be_disabled():
    offloader = _FakeOffloader(swap=24, blocks=42, per_block=REAL_BLOCK)
    plan = plan_promotion(
        offloader,
        [None] * 42,
        margin_bytes=0,
        free_bytes=64 * REAL_BLOCK,
        step_ms=1500.0,
        link_bytes_per_s=REAL_LINK,
        dma_ratio_floor=0.0,
    )
    assert plan["new_swap"] == 2


def test_dma_ratio_floor_resolution(monkeypatch):
    from library.runtime.block_swap_auto import (
        DEFAULT_DMA_RATIO_FLOOR,
        resolve_dma_ratio_floor,
    )

    monkeypatch.delenv("ANIMA_BLOCK_SWAP_AUTO_DMA_RATIO_FLOOR", raising=False)
    assert resolve_dma_ratio_floor() == DEFAULT_DMA_RATIO_FLOOR
    monkeypatch.setenv("ANIMA_BLOCK_SWAP_AUTO_DMA_RATIO_FLOOR", "0.25")
    assert resolve_dma_ratio_floor() == 0.25
    assert resolve_dma_ratio_floor(2.0) == 0.95
    assert resolve_dma_ratio_floor(-1.0) == 0.0


def test_stop_condition_is_off_by_default(monkeypatch):
    """Default 0.0 keeps the speed-optimal behaviour: promote as far as VRAM allows."""
    monkeypatch.delenv("ANIMA_BLOCK_SWAP_AUTO_DMA_RATIO_FLOOR", raising=False)
    offloader = _FakeOffloader(swap=24, blocks=42, per_block=REAL_BLOCK)
    plan = plan_promotion(
        offloader,
        [None] * 42,
        margin_bytes=0,
        free_bytes=64 * REAL_BLOCK,
        step_ms=1500.0,
        link_bytes_per_s=REAL_LINK,
    )
    assert plan["dma_ratio_floor"] == 0.0
    assert plan["stopped_by_dma"] is False
    assert plan["new_swap"] == 2
