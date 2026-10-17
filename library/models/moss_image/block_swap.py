"""MOSS main-block residency using the shared ModelOffloader lifecycle."""
from __future__ import annotations

import torch
from library.runtime.offloading import ModelOffloader


class MOSSBlockSwapMixin:
    def enable_block_swap(self, num_blocks: int, device: torch.device, **kwargs):
        return enable_moss_block_swap(self, num_blocks, device, **kwargs)

    def move_to_device_except_swap_blocks(self, device: torch.device) -> None:
        if not self.blocks_to_swap:
            self.to(device)
            return
        layers = self.layers
        self.layers = None
        try:
            self.to(device)
        finally:
            self.layers = layers

    def prepare_block_swap_before_forward(self, free_cache: bool = True) -> None:
        if self.blocks_to_swap:
            self.offloader.prepare_block_devices_before_forward(
                self.layers, free_cache=free_cache
            )

    def switch_block_swap_for_inference(self) -> None:
        if self.blocks_to_swap:
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self) -> None:
        if self.blocks_to_swap:
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()

    def pause_block_swap(self) -> bool:
        if not self.blocks_to_swap:
            return False
        for index in list(self.offloader.futures):
            self.offloader._wait_blocks_move(index, phase="pause")
        self.offloader.restore_blocks_to_device(self.layers, self.offloader.device)
        if self.offloader.cuda_available:
            torch.cuda.synchronize(self.offloader.device)
        self._paused_blocks_to_swap = self.blocks_to_swap
        self.blocks_to_swap = 0
        return True

    def resume_block_swap(self) -> bool:
        if self._paused_blocks_to_swap is None:
            return False
        self.blocks_to_swap = self._paused_blocks_to_swap
        self._paused_blocks_to_swap = None
        self.prepare_block_swap_before_forward()
        return True

    def flush_block_swap_profile(self, blocking: bool = False) -> None:
        if self.offloader is not None:
            self.offloader.flush_profile_events(blocking=blocking)


def enable_moss_block_swap(model, blocks_to_swap: int, device, **kwargs):
    """Enable swapping 1..depth-2 blocks; zero leaves swapping disabled."""
    if model.offloader is not None:
        raise RuntimeError("MOSS block swap is already configured")
    if type(blocks_to_swap) is not int or not 0 <= blocks_to_swap <= len(model.layers) - 2:
        raise ValueError(f"MOSS blocks_to_swap must be 0..{len(model.layers) - 2}")
    if blocks_to_swap == 0:
        return None
    offloader = ModelOffloader(
        model.layers, blocks_to_swap, torch.device(device), **kwargs
    )
    model.offloader = offloader
    model.blocks_to_swap = blocks_to_swap
    return offloader
