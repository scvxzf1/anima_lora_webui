"""Training block swap for the Diffusers Qwen Image 2.1 transformer."""

from __future__ import annotations

import logging
from types import MethodType
from typing import Any, Callable

import torch
from torch import nn

from library.runtime.offloading import ModelOffloader


logger = logging.getLogger(__name__)


class QwenImage21BlockSwapAdapter:
    """Drive the shared offloader around stock Diffusers block forwards."""

    def __init__(
        self,
        model: nn.Module,
        blocks_to_swap: int,
        device: torch.device,
        *,
        profile_jsonl: str | None = None,
        transfer_dtype: str | None = None,
        restore_mode: str | None = None,
    ) -> None:
        blocks = getattr(model, "transformer_blocks", None)
        if not isinstance(blocks, nn.ModuleList):
            raise TypeError("Qwen Image 2.1 block swap requires transformer_blocks ModuleList")
        if not 1 <= blocks_to_swap <= len(blocks) - 2:
            raise ValueError(
                f"Qwen Image 2.1 blocks_to_swap must be between 1 and {len(blocks) - 2}"
            )

        self.model = model
        self.blocks = blocks
        self.block_indices = {id(block): index for index, block in enumerate(blocks)}
        self.offloader = ModelOffloader(
            blocks,
            blocks_to_swap,
            device,
            profile_jsonl=profile_jsonl,
            transfer_dtype=transfer_dtype,
            restore_mode=restore_mode,
        )
        self._original_enable_checkpointing = model.enable_gradient_checkpointing
        self._original_checkpoint_func = getattr(model, "_gradient_checkpointing_func", None)
        self.inner_forwards: list[Callable[..., Any]] = []

        model.blocks_to_swap = blocks_to_swap
        model.offloader = self.offloader
        model._qwen_image_2_1_block_swap_adapter = self
        self._bind_training_protocol()
        self._install_block_forwards()

    def _install_block_forwards(self) -> None:
        if bool(getattr(self.model, "gradient_checkpointing", False)):
            if self._original_checkpoint_func is None:
                raise RuntimeError("Qwen Image 2.1 checkpointing has no checkpoint function")
            self.model._gradient_checkpointing_func = self._checkpoint_with_swap

        for index, block in enumerate(self.blocks):
            original_forward = block.forward
            self.inner_forwards.append(original_forward)

            def forward_with_swap(
                _block: nn.Module,
                *args: Any,
                _index: int = index,
                **kwargs: Any,
            ) -> Any:
                # The outer checkpoint wrapper drives the initial forward;
                # backward hooks drive its non-reentrant recomputation.
                checkpointed = bool(
                    torch.is_grad_enabled()
                    and getattr(self.model, "gradient_checkpointing", False)
                )
                forward = self.inner_forwards[_index]
                if checkpointed or not getattr(self.model, "blocks_to_swap", 0):
                    return forward(*args, **kwargs)
                return self._run_forward(_index, forward, *args, **kwargs)

            block.forward = MethodType(forward_with_swap, block)

    def _checkpoint_with_swap(
        self, function: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        index = self.block_indices.get(id(function))
        if index is None or not getattr(self.model, "blocks_to_swap", 0):
            return self._original_checkpoint_func(function, *args, **kwargs)
        self.offloader.wait_for_block(index)
        output = self._original_checkpoint_func(function, *args, **kwargs)
        self.offloader.submit_move_blocks(self.blocks, index)
        return output

    def _run_forward(
        self, index: int, forward: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        self.offloader.wait_for_block(index)
        output = forward(*args, **kwargs)
        self.offloader.submit_move_blocks(self.blocks, index)
        return output

    def _bind_training_protocol(self) -> None:
        for name in (
            "enable_gradient_checkpointing",
            "move_to_device_except_swap_blocks",
            "prepare_block_swap_before_forward",
            "switch_block_swap_for_inference",
            "switch_block_swap_for_training",
            "pause_block_swap",
            "resume_block_swap",
            "flush_block_swap_profile",
        ):
            setattr(self.model, name, getattr(self, name))

    def enable_gradient_checkpointing(self, *args: Any, **kwargs: Any) -> None:
        self._original_enable_checkpointing(*args, **kwargs)
        self._original_checkpoint_func = self.model._gradient_checkpointing_func
        self.model._gradient_checkpointing_func = self._checkpoint_with_swap

    def move_to_device_except_swap_blocks(self, device: torch.device) -> None:
        self.model.transformer_blocks = None
        try:
            self.model.to(device)
        finally:
            self.model.transformer_blocks = self.blocks

    def prepare_block_swap_before_forward(self, free_cache: bool = True) -> None:
        if getattr(self.model, "blocks_to_swap", 0):
            self.offloader.prepare_block_devices_before_forward(
                self.blocks, free_cache=free_cache
            )

    def switch_block_swap_for_inference(self) -> None:
        if getattr(self.model, "blocks_to_swap", 0):
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self) -> None:
        if getattr(self.model, "blocks_to_swap", 0):
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()

    def pause_block_swap(self) -> bool:
        if not getattr(self.model, "blocks_to_swap", 0):
            return False
        for index in list(self.offloader.futures):
            self.offloader._wait_blocks_move(index, phase="pause")
        self.offloader.restore_blocks_to_device(self.blocks, self.offloader.device)
        if self.offloader.cuda_available:
            torch.cuda.synchronize(self.offloader.device)
        self.model._paused_blocks_to_swap = self.model.blocks_to_swap
        self.model.blocks_to_swap = 0
        return True

    def resume_block_swap(self) -> bool:
        blocks_to_swap = getattr(self.model, "_paused_blocks_to_swap", None)
        if blocks_to_swap is None:
            return False
        self.model.blocks_to_swap = blocks_to_swap
        self.model._paused_blocks_to_swap = None
        self.prepare_block_swap_before_forward()
        return True

    def flush_block_swap_profile(self, blocking: bool = False) -> None:
        self.offloader.flush_profile_events(blocking=blocking)


def enable_qwen_image_2_1_block_swap(
    model: nn.Module,
    blocks_to_swap: int,
    device: torch.device,
    *,
    profile_jsonl: str | None = None,
    transfer_dtype: str | None = None,
    restore_mode: str | None = None,
) -> QwenImage21BlockSwapAdapter:
    if getattr(model, "_qwen_image_2_1_block_swap_adapter", None) is not None:
        raise RuntimeError("Qwen Image 2.1 block swap is already enabled")
    adapter = QwenImage21BlockSwapAdapter(
        model,
        blocks_to_swap,
        device,
        profile_jsonl=profile_jsonl,
        transfer_dtype=transfer_dtype,
        restore_mode=restore_mode,
    )
    logger.info(
        "Qwen Image 2.1 block swap enabled: %s/%s blocks on %s",
        blocks_to_swap,
        len(adapter.blocks),
        device,
    )
    return adapter
