"""Block-swap adapter for the official Diffusers Z-Image transformer."""

from __future__ import annotations

import logging
from types import MethodType
from typing import Any, Callable

import torch
from torch import nn

from library.runtime.offloading import ModelOffloader


logger = logging.getLogger(__name__)


class ZImageBlockSwapAdapter:
    """Add the training block-swap protocol without copying Diffusers forward."""

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
        layers = getattr(model, "layers", None)
        if not isinstance(layers, nn.ModuleList):
            raise TypeError(
                "Z-Image block swap requires model.layers to be a ModuleList"
            )
        if blocks_to_swap < 1 or blocks_to_swap > len(layers) - 2:
            raise ValueError(
                "Z-Image blocks_to_swap must be between 1 and "
                f"{len(layers) - 2}; got {blocks_to_swap}"
            )

        self.model = model
        self.layers = layers
        self.layer_indices = {id(layer): index for index, layer in enumerate(layers)}
        self.offloader = ModelOffloader(
            layers,
            blocks_to_swap,
            device,
            profile_jsonl=profile_jsonl,
            transfer_dtype=transfer_dtype,
            restore_mode=restore_mode,
        )
        self._original_checkpoint_func = getattr(
            model, "_gradient_checkpointing_func", None
        )
        self._original_enable_gradient_checkpointing = (
            model.enable_gradient_checkpointing
        )
        self._original_forwards: list[Callable[..., Any]] = []

        model.blocks_to_swap = blocks_to_swap
        model.offloader = self.offloader
        model._z_image_block_swap_adapter = self
        self._bind_training_protocol()
        self._install_forward_dispatch()

    def _install_forward_dispatch(self) -> None:
        if bool(getattr(self.model, "gradient_checkpointing", False)):
            if self._original_checkpoint_func is None:
                raise RuntimeError(
                    "Z-Image gradient checkpointing is enabled without a checkpoint function"
                )
            self.model._gradient_checkpointing_func = self._checkpoint_with_swap

        for index, layer in enumerate(self.layers):
            original_forward = layer.forward
            self._original_forwards.append(original_forward)

            def forward_with_swap(
                _layer: nn.Module,
                *args: Any,
                _index: int = index,
                _forward: Callable[..., Any] = original_forward,
                **kwargs: Any,
            ) -> Any:
                # Non-reentrant checkpoint recompute calls the block directly.
                # Its residency is driven by ModelOffloader's backward hooks.
                checkpointed = bool(
                    torch.is_grad_enabled()
                    and getattr(self.model, "gradient_checkpointing", False)
                )
                if checkpointed or not getattr(self.model, "blocks_to_swap", 0):
                    return _forward(*args, **kwargs)
                return self._run_forward(_index, _forward, *args, **kwargs)

            layer.forward = MethodType(forward_with_swap, layer)

    def _checkpoint_with_swap(
        self, function: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        index = self.layer_indices.get(id(function))
        if index is None or not getattr(self.model, "blocks_to_swap", 0):
            return self._original_checkpoint_func(function, *args, **kwargs)
        self.offloader.wait_for_block(index)
        output = self._original_checkpoint_func(function, *args, **kwargs)
        self.offloader.submit_move_blocks(self.layers, index)
        return output

    def _run_forward(
        self,
        index: int,
        forward: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.offloader.wait_for_block(index)
        output = forward(*args, **kwargs)
        self.offloader.submit_move_blocks(self.layers, index)
        return output

    def _bind_training_protocol(self) -> None:
        protocol = {
            "enable_gradient_checkpointing": self.enable_gradient_checkpointing,
            "move_to_device_except_swap_blocks": self.move_to_device_except_swap_blocks,
            "prepare_block_swap_before_forward": self.prepare_block_swap_before_forward,
            "switch_block_swap_for_inference": self.switch_block_swap_for_inference,
            "switch_block_swap_for_training": self.switch_block_swap_for_training,
            "pause_block_swap": self.pause_block_swap,
            "resume_block_swap": self.resume_block_swap,
            "flush_block_swap_profile": self.flush_block_swap_profile,
        }
        for name, method in protocol.items():
            setattr(self.model, name, method)

    def enable_gradient_checkpointing(self, *args: Any, **kwargs: Any) -> None:
        """Preserve swap dispatch when bootstrap re-enables checkpointing."""
        self._original_enable_gradient_checkpointing(*args, **kwargs)
        self._original_checkpoint_func = self.model._gradient_checkpointing_func
        self.model._gradient_checkpointing_func = self._checkpoint_with_swap

    def move_to_device_except_swap_blocks(self, device: torch.device) -> None:
        layers = self.model.layers
        self.model.layers = None
        try:
            self.model.to(device)
        finally:
            self.model.layers = layers

    def prepare_block_swap_before_forward(self, free_cache: bool = True) -> None:
        if not getattr(self.model, "blocks_to_swap", 0):
            return
        self.offloader.prepare_block_devices_before_forward(
            self.layers, free_cache=free_cache
        )

    def switch_block_swap_for_inference(self) -> None:
        if not getattr(self.model, "blocks_to_swap", 0):
            return
        self.offloader.set_forward_only(True)
        self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self) -> None:
        if not getattr(self.model, "blocks_to_swap", 0):
            return
        self.offloader.set_forward_only(False)
        self.prepare_block_swap_before_forward()

    def pause_block_swap(self) -> bool:
        if not getattr(self.model, "blocks_to_swap", 0):
            return False
        for block_index in list(self.offloader.futures):
            self.offloader._wait_blocks_move(block_index, phase="pause")
        self.offloader.restore_blocks_to_device(self.layers, self.offloader.device)
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


def enable_z_image_block_swap(
    model: nn.Module,
    blocks_to_swap: int,
    device: torch.device,
    *,
    profile_jsonl: str | None = None,
    transfer_dtype: str | None = None,
    restore_mode: str | None = None,
) -> ZImageBlockSwapAdapter:
    """Attach the shared training offloader to Z-Image's 30 main layers."""
    existing = getattr(model, "_z_image_block_swap_adapter", None)
    if existing is not None:
        raise RuntimeError("Z-Image block swap is already enabled")
    adapter = ZImageBlockSwapAdapter(
        model,
        blocks_to_swap,
        device,
        profile_jsonl=profile_jsonl,
        transfer_dtype=transfer_dtype,
        restore_mode=restore_mode,
    )
    logger.info(
        "Z-Image block swap enabled: swapping %s/%s main layers on %s",
        blocks_to_swap,
        len(adapter.layers),
        device,
    )
    return adapter
