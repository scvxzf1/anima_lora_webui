"""CPU master helpers for runtime block swapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn

from library.runtime.device import is_weight_swap_excluded, should_move_weight_to_device
from library.runtime.int8_linear import (
    INT8_MAX,
    classify_frozen_linear_module,
    quantize_weight_per_channel,
)


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _weight_device_type(module: nn.Module) -> Optional[str]:
    weight = getattr(module, "weight", None)
    if weight is None:
        return None
    device = getattr(weight.data, "device", None)
    return getattr(device, "type", None)


def _ensure_weight_on_device(module: nn.Module, device: torch.device) -> None:
    weight = getattr(module, "weight", None)
    if weight is None:
        return
    # Never materialize / move ConvRot free-base meta placeholders.
    if is_weight_swap_excluded(module):
        return
    if _weight_device_type(module) != device.type:
        weight.data = weight.data.to(device, non_blocking=True)


def _can_swap_frozen_weight_to_cpu(module: nn.Module) -> bool:
    if is_weight_swap_excluded(module):
        return False
    return should_move_weight_to_device(
        module, torch.device("cpu"), include_trainable=False
    )


@dataclass(frozen=True)
class Int8BlockSwapCpuMaster:
    """CPU master for a frozen Linear weight stored as int8 + per-row scale."""

    quantized: torch.Tensor
    scale: torch.Tensor
    shape: tuple[int, ...]

    @property
    def dtype(self) -> torch.dtype:
        return self.quantized.dtype

    @property
    def numel(self) -> int:
        return int(self.quantized.numel())

    def stored_nbytes(self) -> int:
        return _tensor_nbytes(self.quantized) + _tensor_nbytes(self.scale)

    def to_tensor(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        non_blocking: bool = True,
    ) -> torch.Tensor:
        quantized = self.quantized.to(device=device, non_blocking=non_blocking)
        scale = self.scale.to(device=device, non_blocking=non_blocking)
        rows = quantized.reshape(self.shape[0], -1).to(torch.float32)
        restored = rows * scale.to(torch.float32)[:, None]
        return restored.reshape(self.shape).to(dtype=dtype)


_CpuMaster = Union[torch.Tensor, Int8BlockSwapCpuMaster]


def _int8_block_swap_candidate(
    module_name: str,
    weight: torch.Tensor,
    *,
    scope: str,
) -> bool:
    if weight.dim() != 2:
        return False
    return classify_frozen_linear_module(
        f"blocks.0.{module_name}",
        scope=scope,
    ) is not None


def _capture_cpu_master(
    weight: torch.Tensor,
    *,
    module_name: str = "",
    pin_memory: bool,
    transfer_dtype: str,
    int8_scope: str = "all",
) -> tuple[_CpuMaster, dict[str, float]]:
    master = weight.detach()
    if master.device.type != "cpu":
        master = master.to("cpu", non_blocking=False)
    stats: dict[str, float] = {
        "source_bytes": float(_tensor_nbytes(master)),
        "stored_bytes": float(_tensor_nbytes(master)),
        "max_abs": 0.0,
        "max_abs_error": 0.0,
        "mean_abs_error": 0.0,
        "relative_l2": 0.0,
        "saturated": 0.0,
        "int8_quantized": 0.0,
    }
    if transfer_dtype == "fp8_e4m3":
        source = master
        source_float = source.float()
        stats["max_abs"] = float(source_float.abs().max().item()) if source.numel() else 0.0
        finfo = torch.finfo(torch.float8_e4m3fn)
        stats["saturated"] = float(bool((source_float.abs() > finfo.max).any().item()))
        master = source.to(torch.float8_e4m3fn)
        restored = master.to(source.dtype).float()
        diff = source_float - restored
        stats["stored_bytes"] = float(_tensor_nbytes(master))
        stats["max_abs_error"] = float(diff.abs().max().item()) if source.numel() else 0.0
        stats["mean_abs_error"] = float(diff.abs().mean().item()) if source.numel() else 0.0
        denom = float(source_float.norm().item())
        stats["relative_l2"] = float(diff.norm().item()) / denom if denom > 0.0 else 0.0
    elif (
        transfer_dtype == "int8"
        and _int8_block_swap_candidate(module_name, master, scope=int8_scope)
    ):
        source = master
        source_float = source.float()
        quantized, scale = quantize_weight_per_channel(source)
        restored = (quantized.float() * scale.float()[:, None]).reshape(source.shape)
        diff = source_float - restored
        master = Int8BlockSwapCpuMaster(
            quantized=quantized.contiguous(),
            scale=scale.to(torch.float32).contiguous(),
            shape=tuple(int(dim) for dim in source.shape),
        )
        stats["max_abs"] = float(source_float.abs().max().item()) if source.numel() else 0.0
        stats["saturated"] = float((quantized.abs().to(torch.int16) == int(INT8_MAX)).any().item())
        stats["stored_bytes"] = float(master.stored_nbytes())
        stats["max_abs_error"] = float(diff.abs().max().item()) if source.numel() else 0.0
        stats["mean_abs_error"] = float(diff.abs().mean().item()) if source.numel() else 0.0
        denom = float(source_float.norm().item())
        stats["relative_l2"] = float(diff.norm().item()) / denom if denom > 0.0 else 0.0
        stats["int8_quantized"] = 1.0
    if pin_memory:
        try:
            if isinstance(master, Int8BlockSwapCpuMaster):
                master = Int8BlockSwapCpuMaster(
                    quantized=master.quantized.pin_memory(),
                    scale=master.scale.pin_memory(),
                    shape=master.shape,
                )
            else:
                master = master.pin_memory()
        except Exception:
            pass
    return master, stats


def _parked_cpu_master_tensor(master: _CpuMaster) -> torch.Tensor:
    if isinstance(master, Int8BlockSwapCpuMaster):
        return master.quantized
    return master


def _restore_cpu_master_tensor(
    master: _CpuMaster,
    *,
    device: torch.device,
    dtype: torch.dtype,
    non_blocking: bool = True,
) -> torch.Tensor:
    if isinstance(master, Int8BlockSwapCpuMaster):
        return master.to_tensor(device=device, dtype=dtype, non_blocking=non_blocking)
    return master.to(device=device, dtype=dtype, non_blocking=non_blocking)


def _restore_int8_cpu_master_into_tensor(
    master: Int8BlockSwapCpuMaster,
    dst: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    non_blocking: bool = True,
    chunk_rows: int = 0,
) -> torch.Tensor:
    if tuple(dst.shape) != master.shape:
        raise ValueError(
            f"int8 restore target shape mismatch: {tuple(dst.shape)} != {master.shape}"
        )
    if dst.device.type != device.type:
        raise ValueError(f"int8 restore target must be on {device}, got {dst.device}")
    if dst.dtype != dtype:
        raise ValueError(f"int8 restore target dtype mismatch: {dst.dtype} != {dtype}")
    dst_rows = dst.reshape(master.shape[0], -1)
    if chunk_rows <= 0 or chunk_rows >= master.shape[0]:
        quantized = master.quantized.to(device=device, non_blocking=non_blocking)
        scale = master.scale.to(device=device, dtype=dtype, non_blocking=non_blocking)
        rows = quantized.reshape(master.shape[0], -1).to(dtype=dtype)
        torch.mul(rows, scale[:, None], out=dst_rows)
        return dst

    source_rows = master.quantized.reshape(master.shape[0], -1)
    for row_start in range(0, master.shape[0], chunk_rows):
        row_end = min(master.shape[0], row_start + chunk_rows)
        quantized = source_rows[row_start:row_end].to(
            device=device,
            non_blocking=non_blocking,
        )
        scale = master.scale[row_start:row_end].to(
            device=device,
            dtype=dtype,
            non_blocking=non_blocking,
        )
        rows = quantized.to(dtype=dtype)
        torch.mul(rows, scale[:, None], out=dst_rows[row_start:row_end])
    return dst
