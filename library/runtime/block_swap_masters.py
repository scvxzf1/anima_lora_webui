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

# bnb 是 NF4 (Krea-2 QLoRA) 路径才有的可选依赖; anima 环境通常不装. 这里
# 延迟导入 + 容错, 让 anima 的 bf16/int8 block swap 路径不依赖 bnb.
_PARAMS4BIT_CLS = None


def _params4bit_cls():
    global _PARAMS4BIT_CLS
    if _PARAMS4BIT_CLS is None:
        try:
            from bitsandbytes.nn import Params4bit

            _PARAMS4BIT_CLS = Params4bit
        except Exception:
            _PARAMS4BIT_CLS = False  # 标记"已尝试, 不可用"
    return _PARAMS4BIT_CLS if _PARAMS4BIT_CLS is not False else None


def is_params4bit_weight(weight) -> bool:
    """True 当 weight 是 bnb Params4bit (NF4 量化权重)."""
    cls = _params4bit_cls()
    return cls is not None and isinstance(weight, cls)


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


@dataclass(frozen=True)
class Params4bitBlockSwapCpuMaster:
    """CPU master for a frozen NF4 Linear4bit weight.

    存 bnb Params4bit 的 deepcopy 副本 (4-bit 码 + quant_state 含 state2 双重量化).
    整体 .to(device) 搬运时 bnb 同步搬 4-bit 码和 quant_state (modules.py:341),
    保持 bnb_quantized=True, forward 走 bnb.matmul_4bit 原生反量化路径.

    与 bf16/int8/slab 路径互斥: Params4bit 是复合容器, 不能进 slab 打包.
    见 docs/proposal/krea2_nf4_blockswap.md 方向 A.
    """

    params4bit: object  # Params4bit 实例 (deepcopy 副本, bnb_quantized=True)

    @property
    def dtype(self) -> torch.dtype:
        return getattr(self.params4bit, "quant_storage", torch.uint8)

    def stored_nbytes(self) -> int:
        # 4-bit 码 + quant_state (absmax/code/state2) 实际字节数.
        # params4bit.data 是打包后的 uint8 (2 值/byte).
        data = getattr(self.params4bit, "data", None)
        base = _tensor_nbytes(data) if isinstance(data, torch.Tensor) else 0
        qs = getattr(self.params4bit, "quant_state", None)
        return base + _quant_state_nbytes(qs)

    def restore_to_device(self, device: torch.device) -> object:
        """整体搬 master 到 device, 返回可挂回 module.weight 的 Params4bit.

        Linear4bit.to() 是原地操作, offloader 必须 deepcopy master 再搬, 不能
        直接对 master .to() (会污染 master). 调用方负责 deepcopy.
        """
        return self.params4bit.to(device)


def _quant_state_nbytes(qs) -> int:
    """估算 QuantState (absmax/code/state2) 字节数, 容错无 qs 时返回 0."""
    if qs is None:
        return 0
    total = 0
    for name in ("absmax", "code"):
        t = getattr(qs, name, None)
        if isinstance(t, torch.Tensor):
            total += _tensor_nbytes(t)
    if getattr(qs, "nested", False):
        s2 = getattr(qs, "state2", None)
        if s2 is not None:
            for name in ("absmax", "code"):
                t = getattr(s2, name, None)
                if isinstance(t, torch.Tensor):
                    total += _tensor_nbytes(t)
    return total


_CpuMaster = Union[torch.Tensor, Int8BlockSwapCpuMaster, Params4bitBlockSwapCpuMaster]


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
    # NF4 (bnb Params4bit) 分支: 整体 deepcopy 存为独立 master, 不走
    # detach/fp8/int8/slab 路径 (Params4bit 是复合容器, 不能当普通 tensor).
    # bnb __deepcopy__ (modules.py:258) 完整复制 4-bit 码 + quant_state (含
    # state2 双重量化). 探针 probe_nf4_blockswap_compat 已证此 master 搬回
    # GPU forward delta=0. 见 docs/proposal/krea2_nf4_blockswap.md 方向 A.
    if is_params4bit_weight(weight):
        import copy as _copy

        master_copy = _copy.deepcopy(weight)
        # 确保副本在 CPU (deepcopy 后可能仍在原 device, .to("cpu") 整体搬)
        if getattr(master_copy.data, "device", torch.device("cpu")).type != "cpu":
            master_copy = master_copy.to("cpu")
        master = Params4bitBlockSwapCpuMaster(params4bit=master_copy)
        stored = master.stored_nbytes()
        stats: dict[str, float] = {
            "source_bytes": float(stored),
            "stored_bytes": float(stored),
            "max_abs": 0.0,
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "relative_l2": 0.0,
            "saturated": 0.0,
            "int8_quantized": 0.0,
        }
        return master, stats

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
    if isinstance(master, Params4bitBlockSwapCpuMaster):
        # NF4 master 不进 slab 路径; parked 仅在 swap_plan 把它分流到 nf4_jobs
        # 后不再被调用. 返回 4-bit 码 tensor 作为占位 (用于 _build_swap_plan 的
        # shape 比较兜底, 实际搬运走 _restore_params4bit_master).
        return master.params4bit.data
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
    # Params4bit 分支不在此函数处理 (走 _restore_params4bit_master 整体搬运);
    # 若误入此函数, 走 restore_to_device 兜底, 但 dtype 参数被忽略 (NF4 不换 dtype).
    if isinstance(master, Params4bitBlockSwapCpuMaster):
        return master.restore_to_device(device)
    return master.to(device=device, dtype=dtype, non_blocking=non_blocking)


def _restore_params4bit_master(
    master: Params4bitBlockSwapCpuMaster, device: torch.device
) -> object:
    """从 NF4 CPU master 重建 GPU 上的 Params4bit (整体搬运, 不污染 master).

    Linear4bit.to() 是原地操作, 直接对 master .to() 会污染 master 的 device.
    offloader 持有独立 master 副本, 搬运时必须 deepcopy master 再 .to(device),
    探针 probe_nf4_blockswap_compat 已证此模式 5 轮交替搬运 delta=0.
    返回的 Params4bit 由调用方挂回 module.weight (整体赋, 非 .data).
    """
    import copy as _copy

    return _copy.deepcopy(master.params4bit).to(device)


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
