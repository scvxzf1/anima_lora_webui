from concurrent.futures import ThreadPoolExecutor
import logging
import os
import threading
import time
from typing import Any, Optional, Union, Callable, Tuple
import torch
import torch.nn as nn

from library.runtime.device import (
    clean_memory_on_device,
    synchronize_device,
    weighs_to_device,
)
from library.runtime.block_swap_config import (
    _BLOCK_SWAP_INT8_RESTORE_MODES,
    _BLOCK_SWAP_INT8_SCOPES,
    _BLOCK_SWAP_RESTORE_MODES,
    _BLOCK_SWAP_TRANSFER_DTYPES,
    _DEFAULT_BLOCK_SWAP_PROFILE_POLL_INTERVAL_S,
    _block_swap_prefetch_depth,
    _block_swap_profile_poll_interval_s,
    _env_flag,
    _env_int,
    normalize_block_swap_int8_restore_mode,
    normalize_block_swap_int8_scope,
    normalize_block_swap_restore_mode,
    normalize_block_swap_transfer_dtype,
)
from library.runtime.block_swap_masters import (
    Int8BlockSwapCpuMaster,
    _can_swap_frozen_weight_to_cpu,
    _capture_cpu_master,
    _CpuMaster,
    _ensure_weight_on_device,
    _parked_cpu_master_tensor,
    _restore_cpu_master_tensor,
    _restore_int8_cpu_master_into_tensor,
    _tensor_nbytes,
    _weight_device_type,
)
from library.runtime.block_swap_profiler import BlockSwapProfiler, _resolve_profiler
from library.runtime.block_swap_payload import set_block_swap_payload_placement

logger = logging.getLogger(__name__)


def _try_foreach_h2d_copy(
    dst_tensors: list[torch.Tensor], src_tensors: list[torch.Tensor]
) -> bool:
    """Issue one foreach copy when the swap job list is dtype/device compatible."""

    foreach_copy = getattr(torch, "_foreach_copy_", None)
    if foreach_copy is None or len(dst_tensors) < 2 or len(dst_tensors) != len(src_tensors):
        return False
    for dst, src in zip(dst_tensors, src_tensors):
        if dst.device.type != "cuda" or src.device.type != "cpu":
            return False
        if dst.dtype != src.dtype or dst.numel() != src.numel():
            return False
    try:
        foreach_copy(dst_tensors, src_tensors, non_blocking=True)
        return True
    except Exception:
        return False


def _wait_for_async_cuda_copy_on_current_stream(
    device: torch.device,
    timings: dict[str, Any],
    *,
    wait_start_event: Optional[Any] = None,
    wait_end_event: Optional[Any] = None,
) -> None:
    """Order current-stream work after an async block-swap H2D copy.

    The offloader enqueues H2D copies on a side stream. The training block that
    consumes the restored weights runs on the current stream, so it only needs a
    CUDA stream dependency. Host-side ``Event.synchronize()`` is required for
    profiling timings but is pure overhead in production.
    """

    end_event = timings.get("_h2d_end_event")
    if end_event is None:
        return
    current_stream = torch.cuda.current_stream(device)
    if wait_start_event is not None:
        wait_start_event.record(current_stream)
        timings["_wait_start_event"] = wait_start_event
    current_stream.wait_event(end_event)
    if wait_end_event is not None:
        wait_end_event.record(current_stream)
        timings["_wait_end_event"] = wait_end_event


def _finalize_async_cuda_timings(
    timings: dict[str, Any], *, synchronize: bool, events_ready: bool = False
) -> tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Any], Optional[Any], bool]:
    end_event = timings.pop("_h2d_end_event", None)
    start_event = timings.pop("_h2d_start_event", None)
    ready_event = timings.pop("_ready_event", None)
    wait_start_event = timings.pop("_wait_start_event", None)
    wait_end_event = timings.pop("_wait_end_event", None)
    event_timing = bool(timings.pop("_event_timing", False))
    if end_event is not None and (synchronize or events_ready):
        if synchronize:
            end_event.synchronize()
        if start_event is not None:
            try:
                timings["h2d_ms"] = float(start_event.elapsed_time(end_event))
            except Exception:
                pass
        if ready_event is not None and start_event is not None:
            try:
                timings["event_wait_ms"] = max(
                    0.0, float(ready_event.elapsed_time(start_event))
                )
            except Exception:
                pass
    if wait_start_event is not None and wait_end_event is not None and (synchronize or events_ready):
        if synchronize:
            wait_end_event.synchronize()
        try:
            timings["gpu_wait_ms"] = max(
                0.0, float(wait_start_event.elapsed_time(wait_end_event))
            )
        except Exception:
            pass
    return ready_event, start_event, end_event, wait_start_event, wait_end_event, event_timing


def swap_weight_devices_cuda(
    device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module
) -> dict[str, float]:
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    weight_swap_jobs: list[Tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor]] = []

    modules_to_cpu = {k: v for k, v in layer_to_cpu.named_modules()}
    for module_to_cuda_name, module_to_cuda in layer_to_cuda.named_modules():
        if hasattr(module_to_cuda, "weight") and module_to_cuda.weight is not None:
            module_to_cpu = modules_to_cpu.get(module_to_cuda_name, None)
            if (
                module_to_cpu is not None
                and module_to_cpu.weight.shape == module_to_cuda.weight.shape
            ):
                if not (
                    _can_swap_frozen_weight_to_cpu(module_to_cpu)
                    and _can_swap_frozen_weight_to_cpu(module_to_cuda)
                ):
                    _ensure_weight_on_device(module_to_cpu, device)
                    _ensure_weight_on_device(module_to_cuda, device)
                    continue
                weight_swap_jobs.append(
                    (
                        module_to_cpu,
                        module_to_cuda,
                        module_to_cpu.weight.data,
                        module_to_cuda.weight.data,
                    )
                )
            else:
                _ensure_weight_on_device(module_to_cuda, device)

    torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value

    timings = {"d2h_ms": 0.0, "h2d_ms": 0.0}
    stream = torch.Stream(device="cuda")
    with torch.cuda.stream(stream):
        # cuda to cpu
        d2h_t0 = time.perf_counter()
        for (
            module_to_cpu,
            module_to_cuda,
            cuda_data_view,
            cpu_data_view,
        ) in weight_swap_jobs:
            cuda_data_view.record_stream(stream)
            module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

        stream.synchronize()
        timings["d2h_ms"] = (time.perf_counter() - d2h_t0) * 1000.0

        # cpu to cuda
        h2d_t0 = time.perf_counter()
        for (
            module_to_cpu,
            module_to_cuda,
            cuda_data_view,
            cpu_data_view,
        ) in weight_swap_jobs:
            cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
            module_to_cuda.weight.data = cuda_data_view

    stream.synchronize()
    timings["h2d_ms"] = (time.perf_counter() - h2d_t0) * 1000.0
    torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value
    return timings


def swap_weight_devices_no_cuda(
    device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module
) -> dict[str, float]:
    """
    not tested
    """
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    weight_swap_jobs: list[Tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor]] = []
    for module_to_cpu, module_to_cuda in zip(
        layer_to_cpu.modules(), layer_to_cuda.modules()
    ):
        if hasattr(module_to_cpu, "weight") and module_to_cpu.weight is not None:
            if not (
                _can_swap_frozen_weight_to_cpu(module_to_cpu)
                and _can_swap_frozen_weight_to_cpu(module_to_cuda)
            ):
                _ensure_weight_on_device(module_to_cpu, device)
                _ensure_weight_on_device(module_to_cuda, device)
                continue
            weight_swap_jobs.append(
                (
                    module_to_cpu,
                    module_to_cuda,
                    module_to_cpu.weight.data,
                    module_to_cuda.weight.data,
                )
            )

    # device to cpu
    d2h_t0 = time.perf_counter()
    for (
        module_to_cpu,
        module_to_cuda,
        cuda_data_view,
        cpu_data_view,
    ) in weight_swap_jobs:
        module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

    synchronize_device(device)
    d2h_ms = (time.perf_counter() - d2h_t0) * 1000.0

    # cpu to device
    h2d_t0 = time.perf_counter()
    for (
        module_to_cpu,
        module_to_cuda,
        cuda_data_view,
        cpu_data_view,
    ) in weight_swap_jobs:
        cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
        module_to_cuda.weight.data = cuda_data_view

    synchronize_device(device)
    h2d_ms = (time.perf_counter() - h2d_t0) * 1000.0
    return {"d2h_ms": d2h_ms, "h2d_ms": h2d_ms}


_SwapPlanEntry = Tuple[str, _CpuMaster, _CpuMaster, torch.dtype, torch.dtype]
_SwapPlan = tuple[tuple[_SwapPlanEntry, ...], tuple[str, ...]]
_SwapSlabView = Tuple[int, int, Tuple[int, ...], torch.dtype]
_SwapSlabPlan = tuple[tuple[tuple[str, _SwapSlabView], ...], int]


class Offloader:
    """
    common offloading class
    """

    def __init__(
        self,
        num_blocks: int,
        blocks_to_swap: int,
        device: torch.device,
        debug: bool = False,
        profile_jsonl: Optional[Union[str, BlockSwapProfiler]] = None,
        transfer_dtype: Optional[str] = None,
        restore_mode: Optional[str] = None,
        int8_restore_mode: Optional[str] = None,
        int8_restore_chunk_rows: Optional[int] = None,
        int8_scope: Optional[str] = None,
    ):
        self.num_blocks = num_blocks
        self.blocks_to_swap = blocks_to_swap
        self.device = device
        self.debug = debug
        self.transfer_dtype = normalize_block_swap_transfer_dtype(transfer_dtype)
        restore_mode = restore_mode or os.getenv("ANIMA_BLOCK_SWAP_RESTORE_MODE", "foreach")
        self.restore_mode = normalize_block_swap_restore_mode(restore_mode)
        int8_restore_mode = int8_restore_mode or os.getenv(
            "ANIMA_BLOCK_SWAP_INT8_RESTORE_MODE",
            "copy",
        )
        self.int8_restore_mode = normalize_block_swap_int8_restore_mode(
            int8_restore_mode
        )
        int8_scope = int8_scope or os.getenv("ANIMA_BLOCK_SWAP_INT8_SCOPE", "all")
        self.int8_scope = normalize_block_swap_int8_scope(int8_scope)
        if int8_restore_chunk_rows is None:
            int8_restore_chunk_rows = _env_int(
                "ANIMA_BLOCK_SWAP_INT8_RESTORE_CHUNK_ROWS",
                default=0,
            )
        self.int8_restore_chunk_rows = max(0, int(int8_restore_chunk_rows))

        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.futures = {}
        self.cuda_available = device.type == "cuda"
        self.profiler = _resolve_profiler(profile_jsonl)
        self.profile_step = 0
        self._cpu_weight_masters: Optional[list[dict[str, _CpuMaster]]] = None
        self._cpu_weight_master_dtypes: Optional[list[dict[str, torch.dtype]]] = None
        self._cpu_weight_master_slabs: Optional[list[Optional[torch.Tensor]]] = None
        self._cpu_weight_master_slab_plans: Optional[
            list[Optional[dict[str, _SwapSlabView]]]
        ] = None
        self._block_module_maps: list[Optional[dict[str, nn.Module]]] = [
            None for _ in range(num_blocks)
        ]
        self._swap_plan_cache: dict[tuple[int, int], _SwapPlan] = {}
        self._swap_slab_plan_cache: dict[tuple[int, int], _SwapSlabPlan] = {}
        self._swap_gpu_slab_cache: dict[
            int, tuple[torch.Tensor, tuple[tuple[str, torch.Tensor], ...]]
        ] = {}
        self._copy_stream: Optional[Any] = None
        # Per-slot copy streams let multiple in-flight H2D restores run on
        # independent streams instead of queueing on one. Keyed by slot id.
        self._copy_streams: dict[int, Any] = {}
        # Forward prefetch lead (blocks ahead to begin H2D restore). See
        # docs/findings/blockswap_baseline_20260806.md — bf16 transfer slightly
        # exceeds single-block compute on RTX 3080, so depth 2 hides it.
        self._prefetch_depth = _block_swap_prefetch_depth()
        self._timing_event_pool: list[Any] = []
        self._marker_event_pool: list[Any] = []
        self._event_pool_lock = threading.Lock()
        self._pending_profile_events: list[dict[str, Any]] = []
        self._pending_profile_lock = threading.Lock()
        self._profile_poller_stop = threading.Event()
        self._profile_poller: Optional[threading.Thread] = None
        self._profile_poll_interval_s = _block_swap_profile_poll_interval_s()
        self._profile_gpu_wait_timing = _env_flag(
            "ANIMA_BLOCK_SWAP_PROFILE_GPU_WAIT", default=False
        )
        self._slot_assignments: dict[int, dict[str, Any]] = {}
        self._frozen_weight_bytes_by_block: list[int] = [0 for _ in range(num_blocks)]
        self._bf16_weight_bytes_by_block: list[int] = [0 for _ in range(num_blocks)]
        self._fp8_weight_bytes_by_block: list[int] = [0 for _ in range(num_blocks)]
        self._fp8_max_abs_by_block: list[float] = [0.0 for _ in range(num_blocks)]
        self._fp8_max_abs_error_by_block: list[float] = [0.0 for _ in range(num_blocks)]
        self._fp8_mean_abs_error_by_block: list[float] = [0.0 for _ in range(num_blocks)]
        self._fp8_relative_l2_by_block: list[float] = [0.0 for _ in range(num_blocks)]
        self._fp8_saturated_tensors = 0
        self._int8_weight_bytes_by_block: list[int] = [0 for _ in range(num_blocks)]
        self._int8_max_abs_by_block: list[float] = [0.0 for _ in range(num_blocks)]
        self._int8_max_abs_error_by_block: list[float] = [0.0 for _ in range(num_blocks)]
        self._int8_mean_abs_error_by_block: list[float] = [0.0 for _ in range(num_blocks)]
        self._int8_relative_l2_by_block: list[float] = [0.0 for _ in range(num_blocks)]
        self._int8_saturated_tensors = 0
        self._int8_quantized_tensors = 0
        self._frozen_weight_master_bytes = 0
        self._bf16_master_bytes = 0
        self._fp8_master_bytes = 0
        self._int8_master_bytes = 0
        self._reported_weight_masters = False

    def _ensure_cpu_weight_masters(
        self, blocks: Union[list[nn.Module], nn.ModuleList]
    ) -> None:
        if self._cpu_weight_masters is not None:
            return

        pin_memory = self.cuda_available
        masters: list[dict[str, _CpuMaster]] = []
        master_dtypes: list[dict[str, torch.dtype]] = []
        master_slabs: list[Optional[torch.Tensor]] = []
        master_slab_plans: list[Optional[dict[str, _SwapSlabView]]] = []
        bytes_by_block: list[int] = []
        bf16_bytes_by_block: list[int] = []
        fp8_bytes_by_block: list[int] = []
        int8_bytes_by_block: list[int] = []
        fp8_max_abs_by_block: list[float] = []
        fp8_max_abs_error_by_block: list[float] = []
        fp8_mean_abs_error_by_block: list[float] = []
        fp8_relative_l2_by_block: list[float] = []
        int8_max_abs_by_block: list[float] = []
        int8_max_abs_error_by_block: list[float] = []
        int8_mean_abs_error_by_block: list[float] = []
        int8_relative_l2_by_block: list[float] = []
        total_bytes = 0
        total_bf16_bytes = 0
        total_fp8_bytes = 0
        total_int8_bytes = 0
        fp8_saturated_tensors = 0
        int8_saturated_tensors = 0
        int8_quantized_tensors = 0
        for block in blocks:
            block_masters: dict[str, _CpuMaster] = {}
            block_dtypes: dict[str, torch.dtype] = {}
            block_bytes = 0
            block_bf16_bytes = 0
            block_fp8_bytes = 0
            block_int8_bytes = 0
            block_max_abs = 0.0
            block_max_abs_error = 0.0
            block_mean_abs_errors: list[float] = []
            block_relative_l2s: list[float] = []
            block_int8_max_abs = 0.0
            block_int8_max_abs_error = 0.0
            block_int8_mean_abs_errors: list[float] = []
            block_int8_relative_l2s: list[float] = []
            for name, module in block.named_modules():
                weight = getattr(module, "weight", None)
                if weight is None or not _can_swap_frozen_weight_to_cpu(module):
                    continue
                master, stats = _capture_cpu_master(
                    weight.data,
                    module_name=name,
                    pin_memory=pin_memory,
                    transfer_dtype=self.transfer_dtype,
                    int8_scope=self.int8_scope,
                )
                block_masters[name] = master
                block_dtypes[name] = weight.data.dtype
                stored_bytes = int(stats["stored_bytes"])
                source_bytes = int(stats["source_bytes"])
                fp8_bytes = stored_bytes if self.transfer_dtype == "fp8_e4m3" else 0
                is_int8_quantized = bool(stats["int8_quantized"])
                int8_bytes = stored_bytes if is_int8_quantized else 0
                block_bytes += stored_bytes
                block_bf16_bytes += source_bytes
                block_fp8_bytes += fp8_bytes
                block_int8_bytes += int8_bytes
                if self.transfer_dtype == "fp8_e4m3":
                    block_max_abs = max(block_max_abs, float(stats["max_abs"]))
                    block_max_abs_error = max(
                        block_max_abs_error, float(stats["max_abs_error"])
                    )
                    block_mean_abs_errors.append(float(stats["mean_abs_error"]))
                    block_relative_l2s.append(float(stats["relative_l2"]))
                    fp8_saturated_tensors += int(stats["saturated"])
                if is_int8_quantized:
                    block_int8_max_abs = max(block_int8_max_abs, float(stats["max_abs"]))
                    block_int8_max_abs_error = max(
                        block_int8_max_abs_error, float(stats["max_abs_error"])
                    )
                    block_int8_mean_abs_errors.append(float(stats["mean_abs_error"]))
                    block_int8_relative_l2s.append(float(stats["relative_l2"]))
                    int8_saturated_tensors += int(stats["saturated"])
                    int8_quantized_tensors += 1
            block_masters, block_master_slab, block_master_slab_plan = self._pack_cpu_master_block(
                block_masters,
                pin_memory=pin_memory,
            )
            masters.append(block_masters)
            master_dtypes.append(block_dtypes)
            master_slabs.append(block_master_slab)
            master_slab_plans.append(block_master_slab_plan)
            bytes_by_block.append(block_bytes)
            bf16_bytes_by_block.append(block_bf16_bytes)
            fp8_bytes_by_block.append(block_fp8_bytes)
            int8_bytes_by_block.append(block_int8_bytes)
            fp8_max_abs_by_block.append(block_max_abs)
            fp8_max_abs_error_by_block.append(block_max_abs_error)
            fp8_mean_abs_error_by_block.append(
                sum(block_mean_abs_errors) / len(block_mean_abs_errors)
                if block_mean_abs_errors
                else 0.0
            )
            fp8_relative_l2_by_block.append(
                sum(block_relative_l2s) / len(block_relative_l2s)
                if block_relative_l2s
                else 0.0
            )
            int8_max_abs_by_block.append(block_int8_max_abs)
            int8_max_abs_error_by_block.append(block_int8_max_abs_error)
            int8_mean_abs_error_by_block.append(
                sum(block_int8_mean_abs_errors) / len(block_int8_mean_abs_errors)
                if block_int8_mean_abs_errors
                else 0.0
            )
            int8_relative_l2_by_block.append(
                sum(block_int8_relative_l2s) / len(block_int8_relative_l2s)
                if block_int8_relative_l2s
                else 0.0
            )
            total_bytes += block_bytes
            total_bf16_bytes += block_bf16_bytes
            total_fp8_bytes += block_fp8_bytes
            total_int8_bytes += block_int8_bytes

        self._cpu_weight_masters = masters
        self._cpu_weight_master_dtypes = master_dtypes
        self._cpu_weight_master_slabs = master_slabs
        self._cpu_weight_master_slab_plans = master_slab_plans
        self._ensure_block_module_maps(blocks)
        self._frozen_weight_bytes_by_block = bytes_by_block
        self._bf16_weight_bytes_by_block = bf16_bytes_by_block
        self._fp8_weight_bytes_by_block = fp8_bytes_by_block
        self._fp8_max_abs_by_block = fp8_max_abs_by_block
        self._fp8_max_abs_error_by_block = fp8_max_abs_error_by_block
        self._fp8_mean_abs_error_by_block = fp8_mean_abs_error_by_block
        self._fp8_relative_l2_by_block = fp8_relative_l2_by_block
        self._fp8_saturated_tensors = fp8_saturated_tensors
        self._int8_weight_bytes_by_block = int8_bytes_by_block
        self._int8_max_abs_by_block = int8_max_abs_by_block
        self._int8_max_abs_error_by_block = int8_max_abs_error_by_block
        self._int8_mean_abs_error_by_block = int8_mean_abs_error_by_block
        self._int8_relative_l2_by_block = int8_relative_l2_by_block
        self._int8_saturated_tensors = int8_saturated_tensors
        self._int8_quantized_tensors = int8_quantized_tensors
        self._frozen_weight_master_bytes = total_bytes
        self._bf16_master_bytes = total_bf16_bytes
        self._fp8_master_bytes = total_fp8_bytes
        self._int8_master_bytes = total_int8_bytes
        if not self._reported_weight_masters:
            logger.info(
                "Block swap frozen CPU masters prepared: "
                f"{total_bytes / (1024 ** 3):.2f} GiB across {len(blocks)} blocks "
                f"(transfer_dtype={self.transfer_dtype})"
            )
            self._reported_weight_masters = True
        if self.profiler is not None:
            self.profiler.write(
                {
                    "ev": "block_swap_config",
                    "step": self.profile_step,
                    "num_blocks": self.num_blocks,
                    "blocks_to_swap": self.blocks_to_swap,
                    "transfer_dtype": self.transfer_dtype,
                    "restore_mode": self.restore_mode,
                    "int8_restore_mode": self.int8_restore_mode,
                    "int8_restore_chunk_rows": self.int8_restore_chunk_rows,
                    "int8_scope": self.int8_scope,
                    "frozen_weight_master_bytes": total_bytes,
                    "frozen_weight_bytes_by_block": bytes_by_block,
                    "bf16_master_bytes": total_bf16_bytes,
                    "bf16_weight_bytes_by_block": bf16_bytes_by_block,
                    "fp8_master_bytes": total_fp8_bytes,
                    "fp8_weight_bytes_by_block": fp8_bytes_by_block,
                    "fp8_saturated_tensors": fp8_saturated_tensors,
                    "fp8_max_abs_by_block": fp8_max_abs_by_block,
                    "fp8_max_abs_error_by_block": fp8_max_abs_error_by_block,
                    "fp8_mean_abs_error_by_block": fp8_mean_abs_error_by_block,
                    "fp8_relative_l2_by_block": fp8_relative_l2_by_block,
                    "int8_master_bytes": total_int8_bytes,
                    "int8_weight_bytes_by_block": int8_bytes_by_block,
                    "int8_quantized_tensors": int8_quantized_tensors,
                    "int8_saturated_tensors": int8_saturated_tensors,
                    "int8_max_abs_by_block": int8_max_abs_by_block,
                    "int8_max_abs_error_by_block": int8_max_abs_error_by_block,
                    "int8_mean_abs_error_by_block": int8_mean_abs_error_by_block,
                    "int8_relative_l2_by_block": int8_relative_l2_by_block,
                    "h2d_only": True,
                }
            )

    def _get_block_module_map(
        self, block_idx: int, block: nn.Module
    ) -> dict[str, nn.Module]:
        modules = self._block_module_maps[block_idx]
        if modules is None:
            modules = {name: module for name, module in block.named_modules()}
            self._block_module_maps[block_idx] = modules
        return modules

    def _ensure_block_module_maps(
        self, blocks: Union[list[nn.Module], nn.ModuleList]
    ) -> None:
        for block_idx, block in enumerate(blocks):
            self._get_block_module_map(block_idx, block)

    def _pack_cpu_master_block(
        self,
        block_masters: dict[str, _CpuMaster],
        *,
        pin_memory: bool,
    ) -> tuple[
        dict[str, _CpuMaster],
        Optional[torch.Tensor],
        Optional[dict[str, _SwapSlabView]],
    ]:
        if not block_masters:
            return block_masters, None, None
        if any(isinstance(master, Int8BlockSwapCpuMaster) for master in block_masters.values()):
            return block_masters, None, None
        dtypes = {tensor.dtype for tensor in block_masters.values()}
        if len(dtypes) != 1:
            return block_masters, None, None
        slab_dtype = next(iter(dtypes))
        ordered = tuple(block_masters.items())
        slab_numel = sum(int(tensor.numel()) for _, tensor in ordered)
        slab = torch.empty(slab_numel, dtype=slab_dtype, pin_memory=pin_memory)
        slab_plan: dict[str, _SwapSlabView] = {}
        slab_views: dict[str, torch.Tensor] = {}
        offset = 0
        for name, tensor in ordered:
            flat_numel = int(tensor.numel())
            slab[offset : offset + flat_numel].copy_(tensor.view(-1), non_blocking=False)
            slab_plan[name] = (
                offset,
                flat_numel,
                tuple(int(dim) for dim in tensor.shape),
                tensor.dtype,
            )
            slab_views[name] = slab.narrow(0, offset, flat_numel).view(tensor.shape)
            offset += flat_numel
        return slab_views, slab, slab_plan

    def _warm_swap_plan_cache(
        self, blocks: Union[list[nn.Module], nn.ModuleList]
    ) -> None:
        if self._cpu_weight_masters is None or self._cpu_weight_master_dtypes is None:
            return
        first_swapped_block = self.num_blocks - self.blocks_to_swap
        if first_swapped_block <= 0:
            return
        for block_idx_to_cpu in range(first_swapped_block):
            block_idx_to_cuda = first_swapped_block + block_idx_to_cpu
            if block_idx_to_cuda >= self.num_blocks:
                break
            self._get_swap_plan(
                block_idx_to_cpu,
                blocks[block_idx_to_cpu],
                block_idx_to_cuda,
                blocks[block_idx_to_cuda],
                self._cpu_weight_masters[block_idx_to_cpu],
                self._cpu_weight_masters[block_idx_to_cuda],
                self._cpu_weight_master_dtypes[block_idx_to_cpu],
                self._cpu_weight_master_dtypes[block_idx_to_cuda],
            )

    def _build_swap_plan(
        self,
        block_idx_to_cpu: int,
        block_to_cpu: nn.Module,
        block_idx_to_cuda: int,
        block_to_cuda: nn.Module,
        source_masters: dict[str, _CpuMaster],
        target_masters: dict[str, _CpuMaster],
        source_dtypes: dict[str, torch.dtype],
        target_dtypes: dict[str, torch.dtype],
    ) -> _SwapPlan:
        modules_to_cpu = self._get_block_module_map(block_idx_to_cpu, block_to_cpu)
        modules_to_cuda = self._get_block_module_map(block_idx_to_cuda, block_to_cuda)
        swap_jobs: list[_SwapPlanEntry] = []
        fallback_names: list[str] = []
        for module_name, module_to_cuda in modules_to_cuda.items():
            weight = getattr(module_to_cuda, "weight", None)
            if weight is None:
                continue
            module_to_cpu = modules_to_cpu.get(module_name)
            source_master = source_masters.get(module_name)
            target_master = target_masters.get(module_name)
            source_dtype = source_dtypes.get(module_name)
            target_dtype = target_dtypes.get(module_name)
            if (
                module_to_cpu is not None
                and source_master is not None
                and target_master is not None
                and source_dtype is not None
                and target_dtype is not None
                and module_to_cpu.weight.shape == module_to_cuda.weight.shape
            ):
                swap_jobs.append(
                    (
                        module_name,
                        source_master,
                        target_master,
                        source_dtype,
                        target_dtype,
                    )
                )
            else:
                fallback_names.append(module_name)
        return tuple(swap_jobs), tuple(fallback_names)

    def _get_swap_plan(
        self,
        block_idx_to_cpu: int,
        block_to_cpu: nn.Module,
        block_idx_to_cuda: int,
        block_to_cuda: nn.Module,
        source_masters: dict[str, _CpuMaster],
        target_masters: dict[str, _CpuMaster],
        source_dtypes: dict[str, torch.dtype],
        target_dtypes: dict[str, torch.dtype],
    ) -> _SwapPlan:
        key = (block_idx_to_cpu, block_idx_to_cuda)
        plan = self._swap_plan_cache.get(key)
        if plan is None:
            plan = self._build_swap_plan(
                block_idx_to_cpu,
                block_to_cpu,
                block_idx_to_cuda,
                block_to_cuda,
                source_masters,
                target_masters,
                source_dtypes,
                target_dtypes,
            )
            self._swap_plan_cache[key] = plan
        return plan

    def _build_swap_slab_plan(self, swap_plan: _SwapPlan) -> _SwapSlabPlan:
        weight_swap_jobs, _ = swap_plan
        slab_entries: list[tuple[str, _SwapSlabView]] = []
        slab_numel = 0
        for module_name, _, target_master, _, target_dtype in weight_swap_jobs:
            if isinstance(target_master, Int8BlockSwapCpuMaster):
                return (), 0
            flat_numel = int(target_master.numel())
            slab_entries.append(
                (
                    module_name,
                    (
                        slab_numel,
                        flat_numel,
                        tuple(int(dim) for dim in target_master.shape),
                        target_dtype,
                    ),
                )
            )
            slab_numel += flat_numel
        return tuple(slab_entries), slab_numel

    def _get_swap_slab_plan(
        self,
        block_idx_to_cpu: int,
        block_idx_to_cuda: int,
        swap_plan: _SwapPlan,
    ) -> _SwapSlabPlan:
        key = (block_idx_to_cpu, block_idx_to_cuda)
        plan = self._swap_slab_plan_cache.get(key)
        if plan is None:
            plan = self._build_swap_slab_plan(swap_plan)
            self._swap_slab_plan_cache[key] = plan
        return plan

    def _get_cached_restore_slab(
        self,
        block_idx_to_cpu: int,
        block_idx_to_cuda: int,
        swap_plan: _SwapPlan,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, tuple[tuple[str, torch.Tensor], ...]]]:
        if self.restore_mode != "slab":
            return None
        weight_swap_jobs, _ = swap_plan
        if any(
            isinstance(source_master, Int8BlockSwapCpuMaster)
            or isinstance(target_master, Int8BlockSwapCpuMaster)
            for _, source_master, target_master, _, _ in weight_swap_jobs
        ):
            return None
        if self._cpu_weight_master_slabs is None or self._cpu_weight_master_slab_plans is None:
            return None
        cpu_slab = self._cpu_weight_master_slabs[block_idx_to_cuda]
        cpu_slab_plan = self._cpu_weight_master_slab_plans[block_idx_to_cuda]
        if cpu_slab is None or cpu_slab_plan is None:
            return None
        slot_count = self.num_blocks - self.blocks_to_swap
        if slot_count <= 0:
            return None
        slot_id = block_idx_to_cpu % slot_count
        slab_entries, slab_numel = self._get_swap_slab_plan(
            block_idx_to_cpu,
            block_idx_to_cuda,
            swap_plan,
        )
        if slab_numel <= 0:
            return None
        target_dtypes = {dtype for _, (_, _, _, dtype) in slab_entries}
        if len(target_dtypes) != 1:
            return None
        for module_name, (offset, numel, shape, _) in slab_entries:
            cpu_view = cpu_slab_plan.get(module_name)
            if cpu_view is None:
                return None
            cpu_offset, cpu_numel, cpu_shape, _cpu_dtype = cpu_view
            if cpu_offset != offset or cpu_numel != numel or cpu_shape != shape:
                return None
        cached = self._swap_gpu_slab_cache.get(slot_id)
        slab_dtype = next(iter(target_dtypes))
        if cached is not None:
            gpu_slab, gpu_views = cached
            if gpu_slab.dtype == slab_dtype and int(gpu_slab.numel()) == slab_numel:
                return cpu_slab, gpu_slab, gpu_views
        gpu_slab = torch.empty(slab_numel, dtype=slab_dtype, device=self.device)
        gpu_views = tuple(
            (
                module_name,
                gpu_slab.narrow(0, offset, numel).view(shape),
            )
            for module_name, (offset, numel, shape, _dtype) in slab_entries
        )
        self._swap_gpu_slab_cache[slot_id] = (gpu_slab, gpu_views)
        return cpu_slab, gpu_slab, gpu_views

    def _get_copy_stream(self) -> Any:
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=self.device)
        return self._copy_stream

    def _get_copy_stream_for_slot(self, slot_id: Optional[int]) -> Any:
        """Per-slot copy stream so parallel prefetches don't share one stream."""
        if slot_id is None:
            return self._get_copy_stream()
        stream = self._copy_streams.get(slot_id)
        if stream is None:
            stream = torch.cuda.Stream(device=self.device)
            self._copy_streams[slot_id] = stream
        return stream

    def _acquire_cuda_event(self, *, enable_timing: bool) -> Any:
        with self._event_pool_lock:
            pool = self._timing_event_pool if enable_timing else self._marker_event_pool
            if pool:
                return pool.pop()
        return torch.cuda.Event(enable_timing=enable_timing)

    def _release_cuda_event(self, event: Optional[Any], *, enable_timing: bool) -> None:
        if event is None:
            return
        with self._event_pool_lock:
            pool = self._timing_event_pool if enable_timing else self._marker_event_pool
            pool.append(event)

    def _event_is_ready(self, event: Optional[Any]) -> bool:
        if event is None:
            return True
        query = getattr(event, "query", None)
        if callable(query):
            try:
                return bool(query())
            except Exception:
                return False
        return False

    def _swap_slot_count(self) -> int:
        if self.blocks_to_swap is None:
            return 0
        return max(0, int(self.num_blocks) - int(self.blocks_to_swap))

    def _swap_slot_id(self, block_idx_to_cpu: int) -> Optional[int]:
        slot_count = self._swap_slot_count()
        if slot_count <= 0:
            return None
        return int(block_idx_to_cpu) % slot_count

    def _prefetch_lead_blocks(
        self,
        *,
        phase: str,
        block_idx_to_cpu: int,
        block_idx_to_cuda: int,
    ) -> Optional[int]:
        if self.num_blocks <= 0:
            return None
        if str(phase or "").startswith("backward"):
            return max(0, int(block_idx_to_cpu) - int(block_idx_to_cuda))
        return (int(block_idx_to_cuda) - int(block_idx_to_cpu)) % int(self.num_blocks)

    def _ensure_profile_poller(self) -> None:
        if self.profiler is None or not self.cuda_available:
            return
        if self._profile_poller is not None and self._profile_poller.is_alive():
            return
        self._profile_poller_stop.clear()

        def _poll() -> None:
            while not self._profile_poller_stop.is_set():
                try:
                    self.flush_profile_events(blocking=False)
                except Exception:
                    pass
                self._profile_poller_stop.wait(self._profile_poll_interval_s)

        self._profile_poller = threading.Thread(
            target=_poll,
            name="anima-block-swap-profile",
            daemon=True,
        )
        self._profile_poller.start()

    def _stop_profile_poller(self) -> None:
        thread = self._profile_poller
        if thread is None:
            return
        self._profile_poller_stop.set()
        thread.join(timeout=0.2)
        self._profile_poller = None

    def _build_profile_wait_event(
        self,
        *,
        phase: str,
        meta: dict[str, Any],
        timings: dict[str, Any],
        enqueued_at: Optional[float],
        wait_requested_at: float,
        wait_returned_at: float,
    ) -> dict[str, Any]:
        host_wait_ms = max(0.0, (wait_returned_at - wait_requested_at) * 1000.0)
        gpu_wait_ms = float(timings.get("gpu_wait_ms", 0.0))
        wait_ms = host_wait_ms + gpu_wait_ms
        ready_at = wait_requested_at + (wait_ms / 1000.0)
        timings["transfer_ms"] = float(timings.get("event_wait_ms", 0.0)) + float(
            timings.get("h2d_ms", 0.0)
        )
        queued_at = meta.get("queued_at")
        submit_lag_ms = (
            max(0.0, (ready_at - queued_at) * 1000.0) if queued_at is not None else 0.0
        )
        host_queue_ms = 0.0
        enqueue_to_ready_ms = 0.0
        submit_to_enqueue_ms = 0.0
        prefetch_runway_ms = 0.0
        enqueue_to_wait_ms = 0.0
        estimated_ready_slack_ms = 0.0
        if queued_at is not None and enqueued_at is not None:
            submit_to_enqueue_ms = max(0.0, (enqueued_at - queued_at) * 1000.0)
            host_queue_ms = max(
                0.0,
                submit_to_enqueue_ms - float(timings.get("enqueue_ms", 0.0)),
            )
            enqueue_to_ready_ms = max(0.0, (ready_at - enqueued_at) * 1000.0)
            enqueue_to_wait_ms = max(0.0, (wait_requested_at - enqueued_at) * 1000.0)
            estimated_ready_slack_ms = max(
                0.0,
                enqueue_to_wait_ms - float(timings.get("transfer_ms", 0.0)),
            )
        if queued_at is not None:
            prefetch_runway_ms = max(0.0, (wait_requested_at - queued_at) * 1000.0)
        return {
            "ev": "block_swap",
            "phase": phase or meta.get("phase") or "",
            "submit_phase": meta.get("phase") or "",
            "step": meta.get("step"),
            "block_idx": meta.get("block_idx"),
            "block_idx_to_cpu": meta.get("block_idx_to_cpu"),
            "slot_id": meta.get("slot_id"),
            "slot_count": meta.get("slot_count"),
            "submit_trigger_block_idx": meta.get("submit_trigger_block_idx"),
            "wait_trigger_block_idx": meta.get("wait_trigger_block_idx"),
            "prefetch_lead_blocks": meta.get("prefetch_lead_blocks"),
            "slot_previous_block_idx": meta.get("slot_previous_block_idx"),
            "slot_previous_step": meta.get("slot_previous_step"),
            "slot_previous_phase": meta.get("slot_previous_phase"),
            "slot_reuse_age_ms": float(meta.get("slot_reuse_age_ms") or 0.0),
            "slot_current_age_ms": prefetch_runway_ms,
            "transfer_dtype": self.transfer_dtype,
            "int8_scope": self.int8_scope,
            "wait_ms": wait_ms,
            "host_wait_ms": host_wait_ms,
            "gpu_wait_ms": gpu_wait_ms,
            "h2d_ms": float(timings.get("h2d_ms", 0.0)),
            "d2h_ms": float(timings.get("d2h_ms", 0.0)),
            "event_wait_ms": float(timings.get("event_wait_ms", 0.0)),
            "transfer_ms": float(timings.get("transfer_ms", 0.0)),
            "enqueue_ms": float(timings.get("enqueue_ms", 0.0)),
            "host_queue_ms": host_queue_ms,
            "submit_to_enqueue_ms": submit_to_enqueue_ms,
            "prefetch_runway_ms": prefetch_runway_ms,
            "enqueue_to_wait_ms": enqueue_to_wait_ms,
            "estimated_ready_slack_ms": estimated_ready_slack_ms,
            "enqueue_to_ready_ms": enqueue_to_ready_ms,
            "submit_lag_ms": submit_lag_ms,
            "queued_at": queued_at,
            "enqueued_at": enqueued_at,
            "wait_requested_at": wait_requested_at,
            "wait_returned_at": wait_returned_at,
            "ready_at": ready_at,
            "waited_at": time.time(),
        }

    def _queue_profile_wait_event(
        self,
        *,
        phase: str,
        meta: dict[str, Any],
        timings: dict[str, Any],
        enqueued_at: Optional[float],
        wait_requested_at: float,
        wait_returned_at: float,
    ) -> None:
        ready_event = timings.get("_ready_event")
        start_event = timings.get("_h2d_start_event")
        end_event = timings.get("_h2d_end_event")
        wait_start_event = timings.get("_wait_start_event")
        wait_end_event = timings.get("_wait_end_event")
        if end_event is None:
            self.profiler.write(
                self._build_profile_wait_event(
                    phase=phase,
                    meta=meta,
                    timings=timings,
                    enqueued_at=enqueued_at,
                    wait_requested_at=wait_requested_at,
                    wait_returned_at=wait_returned_at,
                )
            )
            return
        self._ensure_profile_poller()
        with self._pending_profile_lock:
            self._pending_profile_events.append(
                {
                    "phase": phase,
                    "meta": dict(meta),
                    "timings": timings,
                    "enqueued_at": enqueued_at,
                    "wait_requested_at": wait_requested_at,
                    "wait_returned_at": wait_returned_at,
                    "ready_event": ready_event,
                    "start_event": start_event,
                    "end_event": end_event,
                    "wait_start_event": wait_start_event,
                    "wait_end_event": wait_end_event,
                }
            )

    def flush_profile_events(self, *, blocking: bool = False) -> None:
        if self.profiler is None:
            return
        sleep_s = 0.001
        while True:
            with self._pending_profile_lock:
                pending = self._pending_profile_events
                self._pending_profile_events = []
            if not pending:
                return

            still_pending: list[dict[str, Any]] = []
            ready_samples: list[dict[str, Any]] = []
            for sample in pending:
                end_event = sample.get("end_event")
                completion_event = sample.get("wait_end_event") or end_event
                if self._event_is_ready(completion_event):
                    ready_samples.append(sample)
                    continue
                if blocking:
                    synchronize = getattr(completion_event, "synchronize", None)
                    if callable(synchronize):
                        try:
                            synchronize()
                            ready_samples.append(sample)
                            continue
                        except Exception:
                            pass
                still_pending.append(sample)

            events_to_write: list[dict[str, Any]] = []
            for sample in ready_samples:
                timings = sample["timings"]
                (
                    ready_event,
                    start_event,
                    end_event,
                    wait_start_event,
                    wait_end_event,
                    event_timing,
                ) = _finalize_async_cuda_timings(
                    timings,
                    synchronize=False,
                    events_ready=True,
                )
                events_to_write.append(
                    self._build_profile_wait_event(
                        phase=sample["phase"],
                        meta=sample["meta"],
                        timings=timings,
                        enqueued_at=sample["enqueued_at"],
                        wait_requested_at=float(sample["wait_requested_at"]),
                        wait_returned_at=float(sample["wait_returned_at"]),
                    )
                )
                self._release_cuda_event(ready_event, enable_timing=event_timing)
                self._release_cuda_event(start_event, enable_timing=event_timing)
                self._release_cuda_event(end_event, enable_timing=event_timing)
                self._release_cuda_event(wait_start_event, enable_timing=event_timing)
                self._release_cuda_event(wait_end_event, enable_timing=event_timing)
            self.profiler.write_many(events_to_write)

            if not still_pending:
                if not blocking:
                    return
                continue

            with self._pending_profile_lock:
                self._pending_profile_events = still_pending + self._pending_profile_events
            if not blocking:
                return
            time.sleep(sleep_s)

    def swap_weight_devices(
        self,
        block_idx_to_cpu: int,
        block_to_cpu: nn.Module,
        block_idx_to_cuda: int,
        block_to_cuda: nn.Module,
        *,
        ready_event: Optional[Any] = None,
        slot_id: Optional[int] = None,
    ):
        if self._cpu_weight_masters is None:
            if self.cuda_available:
                return swap_weight_devices_cuda(
                    self.device, block_to_cpu, block_to_cuda
                )
            else:
                return swap_weight_devices_no_cuda(
                    self.device, block_to_cpu, block_to_cuda
                )

        source_masters = self._cpu_weight_masters[block_idx_to_cpu]
        target_masters = self._cpu_weight_masters[block_idx_to_cuda]
        source_dtypes = (self._cpu_weight_master_dtypes or [])[block_idx_to_cpu]
        target_dtypes = (self._cpu_weight_master_dtypes or [])[block_idx_to_cuda]
        swap_plan = self._get_swap_plan(
            block_idx_to_cpu,
            block_to_cpu,
            block_idx_to_cuda,
            block_to_cuda,
            source_masters,
            target_masters,
            source_dtypes,
            target_dtypes,
        )
        if self.cuda_available:
            return self._swap_weight_devices_cached_cuda(
                block_idx_to_cpu,
                block_to_cpu,
                block_idx_to_cuda,
                block_to_cuda,
                swap_plan,
                ready_event=ready_event,
                slot_id=slot_id,
            )
        return self._swap_weight_devices_cached_no_cuda(
            block_idx_to_cpu,
            block_to_cpu,
            block_idx_to_cuda,
            block_to_cuda,
            swap_plan,
        )

    def _swap_weight_devices_cached_cuda(
        self,
        block_idx_to_cpu: int,
        layer_to_cpu: nn.Module,
        block_idx_to_cuda: int,
        layer_to_cuda: nn.Module,
        swap_plan: _SwapPlan,
        *,
        ready_event: Optional[Any] = None,
        slot_id: Optional[int] = None,
    ) -> dict[str, float]:
        assert layer_to_cpu.__class__ == layer_to_cuda.__class__
        modules_to_cpu = self._get_block_module_map(block_idx_to_cpu, layer_to_cpu)
        modules_to_cuda = self._get_block_module_map(block_idx_to_cuda, layer_to_cuda)
        weight_swap_jobs, fallback_names = swap_plan
        for module_name in fallback_names:
            module_to_cuda = modules_to_cuda.get(module_name)
            module_to_cpu = modules_to_cpu.get(module_name)
            if module_to_cuda is not None:
                _ensure_weight_on_device(module_to_cuda, self.device)
            if module_to_cpu is not None:
                _ensure_weight_on_device(module_to_cpu, self.device)

        profile_timings = self.profiler is not None
        timings: dict[str, Any] = {
            "d2h_ms": 0.0,
            "h2d_ms": 0.0,
            "event_wait_ms": 0.0,
            "_event_timing": profile_timings,
        }
        if ready_event is not None:
            timings["_ready_event"] = ready_event
        if not weight_swap_jobs:
            return timings

        stream = self._get_copy_stream_for_slot(slot_id)
        slab_bundle = self._get_cached_restore_slab(
            block_idx_to_cpu,
            block_idx_to_cuda,
            swap_plan,
        )
        with torch.cuda.stream(stream):
            if ready_event is not None:
                stream.wait_event(ready_event)

            # Frozen base weights never change during LoRA training. We keep a
            # CPU master for every swappable weight and only restore the next
            # block into the retired block's GPU storage. This removes the
            # per-swap D2H copy that made next-use prefetch miss its window.
            h2d_start = self._acquire_cuda_event(enable_timing=profile_timings)
            h2d_end = self._acquire_cuda_event(enable_timing=profile_timings)
            h2d_start.record(stream)
            cuda_bindings: list[tuple[nn.Module, torch.Tensor]] = []
            if slab_bundle is not None:
                cpu_slab, gpu_slab, gpu_view_items = slab_bundle
                gpu_views = dict(gpu_view_items)
                for (
                    module_name,
                    source_master,
                    target_master,
                    source_dtype,
                    target_dtype,
                ) in weight_swap_jobs:
                    module_to_cpu = modules_to_cpu[module_name]
                    module_to_cuda = modules_to_cuda[module_name]
                    module_to_cpu.weight.data = _parked_cpu_master_tensor(source_master)
                    cuda_bindings.append((module_to_cuda, gpu_views[module_name]))
                gpu_slab.record_stream(stream)
                gpu_slab.copy_(cpu_slab, non_blocking=True)
            else:
                cuda_dsts: list[torch.Tensor] = []
                cpu_srcs: list[torch.Tensor] = []
                has_int8_master = any(
                    isinstance(source_master, Int8BlockSwapCpuMaster)
                    or isinstance(target_master, Int8BlockSwapCpuMaster)
                    for _, source_master, target_master, _, _ in weight_swap_jobs
                )
                for (
                    module_name,
                    source_master,
                    target_master,
                    source_dtype,
                    target_dtype,
                ) in weight_swap_jobs:
                    module_to_cpu = modules_to_cpu[module_name]
                    module_to_cuda = modules_to_cuda[module_name]
                    cuda_data_view = module_to_cpu.weight.data
                    module_to_cpu.weight.data = _parked_cpu_master_tensor(source_master)
                    cuda_data_view.record_stream(stream)
                    cuda_dsts.append(cuda_data_view)
                    if (
                        has_int8_master
                        and self.int8_restore_mode == "direct_bind"
                        and isinstance(target_master, Int8BlockSwapCpuMaster)
                    ):
                        restored = _restore_cpu_master_tensor(
                            target_master,
                            device=self.device,
                            dtype=target_dtype,
                            non_blocking=True,
                        )
                        cuda_bindings.append((module_to_cuda, restored))
                    else:
                        cuda_bindings.append((module_to_cuda, cuda_data_view))
                    if not has_int8_master and isinstance(target_master, torch.Tensor):
                        cpu_srcs.append(target_master)
                if has_int8_master and self.int8_restore_mode == "direct_bind":
                    for cuda_data_view, (
                        _module_name,
                        _source_master,
                        target_master,
                        _source_dtype,
                        target_dtype,
                    ) in zip(cuda_dsts, weight_swap_jobs):
                        if isinstance(target_master, Int8BlockSwapCpuMaster):
                            continue
                        restored = _restore_cpu_master_tensor(
                            target_master,
                            device=self.device,
                            dtype=target_dtype,
                            non_blocking=True,
                        )
                        cuda_data_view.copy_(restored, non_blocking=True)
                elif has_int8_master and self.int8_restore_mode == "reuse_storage":
                    for cuda_data_view, (
                        _module_name,
                        _source_master,
                        target_master,
                        _source_dtype,
                        target_dtype,
                    ) in zip(cuda_dsts, weight_swap_jobs):
                        if isinstance(target_master, Int8BlockSwapCpuMaster):
                            _restore_int8_cpu_master_into_tensor(
                                target_master,
                                cuda_data_view,
                                device=self.device,
                                dtype=target_dtype,
                                non_blocking=True,
                                chunk_rows=self.int8_restore_chunk_rows,
                            )
                            continue
                        restored = _restore_cpu_master_tensor(
                            target_master,
                            device=self.device,
                            dtype=target_dtype,
                            non_blocking=True,
                        )
                        cuda_data_view.copy_(restored, non_blocking=True)
                elif has_int8_master:
                    for cuda_data_view, (
                        _module_name,
                        _source_master,
                        target_master,
                        _source_dtype,
                        target_dtype,
                    ) in zip(cuda_dsts, weight_swap_jobs):
                        restored = _restore_cpu_master_tensor(
                            target_master,
                            device=self.device,
                            dtype=target_dtype,
                            non_blocking=True,
                        )
                        cuda_data_view.copy_(restored, non_blocking=True)
                elif not _try_foreach_h2d_copy(cuda_dsts, cpu_srcs):
                    for cuda_data_view, target_master in zip(cuda_dsts, cpu_srcs):
                        cuda_data_view.copy_(target_master, non_blocking=True)
            for module_to_cuda, cuda_data_view in cuda_bindings:
                module_to_cuda.weight.data = cuda_data_view
            h2d_end.record(stream)

        timings["_h2d_start_event"] = h2d_start
        timings["_h2d_end_event"] = h2d_end
        return timings

    def _swap_weight_devices_cached_no_cuda(
        self,
        block_idx_to_cpu: int,
        layer_to_cpu: nn.Module,
        block_idx_to_cuda: int,
        layer_to_cuda: nn.Module,
        swap_plan: _SwapPlan,
    ) -> dict[str, float]:
        assert layer_to_cpu.__class__ == layer_to_cuda.__class__

        t0 = time.perf_counter()
        modules_to_cpu = self._get_block_module_map(block_idx_to_cpu, layer_to_cpu)
        modules_to_cuda = self._get_block_module_map(block_idx_to_cuda, layer_to_cuda)
        weight_swap_jobs, fallback_names = swap_plan
        for module_name in fallback_names:
            module_to_cuda = modules_to_cuda.get(module_name)
            module_to_cpu = modules_to_cpu.get(module_name)
            if module_to_cuda is not None:
                _ensure_weight_on_device(module_to_cuda, self.device)
            if module_to_cpu is not None:
                _ensure_weight_on_device(module_to_cpu, self.device)
        for (
            module_name,
            source_master,
            target_master,
            source_dtype,
            target_dtype,
        ) in weight_swap_jobs:
            module_to_cpu = modules_to_cpu[module_name]
            module_to_cuda = modules_to_cuda[module_name]
            module_to_cpu.weight.data = _parked_cpu_master_tensor(source_master)
            module_to_cuda.weight.data = _restore_cpu_master_tensor(
                target_master,
                device=self.device,
                dtype=target_dtype,
                non_blocking=True,
            )
        synchronize_device(self.device)
        return {"d2h_ms": 0.0, "h2d_ms": (time.perf_counter() - t0) * 1000.0}

    def restore_blocks_to_device(
        self, blocks: Union[list[nn.Module], nn.ModuleList], device: torch.device
    ) -> None:
        """Restore all cached frozen masters to their execution dtype on device."""
        if self._cpu_weight_masters is None or self._cpu_weight_master_dtypes is None:
            for b in blocks:
                weighs_to_device(b, device)
            return

        self._ensure_block_module_maps(blocks)
        for block_idx, (block, masters, dtypes) in enumerate(
            zip(blocks, self._cpu_weight_masters, self._cpu_weight_master_dtypes)
        ):
            modules = self._get_block_module_map(block_idx, block)
            for name, master in masters.items():
                module = modules.get(name)
                weight = getattr(module, "weight", None) if module is not None else None
                if weight is None:
                    continue
                dtype = dtypes.get(name, weight.data.dtype)
                weight.data = _restore_cpu_master_tensor(
                    master,
                    device=device,
                    dtype=dtype,
                    non_blocking=True,
                )
            weighs_to_device(block, device, include_trainable=True)
            set_block_swap_payload_placement(block, active=False)
        synchronize_device(device)

    def _submit_move_blocks(
        self,
        blocks,
        block_idx_to_cpu,
        block_idx_to_cuda,
        *,
        phase: str,
    ):
        if block_idx_to_cuda in self.futures:
            return

        ready_event = None
        profile_timings = self.profiler is not None
        if self.cuda_available and self._cpu_weight_masters is not None:
            ready_event = self._acquire_cuda_event(enable_timing=profile_timings)
            ready_event.record(torch.cuda.current_stream(self.device))

        def move_blocks(
            bidx_to_cpu,
            block_to_cpu,
            bidx_to_cuda,
            block_to_cuda,
            event,
            submitted_at,
            slot_id,
        ):
            if self.debug:
                start_time = time.perf_counter()
                print(
                    f"Move block {bidx_to_cpu} to CPU and block {bidx_to_cuda} to {'CUDA' if self.cuda_available else 'device'}"
                )

            transfer_t0 = time.perf_counter()
            timings = self.swap_weight_devices(
                bidx_to_cpu,
                block_to_cpu,
                bidx_to_cuda,
                block_to_cuda,
                ready_event=event,
                slot_id=slot_id,
            )
            enqueued_at = time.time()
            timings["enqueue_ms"] = (time.perf_counter() - transfer_t0) * 1000.0

            if self.debug:
                print(
                    f"Moved blocks {bidx_to_cpu} and {bidx_to_cuda} in {time.perf_counter() - start_time:.2f}s"
                )
            return bidx_to_cpu, bidx_to_cuda, timings, enqueued_at

        block_to_cpu = blocks[block_idx_to_cpu]
        block_to_cuda = blocks[block_idx_to_cuda]
        queued_at = time.time()
        slot_count = self._swap_slot_count()
        slot_id = self._swap_slot_id(block_idx_to_cpu)
        previous_slot = (
            self._slot_assignments.get(slot_id) if slot_id is not None else None
        )
        slot_reuse_age_ms = 0.0
        if previous_slot is not None and previous_slot.get("queued_at") is not None:
            slot_reuse_age_ms = max(
                0.0,
                (queued_at - float(previous_slot["queued_at"])) * 1000.0,
            )
        if slot_id is not None:
            self._slot_assignments[slot_id] = {
                "block_idx": block_idx_to_cuda,
                "queued_at": queued_at,
                "step": self.profile_step,
                "phase": phase,
            }

        self.futures[block_idx_to_cuda] = (
            self.thread_pool.submit(
                move_blocks,
                block_idx_to_cpu,
                block_to_cpu,
                block_idx_to_cuda,
                block_to_cuda,
                ready_event,
                queued_at,
                slot_id,
            ),
            {
                "phase": phase,
                "block_idx": block_idx_to_cuda,
                "block_idx_to_cpu": block_idx_to_cpu,
                "queued_at": queued_at,
                "step": self.profile_step,
                "slot_id": slot_id,
                "slot_count": slot_count,
                "submit_trigger_block_idx": block_idx_to_cpu,
                "prefetch_lead_blocks": self._prefetch_lead_blocks(
                    phase=phase,
                    block_idx_to_cpu=block_idx_to_cpu,
                    block_idx_to_cuda=block_idx_to_cuda,
                ),
                "slot_previous_block_idx": (
                    previous_slot.get("block_idx")
                    if previous_slot is not None
                    else None
                ),
                "slot_previous_step": (
                    previous_slot.get("step") if previous_slot is not None else None
                ),
                "slot_previous_phase": (
                    previous_slot.get("phase") if previous_slot is not None else None
                ),
                "slot_reuse_age_ms": slot_reuse_age_ms,
            },
        )

    def _wait_blocks_move(self, block_idx, *, phase: str = ""):
        if block_idx not in self.futures:
            return

        if self.debug:
            print(f"Wait for block {block_idx}")
            start_time = time.perf_counter()

        future, meta = self.futures.pop(block_idx)
        wait_t0 = time.perf_counter()
        wait_requested_at = time.time()
        if "wait_trigger_block_idx" not in meta:
            meta["wait_trigger_block_idx"] = (
                int(block_idx) + 1 if str(phase or "").startswith("backward") else block_idx
            )
        _, bidx_to_cuda, timings, enqueued_at = future.result()
        if self.cuda_available:
            wait_start_event = None
            wait_end_event = None
            if (
                self.profiler is not None
                and self._profile_gpu_wait_timing
                and timings.get("_h2d_end_event") is not None
            ):
                wait_start_event = self._acquire_cuda_event(enable_timing=True)
                wait_end_event = self._acquire_cuda_event(enable_timing=True)
            _wait_for_async_cuda_copy_on_current_stream(
                self.device,
                timings,
                wait_start_event=wait_start_event,
                wait_end_event=wait_end_event,
            )
        wait_returned_at = time.time()

        assert block_idx == bidx_to_cuda, (
            f"Block index mismatch: {block_idx} != {bidx_to_cuda}"
        )
        if self.profiler is not None:
            self._queue_profile_wait_event(
                phase=phase,
                meta=meta,
                timings=timings,
                enqueued_at=enqueued_at,
                wait_requested_at=wait_requested_at,
                wait_returned_at=wait_returned_at,
            )
        else:
            (
                ready_event,
                start_event,
                end_event,
                wait_start_event,
                wait_end_event,
                event_timing,
            ) = _finalize_async_cuda_timings(
                timings,
                synchronize=False,
                events_ready=self._event_is_ready(timings.get("_h2d_end_event")),
            )
            self._release_cuda_event(ready_event, enable_timing=event_timing)
            self._release_cuda_event(start_event, enable_timing=event_timing)
            self._release_cuda_event(end_event, enable_timing=event_timing)
            self._release_cuda_event(wait_start_event, enable_timing=event_timing)
            self._release_cuda_event(wait_end_event, enable_timing=event_timing)

        if self.debug:
            print(
                f"Waited for block {block_idx}: {time.perf_counter() - start_time:.2f}s"
            )


# Gradient tensors
_grad_t = Union[tuple[torch.Tensor, ...], torch.Tensor]


class ModelOffloader(Offloader):
    """
    supports forward offloading
    """

    def __init__(
        self,
        blocks: Union[list[nn.Module], nn.ModuleList],
        blocks_to_swap: int,
        device: torch.device,
        supports_backward: bool = True,
        debug: bool = False,
        profile_jsonl: Optional[Union[str, BlockSwapProfiler]] = None,
        transfer_dtype: Optional[str] = None,
        restore_mode: Optional[str] = None,
        int8_restore_mode: Optional[str] = None,
        int8_restore_chunk_rows: Optional[int] = None,
        int8_scope: Optional[str] = None,
    ):
        super().__init__(
            len(blocks),
            blocks_to_swap,
            device,
            debug,
            profile_jsonl,
            transfer_dtype=transfer_dtype,
            restore_mode=restore_mode,
            int8_restore_mode=int8_restore_mode,
            int8_restore_chunk_rows=int8_restore_chunk_rows,
            int8_scope=int8_scope,
        )

        self.supports_backward = supports_backward
        self.forward_only = (
            not supports_backward
        )  # forward only offloading: can be changed to True for inference

        if self.supports_backward:
            # register backward hooks
            self.remove_handles = []
            for i, block in enumerate(blocks):
                hook = self.create_backward_hook(blocks, i)
                if hook is not None:
                    handle = block.register_full_backward_hook(hook)
                    self.remove_handles.append(handle)

    def set_forward_only(self, forward_only: bool):
        # switching must wait for all pending transfers
        for block_idx in list(self.futures.keys()):
            self._wait_blocks_move(block_idx, phase="mode_switch")
        self.flush_profile_events(blocking=True)
        self.forward_only = forward_only

    def __del__(self):
        try:
            self.flush_profile_events(blocking=True)
        except Exception:
            pass
        try:
            self._stop_profile_poller()
        except Exception:
            pass
        if self.supports_backward:
            for handle in self.remove_handles:
                handle.remove()

    def create_backward_hook(
        self, blocks: Union[list[nn.Module], nn.ModuleList], block_index: int
    ) -> Optional[Callable[[nn.Module, _grad_t, _grad_t], Union[None, _grad_t]]]:
        first_swapped_block = self.num_blocks - self.blocks_to_swap
        # Once backward for a tail block has completed, its GPU storage is no
        # longer needed by autograd and can immediately restore the matching
        # early block. For Anima's fixed 0 -> 27 order this is exact next-use:
        # 27 -> 11, 26 -> 10, ..., 16 -> 0 when blocks_to_swap=12.
        swapping = block_index >= first_swapped_block
        waiting = block_index > 0 and block_index <= self.blocks_to_swap

        if not swapping and not waiting:
            return None

        # create  hook
        block_idx_to_cpu = block_index
        block_idx_to_cuda = block_index - first_swapped_block
        block_idx_to_wait = block_index - 1

        def backward_hook(module: nn.Module, grad_input: _grad_t, grad_output: _grad_t):
            if self.debug:
                print(f"Backward hook for block {block_index}")

            if swapping:
                self._submit_move_blocks(
                    blocks,
                    block_idx_to_cpu,
                    block_idx_to_cuda,
                    phase="backward_prefetch",
                )
            if waiting:
                self._wait_blocks_move(block_idx_to_wait, phase="backward_wait")
            return None

        return backward_hook

    def prepare_block_devices_before_forward(
        self,
        blocks: Union[list[nn.Module], nn.ModuleList],
        free_cache: bool = True,
    ):
        # ``free_cache=False`` skips the trailing ``empty_cache`` so callers
        # that re-enter another forward in the same step (e.g. the FECL
        # no-grad base pass for the FeRA stacked-experts path) don't release
        # the caching allocator's blocks just to have the next forward
        # re-grow them — the visible nvidia-smi swing was ~1 GB per step on
        # a 5060 Ti without changing peak allocated.
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        if self.debug:
            print("Prepare block devices before forward")

        # wait for all pending transfers
        for block_idx in list(self.futures.keys()):
            self._wait_blocks_move(block_idx, phase="prepare")
        self._slot_assignments = {}
        self.profile_step += 1
        self._ensure_cpu_weight_masters(blocks)
        self._warm_swap_plan_cache(blocks)

        for block in blocks:
            set_block_swap_payload_placement(block, active=True)

        for b in blocks[0 : self.num_blocks - self.blocks_to_swap]:
            b.to(self.device)
            weighs_to_device(b, self.device)  # make sure weights are on device

        for b in blocks[self.num_blocks - self.blocks_to_swap :]:
            b.to(
                self.device
            )  # move block to device first. this makes sure that buffers (non weights) are on the device
            weighs_to_device(
                b, torch.device("cpu"), include_trainable=False
            )  # keep adapter/trainable weights on the training device

        synchronize_device(self.device)
        if free_cache:
            clean_memory_on_device(self.device)

    def wait_for_block(self, block_idx: int):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self._wait_blocks_move(block_idx, phase="forward_wait")

    def submit_move_blocks(
        self, blocks: Union[list[nn.Module], nn.ModuleList], block_idx: int
    ):
        # check if blocks_to_swap is enabled
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        # Prefetch with a lead of ``_prefetch_depth`` blocks so the H2D restore
        # stays ahead of compute (training path only). Baseline (docs/findings/
        # blockswap_baseline_20260806.md): bf16 block transfer slightly exceeds
        # one block's forward compute on RTX 3080, so a lead of 1 stalls each
        # block by ~2ms. Forward-only (inference) keeps the exact lead of 1:
        # its slot rotation reuses GPU storage immediately and a deeper lead
        # would overwrite a slot before its block has run.
        depth = 1 if self.forward_only else max(1, int(self._prefetch_depth))
        for step in range(depth):
            # if backward is enabled, we do not swap blocks in forward pass more
            # than blocks_to_swap, because it should be on GPU
            if not self.forward_only and (block_idx + step) >= self.blocks_to_swap:
                break

            block_idx_to_cpu = block_idx + step
            block_idx_to_cuda = self.num_blocks - self.blocks_to_swap + block_idx_to_cpu
            # this works for forward-only offloading. move upstream blocks to cuda
            block_idx_to_cuda = block_idx_to_cuda % self.num_blocks
            self._submit_move_blocks(
                blocks,
                block_idx_to_cpu,
                block_idx_to_cuda,
                phase="forward_prefetch",
            )


def to_device(x: Any, device: torch.device) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, list):
        return [to_device(elem, device) for elem in x]
    elif isinstance(x, tuple):
        return tuple(to_device(elem, device) for elem in x)
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        return x


def to_cpu(x: Any) -> Any:
    """
    Recursively moves torch.Tensor objects (and containers thereof) to CPU.

    Args:
        x: A torch.Tensor, or a (possibly nested) list, tuple, or dict containing tensors.

    Returns:
        The same structure as x, with all torch.Tensor objects moved to CPU.
        Non-tensor objects are returned unchanged.
    """
    if isinstance(x, torch.Tensor):
        return x.cpu()
    elif isinstance(x, list):
        return [to_cpu(elem) for elem in x]
    elif isinstance(x, tuple):
        return tuple(to_cpu(elem) for elem in x)
    elif isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    else:
        return x


def create_cpu_offloading_wrapper(func: Callable, device: torch.device) -> Callable:
    """
    Create a wrapper function that offloads inputs to CPU before calling the original function
    and moves outputs back to the specified device.

    Args:
        func: The original function to wrap.
        device: The device to move outputs back to.

    Returns:
        A wrapped function that offloads inputs to CPU and moves outputs back to the specified device.
    """

    def wrapper(orig_func: Callable) -> Callable:
        def custom_forward(*inputs):
            nonlocal device, orig_func
            cuda_inputs = to_device(inputs, device)
            outputs = orig_func(*cuda_inputs)
            return to_cpu(outputs)

        return custom_forward

    return wrapper(func)
