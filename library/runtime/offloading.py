from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
import threading
import time
from typing import Any, Optional, Union, Callable, Tuple
import torch
import torch.nn as nn

from library.runtime.device import (
    clean_memory_on_device,
    should_move_weight_to_device,
    synchronize_device,
    weighs_to_device,
)

logger = logging.getLogger(__name__)


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
    if _weight_device_type(module) != device.type:
        weight.data = weight.data.to(device, non_blocking=True)


def _can_swap_frozen_weight_to_cpu(module: nn.Module) -> bool:
    return should_move_weight_to_device(
        module, torch.device("cpu"), include_trainable=False
    )


def _capture_cpu_master(weight: torch.Tensor, *, pin_memory: bool) -> torch.Tensor:
    master = weight.detach()
    if master.device.type != "cpu":
        master = master.to("cpu", non_blocking=False)
    if pin_memory:
        try:
            master = master.pin_memory()
        except Exception:
            pass
    return master


def _finalize_async_cuda_timings(timings: dict[str, Any]) -> None:
    end_event = timings.pop("_h2d_end_event", None)
    start_event = timings.pop("_h2d_start_event", None)
    ready_event = timings.pop("_ready_event", None)
    if end_event is None:
        return
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


class BlockSwapProfiler:
    """Append-only JSONL writer for block-swap transfer/wait observations."""

    def __init__(self, path: str):
        self.path = str(path)
        self._lock = threading.Lock()
        self._seq = 0
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)

    def write(self, event: dict[str, Any]) -> None:
        try:
            with self._lock:
                self._seq += 1
                payload = {"seq": self._seq, **event}
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            return


def _resolve_profiler(profile_jsonl: Optional[Union[str, BlockSwapProfiler]]):
    if isinstance(profile_jsonl, BlockSwapProfiler):
        return profile_jsonl
    if profile_jsonl is None:
        return None
    path = str(profile_jsonl).strip()
    if not path or path.lower() in {"off", "none", "false", "0"}:
        return None
    return BlockSwapProfiler(path)


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
    ):
        self.num_blocks = num_blocks
        self.blocks_to_swap = blocks_to_swap
        self.device = device
        self.debug = debug

        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.futures = {}
        self.cuda_available = device.type == "cuda"
        self.profiler = _resolve_profiler(profile_jsonl)
        self.profile_step = 0
        self._cpu_weight_masters: Optional[list[dict[str, torch.Tensor]]] = None
        self._frozen_weight_bytes_by_block: list[int] = [0 for _ in range(num_blocks)]
        self._frozen_weight_master_bytes = 0
        self._reported_weight_masters = False

    def _ensure_cpu_weight_masters(
        self, blocks: Union[list[nn.Module], nn.ModuleList]
    ) -> None:
        if self._cpu_weight_masters is not None:
            return

        pin_memory = self.cuda_available
        masters: list[dict[str, torch.Tensor]] = []
        bytes_by_block: list[int] = []
        total_bytes = 0
        for block in blocks:
            block_masters: dict[str, torch.Tensor] = {}
            block_bytes = 0
            for name, module in block.named_modules():
                weight = getattr(module, "weight", None)
                if weight is None or not _can_swap_frozen_weight_to_cpu(module):
                    continue
                master = _capture_cpu_master(weight.data, pin_memory=pin_memory)
                block_masters[name] = master
                block_bytes += _tensor_nbytes(master)
            masters.append(block_masters)
            bytes_by_block.append(block_bytes)
            total_bytes += block_bytes

        self._cpu_weight_masters = masters
        self._frozen_weight_bytes_by_block = bytes_by_block
        self._frozen_weight_master_bytes = total_bytes
        if not self._reported_weight_masters:
            logger.info(
                "Block swap frozen CPU masters prepared: "
                f"{total_bytes / (1024 ** 3):.2f} GiB across {len(blocks)} blocks"
            )
            self._reported_weight_masters = True
        if self.profiler is not None:
            self.profiler.write(
                {
                    "ev": "block_swap_config",
                    "step": self.profile_step,
                    "num_blocks": self.num_blocks,
                    "blocks_to_swap": self.blocks_to_swap,
                    "frozen_weight_master_bytes": total_bytes,
                    "frozen_weight_bytes_by_block": bytes_by_block,
                    "h2d_only": True,
                }
            )

    def swap_weight_devices(
        self,
        block_idx_to_cpu: int,
        block_to_cpu: nn.Module,
        block_idx_to_cuda: int,
        block_to_cuda: nn.Module,
        *,
        ready_event: Optional[Any] = None,
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
        if self.cuda_available:
            return self._swap_weight_devices_cached_cuda(
                block_to_cpu,
                block_to_cuda,
                source_masters,
                target_masters,
                ready_event=ready_event,
            )
        return self._swap_weight_devices_cached_no_cuda(
            block_to_cpu,
            block_to_cuda,
            source_masters,
            target_masters,
        )

    def _swap_weight_devices_cached_cuda(
        self,
        layer_to_cpu: nn.Module,
        layer_to_cuda: nn.Module,
        source_masters: dict[str, torch.Tensor],
        target_masters: dict[str, torch.Tensor],
        *,
        ready_event: Optional[Any] = None,
    ) -> dict[str, float]:
        assert layer_to_cpu.__class__ == layer_to_cuda.__class__

        weight_swap_jobs: list[
            Tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        modules_to_cpu = {k: v for k, v in layer_to_cpu.named_modules()}
        for module_to_cuda_name, module_to_cuda in layer_to_cuda.named_modules():
            weight = getattr(module_to_cuda, "weight", None)
            if weight is None:
                continue
            module_to_cpu = modules_to_cpu.get(module_to_cuda_name, None)
            source_master = source_masters.get(module_to_cuda_name)
            target_master = target_masters.get(module_to_cuda_name)
            if (
                module_to_cpu is not None
                and source_master is not None
                and target_master is not None
                and module_to_cpu.weight.shape == module_to_cuda.weight.shape
            ):
                weight_swap_jobs.append(
                    (
                        module_to_cpu,
                        module_to_cuda,
                        module_to_cpu.weight.data,
                        source_master,
                        target_master,
                    )
                )
            else:
                _ensure_weight_on_device(module_to_cuda, self.device)
                if module_to_cpu is not None:
                    _ensure_weight_on_device(module_to_cpu, self.device)

        timings: dict[str, Any] = {
            "d2h_ms": 0.0,
            "h2d_ms": 0.0,
            "event_wait_ms": 0.0,
        }
        if not weight_swap_jobs:
            return timings

        stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(stream):
            if ready_event is not None:
                stream.wait_event(ready_event)

            # Frozen base weights never change during LoRA training. We keep a
            # CPU master for every swappable weight and only restore the next
            # block into the retired block's GPU storage. This removes the
            # per-swap D2H copy that made next-use prefetch miss its window.
            h2d_start = torch.cuda.Event(enable_timing=True)
            h2d_end = torch.cuda.Event(enable_timing=True)
            h2d_start.record(stream)
            for (
                module_to_cpu,
                module_to_cuda,
                cuda_data_view,
                source_master,
                target_master,
            ) in weight_swap_jobs:
                module_to_cpu.weight.data = source_master
                cuda_data_view.record_stream(stream)
                cuda_data_view.copy_(target_master, non_blocking=True)
                module_to_cuda.weight.data = cuda_data_view
            h2d_end.record(stream)

        timings["_ready_event"] = ready_event
        timings["_h2d_start_event"] = h2d_start
        timings["_h2d_end_event"] = h2d_end
        return timings

    def _swap_weight_devices_cached_no_cuda(
        self,
        layer_to_cpu: nn.Module,
        layer_to_cuda: nn.Module,
        source_masters: dict[str, torch.Tensor],
        target_masters: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        assert layer_to_cpu.__class__ == layer_to_cuda.__class__

        t0 = time.perf_counter()
        modules_to_cpu = {k: v for k, v in layer_to_cpu.named_modules()}
        for module_to_cuda_name, module_to_cuda in layer_to_cuda.named_modules():
            weight = getattr(module_to_cuda, "weight", None)
            if weight is None:
                continue
            module_to_cpu = modules_to_cpu.get(module_to_cuda_name, None)
            source_master = source_masters.get(module_to_cuda_name)
            target_master = target_masters.get(module_to_cuda_name)
            if (
                module_to_cpu is not None
                and source_master is not None
                and target_master is not None
                and module_to_cpu.weight.shape == module_to_cuda.weight.shape
            ):
                module_to_cpu.weight.data = source_master
                module_to_cuda.weight.data = target_master.to(
                    self.device, non_blocking=True
                )
            else:
                _ensure_weight_on_device(module_to_cuda, self.device)
                if module_to_cpu is not None:
                    _ensure_weight_on_device(module_to_cpu, self.device)
        synchronize_device(self.device)
        return {"d2h_ms": 0.0, "h2d_ms": (time.perf_counter() - t0) * 1000.0}

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
        if self.cuda_available:
            ready_event = torch.cuda.Event(enable_timing=self.profiler is not None)
            ready_event.record(torch.cuda.current_stream(self.device))

        def move_blocks(
            bidx_to_cpu,
            block_to_cpu,
            bidx_to_cuda,
            block_to_cuda,
            event,
            submitted_at,
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

        self.futures[block_idx_to_cuda] = (
            self.thread_pool.submit(
                move_blocks,
                block_idx_to_cpu,
                block_to_cpu,
                block_idx_to_cuda,
                block_to_cuda,
                ready_event,
                queued_at,
            ),
            {
                "phase": phase,
                "block_idx": block_idx_to_cuda,
                "block_idx_to_cpu": block_idx_to_cpu,
                "queued_at": queued_at,
                "step": self.profile_step,
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
        _, bidx_to_cuda, timings, enqueued_at = future.result()
        _finalize_async_cuda_timings(timings)
        ready_at = time.time()
        wait_ms = (time.perf_counter() - wait_t0) * 1000.0
        timings["transfer_ms"] = float(timings.get("event_wait_ms", 0.0)) + float(
            timings.get("h2d_ms", 0.0)
        )
        queued_at = meta.get("queued_at")
        if queued_at is not None:
            timings["submit_lag_ms"] = max(0.0, (ready_at - queued_at) * 1000.0)

        assert block_idx == bidx_to_cuda, (
            f"Block index mismatch: {block_idx} != {bidx_to_cuda}"
        )
        if self.profiler is not None:
            self.profiler.write(
                {
                    "ev": "block_swap",
                    "phase": phase or meta.get("phase") or "",
                    "submit_phase": meta.get("phase") or "",
                    "step": meta.get("step"),
                    "block_idx": block_idx,
                    "block_idx_to_cpu": meta.get("block_idx_to_cpu"),
                    "wait_ms": wait_ms,
                    "h2d_ms": float(timings.get("h2d_ms", 0.0)),
                    "d2h_ms": float(timings.get("d2h_ms", 0.0)),
                    "event_wait_ms": float(timings.get("event_wait_ms", 0.0)),
                    "transfer_ms": float(timings.get("transfer_ms", 0.0)),
                    "enqueue_ms": float(timings.get("enqueue_ms", 0.0)),
                    "submit_lag_ms": float(timings.get("submit_lag_ms", 0.0)),
                    "queued_at": meta.get("queued_at"),
                    "enqueued_at": enqueued_at,
                    "ready_at": ready_at,
                    "waited_at": time.time(),
                }
            )

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
    ):
        super().__init__(len(blocks), blocks_to_swap, device, debug, profile_jsonl)

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
        self.forward_only = forward_only

    def __del__(self):
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
        self.profile_step += 1
        self._ensure_cpu_weight_masters(blocks)

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

        # if backward is enabled, we do not swap blocks in forward pass more than blocks_to_swap, because it should be on GPU
        if not self.forward_only and block_idx >= self.blocks_to_swap:
            return

        block_idx_to_cpu = block_idx
        block_idx_to_cuda = self.num_blocks - self.blocks_to_swap + block_idx
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
