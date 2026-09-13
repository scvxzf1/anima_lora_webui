"""CUDA telemetry and retry boundaries for side-effect-free cache encoders."""

from __future__ import annotations

import gc
import logging
import random
import time
from dataclasses import asdict

import torch

from library.preprocess.batch_policy import AutoBatchPolicy, BatchMeasurement

logger = logging.getLogger(__name__)


def batch_size_arg(value) -> int | str:
    if str(value).strip().lower() == "auto":
        return "auto"
    try:
        parsed = int(str(value).strip())
    except (ValueError, TypeError):
        raise ValueError("batch size must be 'auto' or a positive integer") from None
    if parsed < 1:
        raise ValueError("batch size must be 'auto' or a positive integer")
    return parsed


def iter_batches(items, size):
    """Evaluate the size callback only after the consumer finishes each batch."""
    offset = 0
    while offset < len(items):
        count = max(1, int(size()))
        yield items[offset : offset + count]
        offset += count


def concatenate_outputs(chunks):
    if not chunks:
        raise ValueError("cannot infer output shape from an empty encoding request")
    if isinstance(chunks[0], torch.Tensor):
        return torch.cat(chunks, dim=0)
    return tuple(
        None if values[0] is None else torch.cat(values, dim=0)
        for values in zip(*chunks, strict=True)
    )


class AutoBatcher:
    """Measure compute + CPU output transfer, never include cache writes in retry.

    ``encode`` must return CPU tensors, with no disk writes or random caption
    generation. Exceptions must leave its stack before allocator cleanup.
    Non-CUDA devices deliberately stay at batch 1; integer mode never uses this.
    """

    def __init__(
        self,
        device,
        *,
        label: str,
        max_batch=32,
        on_event=None,
        cleanup=None,
        predict=True,
        throughput=True,
        retry_oom=True,
    ):
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        self.label = label
        self.policy = AutoBatchPolicy(
            max_batch if self.cuda else 1, predict=predict, throughput=throughput
        )
        self.on_event = on_event
        self.cleanup = cleanup
        self.retry_oom = retry_oom
        self._logged_batch_size = 1

    @property
    def batch_size(self):
        return self.policy.current

    def _budget(self):
        if not self.cuda:
            return 0
        free, total = torch.cuda.mem_get_info(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        get_fraction = getattr(torch.cuda, "get_per_process_memory_fraction", None)
        index = self.device.index
        if index is None:
            index = torch.cuda.current_device()
        fraction = get_fraction(index) if get_fraction else 1.0
        available = min(free + reserved, int(total * fraction))
        return max(0, available - max(512 * 1024**2, int(total * 0.10)))

    def _emit(self, status, batch, **details):
        event = dict(
            label=self.label,
            status=status,
            batch=batch,
            next_batch=self.batch_size,
            **details,
        )
        # Rich log rendering consumes the global Python RNG. Telemetry must not
        # change subsequent caption shuffles when the batch schedule changes.
        state = random.getstate()
        try:
            if self.on_event:
                self.on_event(event)
            if status == "oom" or self._logged_batch_size != self.batch_size:
                logger.info(
                    "auto batch [%s]: %s batch=%d -> %d",
                    self.label,
                    status,
                    batch if status == "oom" else self._logged_batch_size,
                    self.batch_size,
                )
            self._logged_batch_size = self.batch_size
        finally:
            random.setstate(state)

    def iter_encoded(self, items, encode):
        offset = 0
        while offset < len(items):
            budget = self._budget()
            batch = min(self.policy.prepare(budget), len(items) - offset)
            if self.cuda:
                torch.cuda.synchronize(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
            base = torch.cuda.memory_allocated(self.device) if self.cuda else 0
            start = time.perf_counter()
            failed = False
            try:
                outputs = encode(items[offset : offset + batch])
                if self.cuda:
                    torch.cuda.synchronize(self.device)
            except torch.cuda.OutOfMemoryError as exc:
                if not self.cuda or batch == 1 or not self.retry_oom:
                    exc.add_note(
                        f"auto batch [{self.label}]: batch={batch} cannot fit; "
                        "free device memory or change model placement/precision"
                    )
                    raise
                self.policy.failure(batch)
                failed = True
            if failed:
                try:
                    if self.cleanup:
                        self.cleanup()
                finally:
                    gc.collect()
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()
                self._emit("oom", batch, seconds=time.perf_counter() - start)
                continue
            peak = torch.cuda.max_memory_allocated(self.device) if self.cuda else 0
            measured = BatchMeasurement(
                batch, time.perf_counter() - start, base, peak, budget
            )
            self.policy.success(measured)
            self._emit(
                "ok",
                batch,
                **{k: v for k, v in asdict(measured).items() if k != "batch"},
            )
            yield offset, outputs
            del outputs
            offset += batch

    def encode_all(self, items, encode):
        return concatenate_outputs(
            [value for _, value in self.iter_encoded(items, encode)]
        )
