"""Calibrate before the parent loads models; never retry a live training step."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import secrets
import tempfile

from .config import configuration_errors, probe_arguments
from .policy import Measurement, SwapSearch
from .process import run_process, write_result
from .preferences import preference, swap_io_limit_bytes

logger = logging.getLogger(__name__)


def measurement(result: dict, blocks: int, *, swap_limit_bytes=None) -> Measurement:
    status = result.get("status")
    if status not in ("ok", "cuda_oom"):
        raise RuntimeError(
            f"AUTO block-swap probe failed: {status}: {result.get('error', '')}"
        )
    if result["host_min_available"] <= result["host_reserve"]:
        # Physical host pressure is not a monotonic GPU OOM boundary; do not bisect it.
        raise RuntimeError(
            "AUTO block-swap host RAM/paging reserve exhausted during the probe"
        )
    swap_limit = (
        result.get("swap_io_limit_bytes", 64 * 1024**2)
        if swap_limit_bytes is None
        else swap_limit_bytes
    )
    if swap_limit and result["swap_io_bytes"] > swap_limit:
        raise RuntimeError(
            "AUTO block-swap RAM/paging or configured system swap IO/paging limit exceeded"
        )
    if status == "cuda_oom":
        return Measurement(blocks, False)
    return Measurement(
        blocks, result.get("safe") is True, result["seconds"], result["headroom"]
    )


def _validate_launch(args) -> None:
    import torch

    errors = configuration_errors(
        args, world_size=int(os.environ.get("WORLD_SIZE", "1"))
    )
    if errors:
        raise ValueError("; ".join(errors))
    if not torch.cuda.is_available():
        raise ValueError("AUTO block swap requires CUDA")
    from library.training.checkpoints import _checkpoint_state_candidates

    if getattr(args, "checkpointing_epochs", None) and _checkpoint_state_candidates(
        args
    ):
        raise ValueError(
            "AUTO v1 cannot calibrate a run with an existing auto-resume checkpoint"
        )
    if getattr(args, "base_compute", "bf16") == "nf4":
        from library.models.krea2_raw.quantize import inspect_nf4_checkpoint

        path = (
            getattr(args, "nf4_prequantized_path", None)
            or args.pretrained_model_name_or_path
        )
        if not inspect_nf4_checkpoint(path).is_nf4:
            raise ValueError(
                "AUTO v1 requires a prequantized NF4 checkpoint; online quantization is not probed"
            )


def run_calibration(args, directory: Path, *, runner=run_process) -> int:
    """The runner boundary is injectable for deterministic fault/ablation tests."""
    timeout = int(getattr(args, "auto_block_swap_timeout", 1800))
    swap_limit = swap_io_limit_bytes(args)
    inventory_dir = directory / "inventory"
    inventory = runner(
        probe_arguments(args, inventory_dir, blocks=1, inventory=True),
        inventory_dir,
        timeout=timeout,
    )
    write_result(directory / "inventory.json", inventory)
    if inventory.get("status") != "inventory":
        raise RuntimeError(
            f"AUTO block-swap inventory failed: {inventory.get('status')}"
        )
    maximum = min(inventory["model_swap_limit"], inventory["host_swap_limit"])
    search = SwapSearch(
        maximum, max_trials=int(getattr(args, "auto_block_swap_max_trials", 6)),
        preference=preference(args),
    )
    results = []
    report = {"status": "calibrating", "inventory": inventory, "trials": results}
    write_result(directory / "summary.json", report)
    while (blocks := search.next_candidate()) is not None:
        logger.info(
            "AUTO block swap: probing %s/%s blocks", blocks, inventory["block_count"]
        )
        trial_dir = directory / f"trial-{len(results):02d}-swap-{blocks}"
        result = runner(
            probe_arguments(args, trial_dir, blocks=blocks), trial_dir, timeout=timeout
        )
        results.append({"blocks": blocks, **result})
        write_result(directory / "summary.json", report)
        search.observe(measurement(result, blocks, swap_limit_bytes=swap_limit))
    selected = search.select()
    # A fresh confirmation prevents choosing a one-off lucky allocator state.
    confirmation_dir = directory / f"confirm-swap-{selected}"
    confirmation = runner(
        probe_arguments(args, confirmation_dir, blocks=selected),
        confirmation_dir,
        timeout=timeout,
    )
    report["confirmation"] = confirmation
    write_result(directory / "summary.json", report)
    if not measurement(confirmation, selected, swap_limit_bytes=swap_limit).safe:
        raise RuntimeError(
            "AUTO block-swap confirmation failed; refusing to start training"
        )
    report.update(
        {"status": "selected", "selected_blocks": selected, "seed": args.seed}
    )
    write_result(directory / "summary.json", report)
    return selected


def calibrate_if_requested(args) -> None:
    if not getattr(args, "auto_block_swap", False) or getattr(
        args, "_auto_swap_resolved", False
    ):
        return
    _validate_launch(args)
    if args.seed is None:
        args.seed = secrets.randbits(32)
    root = Path(args.output_dir) / "auto-block-swap"
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="calibration-", dir=root))
    logger.info("AUTO block-swap calibration report: %s", directory / "summary.json")
    try:
        if getattr(args, "auto_block_swap_mode", "startup") == "dynamic":
            from .online_startup import initialize

            initialize(args, directory)
            return
        selected = run_calibration(args, directory)
    except BaseException as exc:
        write_result(
            directory / "failure.json",
            {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)},
        )
        raise
    args.blocks_to_swap = selected
    args._auto_swap_resolved = True
    args._auto_swap_report = str(directory / "summary.json")
    logger.info(
        "AUTO block swap selected %s blocks; fixed for this training run", selected
    )
