"""Small train.py bootstrap helpers (samples, compile resolutions, cleanup)."""

from __future__ import annotations

import logging
import os
import signal
from typing import Optional

from accelerate import Accelerator

from library import train_util
from library.anima import training as anima_train_utils
from library.runtime.device import clean_memory_on_device

logger = logging.getLogger(__name__)


def collect_bucket_resolutions(*dataset_groups) -> list[tuple[int, int]]:
    resos: set[tuple[int, int]] = set()
    for group in dataset_groups:
        if group is None:
            continue
        datasets = getattr(group, "datasets", None) or []
        for dataset in datasets:
            bucket_manager = getattr(dataset, "bucket_manager", None)
            active_resos = getattr(bucket_manager, "resos", None)
            if active_resos:
                resos.update((int(w), int(h)) for w, h in active_resos)
                continue
            image_data = getattr(dataset, "image_data", {}) or {}
            for info in image_data.values():
                reso = getattr(info, "bucket_reso", None)
                if reso is not None:
                    resos.add((int(reso[0]), int(reso[1])))
    return sorted(resos)


def collect_compile_resolutions(
    *dataset_groups,
    sample_prompts: str | None = None,
) -> list[tuple[int, int]]:
    resos = set(collect_bucket_resolutions(*dataset_groups))
    resos.update(train_util.sample_prompt_resolutions(sample_prompts))
    return sorted(resos)


def install_stop_signal_handlers() -> None:
    """Make SIGTERM follow the same cleanup path as Ctrl-C."""

    if not hasattr(signal, "SIGTERM"):
        return

    def _raise_keyboard_interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)


def resolve_block_swap_profile_jsonl(args) -> Optional[str]:
    explicit = getattr(args, "block_swap_profile_jsonl", None)
    if explicit is None:
        return None
    value = str(explicit).strip()
    if value.lower() in {"", "off", "none", "false", "0"}:
        return None
    if value.lower() != "auto":
        return value

    output_dir = getattr(args, "output_dir", None)
    if not output_dir:
        return None
    output_name = getattr(args, "output_name", None) or "run"
    parent = os.path.dirname(os.path.normpath(output_dir))
    logs_dir = os.path.join(parent or output_dir, "logs")
    return os.path.join(logs_dir, f"{output_name}.block_swap_profile.jsonl")


def decode_deferred_samples_safely(
    accelerator: Accelerator,
    args,
    loop_state,
    vae,
    *,
    optimizer_eval_fn=None,
) -> None:
    if not accelerator.is_main_process or not sample_preview_enabled(args):
        return
    try:
        if optimizer_eval_fn is not None:
            try:
                optimizer_eval_fn()
            except Exception as exc:  # noqa: BLE001 - cleanup must keep going
                logger.warning(
                    "Could not switch optimizer to eval before sample decode: "
                    f"{exc}"
                )
        try:
            accelerator.unwrap_model(loop_state.unet).to("cpu")
        except Exception:
            pass
        clean_memory_on_device(accelerator.device)
        anima_train_utils.decode_pending_samples(accelerator, args, vae)
    except Exception as exc:  # noqa: BLE001 - never mask the real train exit
        logger.error(f"Failed to decode deferred sample images during cleanup: {exc}")


def normalize_sample_args(args):
    """Normalize inline sample prompts and disabled sample cadence."""
    for knob in ("sample_every_n_epochs", "sample_every_n_steps"):
        value = getattr(args, knob, None)
        if value is not None and value <= 0:
            setattr(args, knob, None)

    value = getattr(args, "sample_prompts", None)
    if value is None:
        return

    if isinstance(value, (list, tuple)):
        lines = [str(item).strip() for item in value]
    elif isinstance(value, str):
        if os.path.isfile(value):
            return
        lines = [line.strip() for line in value.splitlines()]
    else:
        return

    lines = [line for line in lines if line and not line.startswith("#")]
    if not lines:
        args.sample_prompts = None
        return

    os.makedirs(args.output_dir, exist_ok=True)
    prompt_path = os.path.join(args.output_dir, "sample_prompts.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Wrote {len(lines)} inline sample prompt(s) to {prompt_path}")
    args.sample_prompts = prompt_path


def sample_preview_enabled(args) -> bool:
    return bool(
        getattr(args, "sample_prompts", None)
        and (
            getattr(args, "sample_at_first", False)
            or getattr(args, "sample_every_n_steps", None)
            or getattr(args, "sample_every_n_epochs", None)
        )
    )
