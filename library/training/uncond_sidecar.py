"""Lazy staging/loading of the T5('') uncond crossattn sidecar."""

from __future__ import annotations

import argparse
import logging

import torch

logger = logging.getLogger(__name__)


def ensure_uncond_crossattn(
    trainer,
    args: argparse.Namespace,
    accelerator,
    weight_dtype: torch.dtype,
) -> None:
    """Lazily load the T5("") crossattn sidecar onto ``trainer._state.uncond_crossattn_1``.

    Primary producer is ``make preprocess-te`` (drops the file at
    ``post_image_dataset/_anima_uncond_te.safetensors``); this method is
    the fallback that stages on demand if a training run was kicked off
    without the preprocess step.
    """
    if trainer._state.uncond_crossattn_1 is not None:
        return
    from library.inference.uncond import (
        DEFAULT_UNCOND_DIR,
        default_uncond_path,
        load_uncond_crossattn,
        stage_uncond_sidecar,
    )

    sidecar = default_uncond_path()
    if not sidecar.exists():
        logger.info(
            f"T5('') uncond sidecar missing at {sidecar} — staging "
            f"on demand (would normally be produced by `make preprocess-te`)."
        )
        stage_uncond_sidecar(
            DEFAULT_UNCOND_DIR,
            qwen3_path=args.qwen3,
            dit_path=args.pretrained_model_name_or_path,
            t5_tokenizer_path=getattr(args, "t5_tokenizer_path", None),
            seq_len=512,
            overwrite=False,
        )
    trainer._state.uncond_crossattn_1 = load_uncond_crossattn(
        str(sidecar), device=accelerator.device, dtype=weight_dtype
    )
    logger.info(
        f"T5('') uncond loaded: {sidecar} "
        f"shape={tuple(trainer._state.uncond_crossattn_1.shape)}"
    )
