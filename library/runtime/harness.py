"""Shared DiT + adapter run harness.

The *model-side* boilerplate every DiT-loading tool rewrites: load the DiT,
attach an optional adapter, and apply ``torch.compile`` — all in the one
ordering the pipeline actually requires. Promoted out of ``bench/`` (where it
was ``bench/_anima.py::build_anima``) so ``bench`` / ``scripts`` / ``preprocess``
and low-level probes share a single harness instead of copying it.

The compile-after-apply ordering is the load-bearing invariant:
``torch.compile`` traces the adapter's monkey-patched forward, so
``compile_blocks`` MUST run after ``network.apply_to`` + ``load_weights``.
Open-coding this means newcomers either skip ``--compile`` entirely or call it
in the wrong order; ``build_anima`` exists to remove the choice.

Usage::

    from library.runtime.harness import build_anima

    bundle = build_anima(args, dit_path=..., adapter=..., train_mode=False)
    anima, network = bundle.anima, bundle.network

``build_anima`` reads its knobs off an argparse ``Namespace`` (``device`` /
``dtype`` / ``attn_mode`` / ``gradient_checkpointing`` /
``cpu_offload_checkpointing`` / ``compile`` / bench-only ``compile_mode``);
the matching
parser surface lives in ``library.runtime.cli.add_device_args`` +
``bench._anima.add_common_args``. Callers without a parser can pass a plain
``argparse.Namespace(**kwargs)``.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Optional

import torch

log = logging.getLogger("library.runtime.harness")


@dataclass
class AnimaBundle:
    """Container for a built Anima model + optional adapter.

    Returned by ``build_anima``. ``network`` is ``None`` when no adapter
    was requested. ``device`` / ``dtype`` are the resolved torch values
    (callers that need them on the device side don't have to re-resolve).
    """

    anima: object  # library.anima Anima — typed as object to avoid heavy import
    network: Optional[object]  # networks.lora_anima.network.LoRANetwork or None
    device: torch.device
    dtype: torch.dtype


def build_anima(
    args: argparse.Namespace,
    *,
    dit_path: str | None = None,
    adapter: str | None = None,
    train_mode: bool = False,
    network_requires_grad: bool = True,
    multiplier: float = 1.0,
) -> AnimaBundle:
    """Load the DiT (+ optional adapter) with the correct ordering.

    Sequence — read the inline comments before changing:
        1. ``load_anima_model`` → DiT on device, weights cast to ``dtype``.
        2. ``anima.requires_grad_(False)`` + ``reset_mod_guidance``.
        3. If ``adapter``: ``create_network_from_weights`` → ``apply_to``
           → ``load_weights`` → ``network.to(device, dtype)``.
        4. If ``--gradient_checkpointing``: enable it (gated by
           ``anima.training`` — so train mode below must come after).
        5. ``anima.train()`` / ``anima.eval()`` per ``train_mode``. Network
           same mode (the training-time LoRA forward path is gated on
           ``network.training`` — T-LoRA mask, fp32 bottleneck, etc.).
        6. **``compile_blocks`` last** — the adapter monkey-patches must
           already be installed or torch.compile traces the wrong forward.

    Arguments:
        args: argparse namespace populated by ``add_device_args`` /
            ``bench._anima.add_common_args``. Reads ``device``, ``dtype``,
            ``attn_mode``, ``gradient_checkpointing``,
            ``cpu_offload_checkpointing``, ``compile``, bench-only
            ``compile_mode``.
        dit_path: Path to the base DiT safetensors. Defaults to
            ``args.dit`` if the caller's argparse exposed one.
        adapter: Optional adapter safetensors path. When set, the adapter
            is loaded and applied with ``multiplier`` as the apply-time
            scale.
        train_mode: If True, both anima + network are put in train mode.
            Required for any caller that calls ``backward()`` — the LoRA
            training-path forward, T-LoRA mask, and gradient checkpointing
            are all gated on ``self.training``.
        network_requires_grad: When ``train_mode=True`` and ``adapter`` is
            set, controls whether adapter params have ``requires_grad=True``
            (default) or are frozen.
        multiplier: Adapter forward-time multiplier. ``set_multiplier(0.0)``
            can flip it later to recover the base-model output.

    Returns:
        ``AnimaBundle(anima, network, device, dtype)``.
    """
    # Late imports — this module should import cheaply even on CPU-only smoke
    # runs that never load a DiT.
    from library.anima import weights as anima_utils
    from library.runtime.device import str_to_dtype

    device = torch.device(getattr(args, "device", "cuda"))
    dtype = str_to_dtype(getattr(args, "dtype", "bf16"))
    attn_mode = getattr(args, "attn_mode", "flash")

    if dit_path is None:
        dit_path = getattr(args, "dit", None)
    if dit_path is None:
        raise SystemExit(
            "build_anima: no DiT path. Pass dit_path= explicitly or expose "
            "--dit in your argparse."
        )

    log.info(f"loading base DiT: {dit_path}")
    anima = anima_utils.load_anima_model(
        device=device,
        dit_path=dit_path,
        attn_mode=attn_mode,
        loading_device=device,
        dit_weight_dtype=dtype,
    )
    anima.to(device, dtype=dtype).requires_grad_(False)
    anima.reset_mod_guidance()

    network = None
    if adapter is not None:
        log.info(f"loading adapter:  {adapter}")
        # Late import — adapter machinery has its own load-time cost.
        from networks.lora_anima.factory import create_network_from_weights

        network, _sd = create_network_from_weights(
            multiplier,
            adapter,
            None,  # ae (unused for harness callers)
            None,  # text_encoders (unused for harness callers)
            anima,
            for_inference=not train_mode,
        )
        network.apply_to([], anima, apply_text_encoder=False, apply_unet=True)
        info = network.load_weights(adapter)
        log.info(f"adapter loaded — {info}")

        network.to(device=device, dtype=dtype)
        if train_mode and network_requires_grad:
            network.requires_grad_(True)
        else:
            network.requires_grad_(False)
        anima.requires_grad_(False)  # always — DiT stays frozen in the harness

        trainable = [p for p in network.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in trainable)
        if train_mode and network_requires_grad:
            if n_train == 0:
                raise SystemExit(
                    "build_anima: adapter loaded with train_mode=True but "
                    "no trainable parameters were detected. Check the "
                    "checkpoint."
                )
            log.info(
                f"adapter trainable params: {n_train:,} ({len(trainable)} tensors)"
            )

    # Grad checkpointing is gated on anima.training (see models.py); set the
    # flag here but its effect requires train_mode below.
    if getattr(args, "gradient_checkpointing", False):
        cpu_off = getattr(args, "cpu_offload_checkpointing", False)
        suffix = " (cpu offload)" if cpu_off else ""
        log.info(f"enabling gradient checkpointing{suffix}")
        anima.enable_gradient_checkpointing(cpu_offload=cpu_off)

    if train_mode:
        anima.train()
        if network is not None:
            network.train()
    else:
        anima.eval()
        if network is not None:
            network.eval()

    # COMPILE LAST. Adapter monkey-patches must be installed first or
    # torch.compile traces the wrong forward.
    if getattr(args, "compile", False):
        # Historical bench flag name: this is torch.compile's inductor mode,
        # not the removed training-time full/block compile selector.
        mode = getattr(args, "compile_mode", None)
        log.info(
            f"compiling DiT blocks{' (mode=' + mode + ')' if mode else ''} "
            "— first batch pays ~30-60s compile cost"
        )
        anima.compile_blocks(mode=mode)

    return AnimaBundle(anima=anima, network=network, device=device, dtype=dtype)
