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
``cpu_offload_checkpointing`` / ``compile`` / ``compile_mode``); the matching
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
            ``cpu_offload_checkpointing``, ``compile``, ``compile_mode``.
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
        mode = getattr(args, "compile_mode", None)
        log.info(
            f"compiling DiT blocks{' (mode=' + mode + ')' if mode else ''} "
            "— first batch pays ~30-60s compile cost"
        )
        anima.compile_blocks(mode=mode)

    return AnimaBundle(anima=anima, network=network, device=device, dtype=dtype)


# --- Training-side build helpers -------------------------------------------
#
# ``build_anima`` above owns the *inference / existing-adapter* path: it loads a
# checkpoint with ``create_network_from_weights`` + ``load_weights``. The
# distillation trainers (``scripts/distill_{mod,spd,turbo}.py``) instead build a
# *fresh, untrained* network (or train an in-model MLP), each with its own
# ordering of freeze / optimizer / per-forward swap toggles — so they can't call
# ``build_anima`` wholesale. These three composable helpers factor out the parts
# that were copied verbatim across all three (the block-swap placement, the
# dynamo-cache-bump + ``compile_blocks``, and the grad-checkpoint toggle) without
# imposing a single ordering. Call them in whatever order your trainer needs;
# the compile-after-monkey-patch invariant still applies — run
# ``compile_dit_blocks`` only after the network's ``apply_to``.


def place_dit_for_training(
    anima: object, device: torch.device, *, blocks_to_swap: int = 0
) -> None:
    """Move a (frozen-base) DiT onto ``device`` for a training run.

    With block swap on, the swapped blocks stay on CPU and ride the
    forward+backward swap hooks while everything else moves to ``device``;
    without it the whole model moves. This arms the *training* swap path (two
    block movements per step) — distinct from the inference placement
    ``build_anima`` does. Call before ``compile_dit_blocks`` / ``train()``.
    """
    if blocks_to_swap > 0:
        anima.enable_block_swap(blocks_to_swap, device)
        anima.move_to_device_except_swap_blocks(device)
        anima.switch_block_swap_for_training()  # forward+backward block movement
    else:
        anima.to(device)


def compile_dit_blocks(
    anima: object,
    *,
    enabled: bool = True,
    cache_size_limit: int = 64,
    backend: str = "inductor",
    mode: Optional[str] = None,
    dynamic_seq: bool = False,
    n_token_families: Optional[int] = None,
    seq_range: Optional[tuple[int, int]] = None,
) -> None:
    """``torch.compile`` each ``Block._forward`` for a distillation/training run.

    ``compile_blocks`` turns on native-shape flattening (every aspect bucket
    runs at its real token count, no padding → no flash pad-leak into the
    target) and traces one block graph per distinct token count. Distillation
    pools span more than the 2 ``CONSTANT_TOKEN_BUCKETS`` families, so pre-raise
    the dynamo cache to ``cache_size_limit`` (``compile_blocks``' own ``max()``
    won't lower it) so each shape traces instead of falling back to eager
    mid-warmup. No-op when ``enabled`` is False.

    COMPILE LAST — install the adapter / network monkey-patches first, or
    torch.compile traces the wrong forward (the invariant ``build_anima``
    encodes).
    """
    if not enabled:
        return
    from library.runtime.dynamo import pin_dynamo_limit

    pin_dynamo_limit("recompile_limit", cache_size_limit)
    anima.compile_blocks(
        backend,
        mode=mode,
        n_token_families=n_token_families,
        dynamic_seq=dynamic_seq,
        seq_range=seq_range,
    )


def compile_signature(
    *,
    n_token_families: Optional[int],
    seq_range: Optional[tuple[int, int]],
    dynamic_seq: bool,
    backend: str = "inductor",
    mode: Optional[str] = None,
) -> str:
    """Stable signature for per-run torch.compile cache isolation."""

    return (
        f"families={n_token_families};seq_range={seq_range};"
        f"dynamic_seq={dynamic_seq};backend={backend};mode={mode or None}"
    )


_compile_cache_base: Optional[str] = None


def isolate_compile_cache(signature: str) -> str:
    """Route persistent torch.compile caches to a per-signature directory.

    FxGraphCache / AOTAutogradCache do not encode our mark_dynamic bounds in a
    way that safely separates runs. Reusing a stale cache compiled with narrower
    seq bounds can poison a later checkpoint recompute. Isolating by signature
    keeps same-config warm cache reuse while preventing cross-config guard drift.
    """

    global _compile_cache_base
    import hashlib
    import os

    if _compile_cache_base is None:
        base = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        if not base:
            try:
                from torch._inductor.runtime.cache_dir_utils import default_cache_dir

                base = default_cache_dir()
            except Exception:  # noqa: BLE001 - torch internals vary by version
                import getpass
                import tempfile

                base = os.path.join(
                    tempfile.gettempdir(), f"torchinductor_{getpass.getuser()}"
                )
        _compile_cache_base = base

    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:16]
    target = os.path.join(_compile_cache_base, f"anima-sig-{digest}")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = target
    log.info("torch.compile cache isolated per compile signature: %s", target)
    log.info("compile signature: %s", signature)
    return target


def _apply_activation_memory_budget(
    budget: float, *, grad_ckpt: bool, logger: logging.Logger = log
) -> None:
    """Apply AOT partitioner memory cap when it is safe.

    Under gradient checkpointing this is intentionally skipped: repartitioning
    the compiled joint graph can make checkpoint recompute select a different
    graph than the original forward, producing saved-vs-recomputed metadata
    mismatches.
    """

    if budget < 1.0 and not grad_ckpt:
        import torch._functorch.config as _functorch_config

        _functorch_config.activation_memory_budget = budget
        logger.info(
            "torch.compile activation_memory_budget = %.3g "
            "(partitioner recomputes cheap intermediates in backward)",
            budget,
        )
    elif budget < 1.0:
        logger.info(
            "activation_memory_budget ignored: incompatible with "
            "gradient_checkpointing (and redundant under it)"
        )


def compile_blocks_for_training(
    unet: object,
    network: object,
    *,
    backend: str,
    mode: Optional[str] = None,
    bucket_resolutions: Optional[list[tuple[int, int]]] = None,
    n_token_families: Optional[int] = None,
    seq_range: Optional[tuple[int, int]] = None,
    dynamic_seq: bool = False,
    activation_memory_budget: float = 1.0,
    grad_ckpt: bool = False,
    logger: logging.Logger = log,
) -> None:
    """Training compile sequence; call after adapter apply/load/grad-ckpt.

    The ordering matters: Dynamo must trace adapter monkey-patched Linear
    forwards and the same callable checkpoint will recompute in backward.
    """

    if n_token_families is None and bucket_resolutions:
        counts = {
            (int(h) // getattr(unet, "patch_spatial", 16))
            * (int(w) // getattr(unet, "patch_spatial", 16))
            for w, h in bucket_resolutions
        }
        if counts:
            n_token_families = len(counts)
            seq_range = (min(counts), max(counts)) if seq_range is None else seq_range

    _apply_activation_memory_budget(
        activation_memory_budget, grad_ckpt=grad_ckpt, logger=logger
    )
    isolate_compile_cache(
        compile_signature(
            n_token_families=n_token_families,
            seq_range=seq_range,
            dynamic_seq=dynamic_seq,
            backend=backend,
            mode=mode,
        )
    )
    unet.compile_blocks(
        backend,
        mode=mode,
        bucket_resolutions=bucket_resolutions,
        n_token_families=n_token_families,
        dynamic_seq=dynamic_seq,
        seq_range=seq_range,
    )
    if hasattr(network, "compile_cond_stream"):
        network.compile_cond_stream(
            backend,
            mode=mode,
            n_token_families=n_token_families,
            dynamic_seq=dynamic_seq,
            seq_range=seq_range,
        )


def enable_training_grad_ckpt(anima: object, *, enabled: bool) -> None:
    """Toggle unsloth CPU-offload gradient checkpointing for a training run.

    Recomputes block activations in backward, offloading saved tensors to CPU
    between forward/backward. The model must stay in ``train()`` mode —
    ``Block.forward`` gates checkpointing on ``self.training``. Logs and no-ops
    when ``enabled`` is False.
    """
    if enabled:
        anima.enable_gradient_checkpointing(unsloth_offload=True)
        log.info("gradient checkpointing: on (unsloth CPU offload)")
    else:
        log.info("gradient checkpointing: off")
