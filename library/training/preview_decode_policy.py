"""Transactional training-preview VAE decoding with CUDA-OOM-only degradation."""

import logging
import traceback

import torch

from library.training.preview_decode_budget import PreviewDecodeBudgetError, cpu_decode_budget
from library.training.preview_decode_events import emit_decode_event

logger = logging.getLogger(__name__)


class PreviewDecodeRestoreError(RuntimeError):
    """VAE state could not be restored; callers must stop rather than retry."""


def _is_cuda_oom(error):
    # torch.OutOfMemoryError is the same class as torch.cuda.OutOfMemoryError
    # on current PyTorch. The type alone cannot identify the failed allocator.
    message = str(error).lower()
    host_markers = ("cpu", "host", "defaultcpuallocator", "can't allocate memory",
                    "illegal memory access", "timeout", "nan")
    return ("cuda" in message and "out of memory" in message
            and not any(marker in message for marker in host_markers))


def _clear_failed_decode(vae, error):
    traceback.clear_frames(error.__traceback__)
    error.__traceback__ = None
    clear = getattr(vae, "clear_cache", None)
    if clear:
        clear()
    torch.cuda.empty_cache()


def _tiling(vae, enabled):
    method = getattr(vae, "enable_tiling" if enabled else "disable_tiling", None)
    if method:
        method()
    elif hasattr(vae, "use_tiling"):
        vae.use_tiling = enabled
    elif enabled:
        raise RuntimeError("Training preview tiled decode requires VAE tiling support")
    # Flux AutoencoderKL otherwise keeps its 1024px config threshold, which
    # makes a 1024px preview silently take the non-tiled branch.
    sample = getattr(vae, "tile_sample_min_size", None)
    latent = getattr(vae, "tile_latent_min_size", None)
    if enabled and isinstance(sample, int) and isinstance(latent, int) and sample > 512 and latent > 0:
        ratio = sample // latent
        if ratio > 0:
            vae.tile_sample_min_size = 512
            vae.tile_latent_min_size = 512 // ratio


def model_device(vae):
    device = getattr(vae, "device", None)
    if device is not None:
        return device
    return next(vae.parameters()).device


def decode_with_policy(vae, latent, device, policy, decode_latent, *, event_sink=None,
                       budget_service=cpu_decode_budget):
    """Return an image while preserving device, dtype and original tile settings."""
    if policy not in ("auto", "tiled", "cpu"):
        raise ValueError("sample_vae_decode must be auto, tiled, or cpu")
    original_device, original_dtype = model_device(vae), vae.dtype
    original_tiling = getattr(vae, "use_tiling", None)
    original_tile_sizes = {name: getattr(vae, name) for name in
                           ("tile_sample_min_size", "tile_latent_min_size") if hasattr(vae, name)}
    stages = ("normal", "tiled", "cpu") if policy == "auto" else (policy,)
    error = None
    try:
        for index, stage in enumerate(stages):
            emit_decode_event(event_sink, "attempt", stage=stage, policy=policy)
            try:
                if stage != "normal":
                    _tiling(vae, True)
                if stage == "cpu":
                    budget = budget_service(vae, latent)
                    emit_decode_event(event_sink, "cpu_budget", stage=stage, **budget)
                    if not budget["accepted"]:
                        raise PreviewDecodeBudgetError("Insufficient host RAM headroom for CPU preview decode")
                vae.to(torch.device("cpu") if stage == "cpu" else device)
                with torch.no_grad():
                    image = decode_latent(vae, latent)
                logger.info("Training preview effective VAE decode: %s on %s", stage, model_device(vae))
                emit_decode_event(event_sink, "success", stage=stage)
                return image
            except Exception as exc:
                cuda_oom = _is_cuda_oom(exc)
                oom = isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()
                emit_decode_event(event_sink, "oom" if oom else "reject", stage=stage,
                                  error_type=type(exc).__name__, error=str(exc), cuda_oom=cuda_oom)
                retry = (cuda_oom and policy == "auto" and stage != "cpu"
                         and index < len(stages) - 1)
                # Terminal errors also leave feature-cache tensors on GPU.
                try:
                    _clear_failed_decode(vae, exc)
                except Exception as cleanup_error:
                    fatal = PreviewDecodeRestoreError("Training preview failed to clear VAE feature cache")
                    fatal.add_note(f"Original decode error: {exc!r}")
                    emit_decode_event(event_sink, "restoration_fatal", stage=stage, error=str(fatal))
                    raise fatal from cleanup_error
                if not retry:
                    raise
                logger.warning("Training preview VAE %s CUDA OOM; retrying %s", stage, stages[index + 1])
                emit_decode_event(event_sink, "fallback", stage=stage, next_stage=stages[index + 1])
    except BaseException as exc:
        error = exc
        raise
    finally:
        failures = []
        try:
            vae.to(original_device)
            if vae.dtype != original_dtype:
                vae.to(dtype=original_dtype)
        except BaseException as exc:
            failures.append(exc)
        try:
            if original_tiling is not None:
                _tiling(vae, original_tiling)
        except BaseException as exc:
            failures.append(exc)
        try:
            for name, value in original_tile_sizes.items():
                setattr(vae, name, value)
        except BaseException as exc:
            failures.append(exc)
        if failures:
            fatal = PreviewDecodeRestoreError("Training preview failed to restore VAE device/dtype/tiling")
            if error is not None:
                fatal.add_note(f"Original preview error: {error!r}")
            emit_decode_event(event_sink, "restoration_fatal", error=str(fatal))
            raise fatal from failures[0]
