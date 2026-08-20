"""DC-Gen Anima forge: build / save / load the new-latent-space base DiT.

The forged base = original Anima backbone + a new ``x_embedder`` (c32/p1) and
``final_layer`` (c32/p1). Every backbone parameter whose shape is unchanged is
copied from the original checkpoint; only the two latent-bound layers keep
their (random or alignment-trained) new values.

Checkpoint metadata records the latent-space contract so a loader can
reconstruct the right geometry without sidecar config files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from library.anima.models import Anima
from library.env import resolve_under_home
from library.models.latent_space import DCGEN_F32C32_P1, LatentSpaceSpec

logger = logging.getLogger(__name__)

_META_KEYS = (
    "dcgen_space",
    "latent_channels",
    "patch_spatial",
    "vae_spatial_compression",
    "dc_ae",
)


@dataclass
class ForgeReport:
    copied: int
    skipped: list[tuple[str, tuple[int, ...], tuple[int, ...]]]
    total: int


def forge_new_dit(
    old_dit: Anima,
    *,
    spec: LatentSpaceSpec = DCGEN_F32C32_P1,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[Anima, ForgeReport]:
    """Build the forged DiT from an already-loaded original Anima DiT.

    ``old_dit`` is expected on-device in training dtype. The new DiT is
    constructed with the target space geometry; all shape-matching parameters
    are copied in place. ``x_embedder.proj.1.weight``, ``final_layer.linear.weight``
    and the learnable position-embedding grid keep their fresh (random or
    alignment-trained) values.
    """
    device = device or old_dit.device
    dtype = dtype or old_dit.dtype
    new_dit = Anima(
        max_img_h=old_dit.max_img_h,
        max_img_w=old_dit.max_img_w,
        max_frames=old_dit.max_frames,
        in_channels=spec.latent_channels,
        out_channels=spec.latent_channels,
        patch_spatial=spec.patch_spatial,
        patch_temporal=spec.patch_temporal,
        vae_spatial_compression=spec.vae_spatial_compression,
        model_channels=old_dit.model_channels,
        num_blocks=old_dit.num_blocks,
        num_heads=old_dit.num_heads,
        use_llm_adapter=old_dit.use_llm_adapter,
    ).to(device=device, dtype=dtype)

    new_state = new_dit.state_dict()
    copied = 0
    skipped: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for k, v in old_dit.state_dict().items():
        if k in new_state and new_state[k].shape == v.shape:
            new_state[k].copy_(v)
            copied += 1
        elif k in new_state:
            skipped.append((k, tuple(new_state[k].shape), tuple(v.shape)))
    report = ForgeReport(copied=copied, skipped=skipped, total=len(new_state))
    logger.info(
        "forged DiT: copied %d/%d params, skipped %d shape-mismatched",
        copied,
        report.total,
        len(skipped),
    )
    return new_dit, report


def _metadata(spec: LatentSpaceSpec, dc_ae: str) -> dict[str, str]:
    return {
        "dcgen_space": spec.name,
        "latent_channels": str(spec.latent_channels),
        "patch_spatial": str(spec.patch_spatial),
        "vae_spatial_compression": str(spec.vae_spatial_compression),
        "dc_ae": dc_ae,
    }


def save_forged_dit(
    new_dit: Anima,
    path: str | Path,
    *,
    spec: LatentSpaceSpec = DCGEN_F32C32_P1,
    dc_ae: str = "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
) -> Path:
    """Save the forged base as a single safetensors checkpoint with metadata."""
    path = Path(resolve_under_home(str(path)))
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the checkpoint in its training dtype (bf16) so a forged 2.1B base
    # stays ~4.2GB instead of 8.4GB fp32.
    state = {k: v.detach().cpu() for k, v in new_dit.state_dict().items()}
    save_file(state, str(path), metadata=_metadata(spec, dc_ae))
    logger.info("saved forged DiT: %s (%d tensors)", path, len(state))
    return path


def load_forged_dit(
    path: str | Path,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    attn_mode: str = "torch",
) -> tuple[Anima, dict[str, str]]:
    """Load a forged base checkpoint produced by :func:`save_forged_dit`.

    Geometry is read back from the checkpoint metadata, so callers do not need
    to remember which space the file was forged for.
    """
    from library.anima.weights import load_anima_model

    path = Path(resolve_under_home(str(path)))
    meta = {}
    with __import__("safetensors").safe_open(str(path), framework="pt", device="cpu") as f:
        meta = dict(f.metadata() or {})

    missing_meta = [k for k in _META_KEYS if k not in meta]
    if missing_meta:
        raise ValueError(f"forged checkpoint missing metadata {missing_meta}: {path}")

    spec = LatentSpaceSpec(
        name=meta["dcgen_space"],
        vae_spatial_compression=int(meta["vae_spatial_compression"]),
        latent_channels=int(meta["latent_channels"]),
        patch_spatial=int(meta["patch_spatial"]),
        patch_temporal=1,
        cache_suffix=f"_{meta['dcgen_space']}.npz",
        normalization="scaling_factor",
        scaling_factor=DCGEN_F32C32_P1.scaling_factor,
    )
    dit = load_anima_model(
        device=torch.device(device),
        dit_path=str(path),
        attn_mode=attn_mode,
        loading_device=torch.device(device),
        dit_weight_dtype=dtype,
        in_channels=spec.latent_channels,
        out_channels=spec.latent_channels,
        patch_spatial=spec.patch_spatial,
        patch_temporal=spec.patch_temporal,
        vae_spatial_compression=spec.vae_spatial_compression,
    )
    return dit, meta
