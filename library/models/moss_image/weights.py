"""Strict local Diffusers component loading for MOSS; no DiT quantization."""
from pathlib import Path
import torch


def _load_component(cls, path, component, *, dtype, device):
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("MOSS components require BF16, FP16 or FP32")
    root = Path(path).expanduser()
    if (root / component / "config.json").is_file():
        root = root / component
    if not (root / "config.json").is_file():
        raise ValueError(f"Missing local MOSS {component}/config.json: {root}")
    model, info = cls.from_pretrained(
        str(root), torch_dtype=dtype, local_files_only=True,
        low_cpu_mem_usage=True, output_loading_info=True,
    )
    errors = {key: info.get(key) for key in (
        "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"
    ) if info.get(key)}
    if errors:
        raise ValueError(f"MOSS {component} strict load failed: {errors}")
    return model.to(device=device, dtype=dtype).eval()


def load_moss_transformer(path, *, dtype, device="cpu"):
    from .dit import MOSSImageTransformer
    return _load_component(
        MOSSImageTransformer, path, "transformer", dtype=dtype, device=device
    )


def load_moss_vae(path, *, dtype, device="cpu"):
    from diffusers import AutoencoderKLFlux2
    vae = _load_component(
        AutoencoderKLFlux2, path, "vae", dtype=dtype, device=device
    )
    if vae.config.latent_channels != 32 or vae.bn.num_features != 128:
        raise ValueError("MOSS requires FLUX.2 VAE with native 32c / packed 128c")
    if tuple(vae.config.patch_size) != (2, 2):
        raise ValueError("MOSS requires FLUX.2 VAE patch_size=(2,2)")
    return vae
