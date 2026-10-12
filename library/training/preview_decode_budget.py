"""Read-only, conservative host headroom check before preview CPU decoding."""

from dataclasses import asdict, dataclass

import torch

from library.training.auto_block_swap.resources import host_memory


class PreviewDecodeBudgetError(RuntimeError):
    """CPU preview cannot start within the estimated host headroom."""


@dataclass(frozen=True)
class DecodeEstimate:
    model_copy_bytes: int
    latent_bytes: int
    tile_workspace_bytes: int

    @property
    def estimated_bytes(self):
        return self.model_copy_bytes + self.latent_bytes + self.tile_workspace_bytes


def estimate_cpu_decode(vae, latent):
    """Estimate extra allocation, without copying any tensor or touching precision.

    Workspace is a heuristic: eight float32 256-channel tile features plus
    full-resolution pixel conversion buffers. It is not an allocator limit.
    """
    if not isinstance(latent, torch.Tensor):
        raise PreviewDecodeBudgetError("CPU preview budget requires a tensor latent")
    if not callable(getattr(vae, "parameters", None)) or not callable(getattr(vae, "buffers", None)):
        raise PreviewDecodeBudgetError("CPU preview budget requires inspectable VAE parameters/buffers")
    seen, model_bytes = set(), 0
    for tensors in (vae.parameters(), vae.buffers()):
        for tensor in tensors:
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            if tensor.device.type != "cpu":
                model_bytes += tensor.numel() * tensor.element_size()
    tile = getattr(vae, "tile_sample_min_size", 512)
    tile = min(tile, 512) if isinstance(tile, int) and tile > 0 else 512
    batch = latent.shape[0] if latent.ndim else 1
    pixels = batch * latent.shape[-2] * latent.shape[-1] * 64 if latent.ndim >= 2 else batch
    latent_bytes = latent.numel() * max(4, latent.element_size()) * 3
    workspace = batch * tile * tile * 256 * 4 * 8 + pixels * 3 * 4 * 3
    return DecodeEstimate(model_bytes, latent_bytes, workspace)


def cpu_decode_budget(vae, latent, *, estimator=estimate_cpu_decode, memory_reader=host_memory):
    estimate, memory = estimator(vae, latent), memory_reader()
    result = {**asdict(estimate), "estimated_bytes": estimate.estimated_bytes,
              "available_bytes": memory.available, "reserve_bytes": memory.reserve}
    result["accepted"] = estimate.estimated_bytes + memory.reserve <= memory.available
    return result
