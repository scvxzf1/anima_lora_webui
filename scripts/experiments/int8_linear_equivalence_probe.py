#!/usr/bin/env python
"""Small-batch equivalence probe for frozen int8 base Linear storage.

This is a Phase-1 smoke test for the selective int8 route. It does not alter
the training entrypoint. The default probe uses an Anima-shaped module surface;
``--model-kind anima`` uses the real ``library.anima.models.Anima`` tiny
forward path, and ``--model-kind checkpoint`` loads a full checkpoint with a
real cached latent/text batch:

* frozen ``blocks.*.mlp.layer{1,2}`` and optional attention projections,
* a trainable adapter outside the frozen base path,
* bf16 baseline vs int8-stored frozen base Linear weights,
* output, loss, and trainable-adapter gradient-norm comparison.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn import functional as F

from library.runtime.int8_linear import (
    patch_lora_frozen_base_forwards_with_int8,
    replace_frozen_base_linears_with_int8,
    selected_int8_linear_modules,
)


DEFAULT_DIT_PATH = Path("models/diffusion_models/anima-preview3-base.safetensors")
DEFAULT_DATA_DIR = Path("post_image_dataset/rokkotsu_goddess")
PIXEL_RES_SUFFIX_RE = re.compile(r"_(?P<w>\d{3,5})x(?P<h>\d{3,5})$")


@dataclass(frozen=True)
class CachedBatchPair:
    latent_path: Path
    text_path: Path
    base_stem: str


class ProbeBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(dim, dim, bias=False)
        self.self_attn.output_proj = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Module()
        self.mlp.layer1 = nn.Linear(dim, hidden_dim, bias=False)
        self.mlp.layer2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.self_attn.output_proj(torch.tanh(self.self_attn.q_proj(x)))
        hidden = self.mlp.layer1(x + attn)
        return self.mlp.layer2(F.gelu(hidden))


class ProbeModel(nn.Module):
    def __init__(self, *, dim: int, hidden_dim: int, num_blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [ProbeBlock(dim, hidden_dim) for _ in range(num_blocks)]
        )
        self.adapter = nn.Linear(dim, dim, bias=False)
        self.final_layer = nn.Linear(dim, dim, bias=False)

        for block in self.blocks:
            block.requires_grad_(False)
        self.final_layer.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_layer(hidden) + self.adapter(x)


class TinyAnimaProbe(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        hidden_dim: int,
        num_blocks: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        from library.anima.models import Anima

        num_heads = 2
        if dim % num_heads != 0:
            raise ValueError("tiny Anima dim must be divisible by 2 heads")
        head_dim = dim // num_heads
        dim_h = head_dim // 6 * 2
        dim_t = head_dim - 2 * dim_h
        if dim_h <= 2 or dim_t <= 2:
            raise ValueError("tiny Anima dim is too small for 3D RoPE; use dim>=24")
        self.anima = Anima(
            max_img_h=4,
            max_img_w=4,
            max_frames=1,
            in_channels=4,
            out_channels=4,
            patch_spatial=2,
            patch_temporal=1,
            concat_padding_mask=False,
            model_channels=dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            mlp_ratio=hidden_dim / dim,
            crossattn_emb_channels=dim,
            pos_emb_learnable=True,
            use_adaln_lora=True,
            adaln_lora_dim=max(4, dim // 3),
            use_llm_adapter=False,
            attn_mode="torch",
        )
        self.adapter = nn.Conv3d(4, 4, kernel_size=1, bias=False)
        self.anima.requires_grad_(False)
        self.to(dtype=dtype)

    @property
    def blocks(self) -> nn.ModuleList:
        return self.anima.blocks

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        return self.anima(x + self.adapter(x), timesteps, context)


class CheckpointAnimaProbe(nn.Module):
    """Full-checkpoint probe wrapper with one tiny trainable input adapter."""

    def __init__(
        self,
        anima: nn.Module,
        *,
        dtype: torch.dtype,
        device: torch.device,
        adapter_state: dict[str, torch.Tensor],
    ) -> None:
        super().__init__()
        self.anima = anima
        self.input_adapter = nn.Conv3d(16, 16, kernel_size=1, bias=False)
        self.input_adapter.load_state_dict(adapter_state)
        self.input_adapter.to(device=device, dtype=dtype)

    @property
    def blocks(self) -> nn.ModuleList:
        return self.anima.blocks

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.anima(
            x + self.input_adapter(x),
            timesteps,
            context,
            padding_mask=padding_mask,
        )


class CheckpointLoraProbe(nn.Module):
    """Full-checkpoint probe wrapper with a real LoRA monkey-patched network."""

    def __init__(self, anima: nn.Module, network: nn.Module) -> None:
        super().__init__()
        self.anima = anima
        self.network = network

    @property
    def blocks(self) -> nn.ModuleList:
        return self.anima.blocks

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.anima(
            x,
            timesteps,
            context,
            padding_mask=padding_mask,
        )


def _dtype_from_name(name: str) -> torch.dtype:
    normalized = name.strip().lower()
    if normalized == "bf16":
        return torch.bfloat16
    if normalized == "fp16":
        return torch.float16
    if normalized == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _default_device_name() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is not available: {name}")
    return device


def _latent_cache_base_stem(path: Path) -> str | None:
    stem = path.name
    if not stem.endswith("_anima.npz"):
        return None
    base = stem[: -len("_anima.npz")]
    return PIXEL_RES_SUFFIX_RE.sub("", base)


def _text_cache_base_stem(path: Path) -> str | None:
    stem = path.name
    if not stem.endswith("_anima_te.safetensors"):
        return None
    return stem[: -len("_anima_te.safetensors")]


def discover_cached_batch_pairs(data_dir: Path) -> list[CachedBatchPair]:
    """Find latent/TE cache pairs produced by Anima preprocessing."""
    data_dir = Path(data_dir)
    latent_paths = sorted(data_dir.rglob("*_anima.npz"))
    text_paths = sorted(data_dir.rglob("*_anima_te.safetensors"))
    text_by_key: dict[tuple[Path, str], Path] = {}
    for path in text_paths:
        base_stem = _text_cache_base_stem(path)
        if base_stem is None:
            continue
        text_by_key[(path.parent, base_stem)] = path

    pairs: list[CachedBatchPair] = []
    for latent_path in latent_paths:
        base_stem = _latent_cache_base_stem(latent_path)
        if base_stem is None:
            continue
        text_path = text_by_key.get((latent_path.parent, base_stem))
        if text_path is None:
            continue
        pairs.append(
            CachedBatchPair(
                latent_path=latent_path,
                text_path=text_path,
                base_stem=base_stem,
            )
        )
    return pairs


def select_cached_batch_pair(data_dir: Path, cache_index: int) -> CachedBatchPair:
    pairs = discover_cached_batch_pairs(data_dir)
    if not pairs:
        raise FileNotFoundError(f"no matched *_anima.npz / *_anima_te.safetensors pairs under {data_dir}")
    if cache_index < 0 or cache_index >= len(pairs):
        raise IndexError(f"cache index {cache_index} is outside 0..{len(pairs) - 1}")
    return pairs[cache_index]


def _parse_text_variant(value: int | str) -> int | str:
    if value == "random":
        return "random"
    return int(value)


def _make_adapter_state(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    adapter = nn.Conv3d(16, 16, kernel_size=1, bias=False)
    with torch.no_grad():
        adapter.weight.copy_(torch.randn(adapter.weight.shape, generator=generator) * 1e-4)
    return {key: value.detach().clone() for key, value in adapter.state_dict().items()}


def _create_checkpoint_lora_network(
    anima: nn.Module,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    network_dim: int,
    network_alpha: float,
    target_scope: str,
) -> nn.Module:
    from networks.lora_anima.factory import create_network

    target_modules = selected_int8_linear_modules(target_scope)
    target_pattern = "|".join(re.escape(name) for name in sorted(target_modules))
    torch.manual_seed(seed)
    network = create_network(
        1.0,
        network_dim,
        network_alpha,
        None,
        [],
        anima,
        exclude_patterns=[
            rf"^(?!blocks\.\d+\.({target_pattern})$).*",
        ],
        train_llm_adapter="false",
        lora_fp32_compute="false",
    )
    network.apply_to([], anima, apply_text_encoder=False, apply_unet=True)
    network.to(device=device, dtype=dtype)
    network.train()
    return network


def _load_checkpoint_batch(
    pair: CachedBatchPair,
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    text_variant: int | str,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, dict[str, Any]]:
    from library.io.cache import load_cached_crossattn_emb, load_cached_latents

    latents, resolution, orig_h, orig_w = load_cached_latents(str(pair.latent_path))
    crossattn_emb = load_cached_crossattn_emb(
        str(pair.text_path),
        variant=_parse_text_variant(text_variant),
    )
    if crossattn_emb is None:
        raise RuntimeError(f"missing crossattn_emb in {pair.text_path}")

    latents = latents.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device=device, dtype=dtype)
    context = crossattn_emb.unsqueeze(0).repeat(batch_size, 1, 1).to(device=device, dtype=dtype)

    generator_device = device.type if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed + 17)
    noise = torch.randn(latents.shape, generator=generator, device=device, dtype=dtype)
    sigmas = torch.rand((batch_size,), generator=generator, device=device, dtype=torch.float32)
    sigmas = sigmas.clamp(1e-4, 1.0 - 1e-4)
    sigmas_for_latents = sigmas.to(dtype=dtype).view(-1, 1, 1, 1)
    noisy_model_input = (1.0 - sigmas_for_latents) * latents + sigmas_for_latents * noise
    target = noise - latents
    padding_mask = torch.zeros(
        batch_size,
        1,
        latents.shape[-2],
        latents.shape[-1],
        dtype=dtype,
        device=device,
    )

    metadata = {
        "latent_path": str(pair.latent_path),
        "text_path": str(pair.text_path),
        "base_stem": pair.base_stem,
        "latent_resolution": resolution,
        "original_size": [orig_w, orig_h],
        "latent_shape": list(latents.shape[1:]),
        "context_shape": list(context.shape[1:]),
        "sigma_min": float(sigmas.min().item()),
        "sigma_max": float(sigmas.max().item()),
    }
    model_inputs = (
        noisy_model_input.unsqueeze(2),
        sigmas.to(dtype=dtype),
        context,
        padding_mask,
    )
    return model_inputs, target.unsqueeze(2), metadata


def _relative_delta(next_value: float, base_value: float) -> float:
    denom = abs(base_value)
    if denom <= 1e-12:
        return 0.0 if abs(next_value) <= 1e-12 else math.inf
    return abs(next_value - base_value) / denom


def _grad_norm(model: nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if not param.requires_grad or param.grad is None:
            continue
        grad = param.grad.detach().float()
        total += float(torch.sum(grad * grad).item())
    return math.sqrt(total)


def _run_step(
    model: nn.Module,
    model_inputs: tuple[torch.Tensor, ...],
    target: torch.Tensor,
    *,
    capture_blocks: bool = False,
    forward_only: bool = False,
) -> dict[str, Any]:
    captured: dict[str, torch.Tensor] = {}
    hooks = []
    if capture_blocks:
        blocks = getattr(model, "blocks", None)
        if blocks is not None:
            for idx, block in enumerate(blocks):
                hooks.append(
                    block.register_forward_hook(
                        lambda _module, _inputs, output, idx=idx: captured.__setitem__(
                            f"blocks.{idx}", output.detach().float().cpu()
                        )
                    )
                )
    model.zero_grad(set_to_none=True)
    try:
        grad_context = torch.no_grad() if forward_only else contextlib.nullcontext()
        with grad_context:
            output = model(*model_inputs)
            loss = F.mse_loss(output.float(), target.float())
        grad_norm = None
        if not forward_only:
            loss.backward()
            grad_norm = _grad_norm(model)
        return {
            "output": output.detach().float().cpu(),
            "loss": float(loss.detach().item()),
            "grad_norm": grad_norm,
            "block_outputs": captured,
        }
    finally:
        for hook in hooks:
            hook.remove()


def _tensor_rel_l2_and_cosine(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
) -> tuple[float, float]:
    diff = candidate - baseline
    output_norm = float(baseline.norm().item())
    rel_l2 = float(diff.norm().item()) / output_norm if output_norm > 0 else 0.0
    cosine = (
        float(F.cosine_similarity(baseline.reshape(1, -1), candidate.reshape(1, -1), dim=1).item())
        if baseline.numel()
        else 1.0
    )
    return rel_l2, cosine


def _block_output_deltas(
    baseline: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    out = {}
    for key, base in sorted(baseline.items()):
        if key not in candidate:
            continue
        rel_l2, cosine = _tensor_rel_l2_and_cosine(base, candidate[key])
        out[key] = {"relative_l2": rel_l2, "cosine": cosine}
    return out


def _finish_result(
    *,
    model_kind: str,
    scope: str,
    seed: int,
    dtype: torch.dtype,
    batch_size: int,
    dim: int,
    hidden_dim: int,
    num_blocks: int,
    replacements,
    baseline_step: dict[str, Any],
    int8_step: dict[str, Any],
    max_output_rel_l2: float,
    max_loss_rel_delta: float,
    max_grad_norm_rel_delta: float,
) -> dict[str, Any]:
    output_rel_l2, output_cosine = _tensor_rel_l2_and_cosine(
        baseline_step["output"],
        int8_step["output"],
    )
    block_output_deltas = _block_output_deltas(
        baseline_step.get("block_outputs", {}),
        int8_step.get("block_outputs", {}),
    )
    block_output_rel_l2_max = max(
        (item["relative_l2"] for item in block_output_deltas.values()),
        default=0.0,
    )
    loss_rel_delta = _relative_delta(int8_step["loss"], baseline_step["loss"])
    baseline_grad_norm = baseline_step["grad_norm"]
    int8_grad_norm = int8_step["grad_norm"]
    grad_norm_rel_delta = (
        _relative_delta(int8_grad_norm, baseline_grad_norm)
        if baseline_grad_norm is not None and int8_grad_norm is not None
        else None
    )
    payload_bytes = sum(item.payload_bytes for item in replacements)
    bf16_bytes = sum(item.bf16_bytes for item in replacements)
    gate_pass = (
        output_rel_l2 <= max_output_rel_l2
        and loss_rel_delta <= max_loss_rel_delta
        and (
            grad_norm_rel_delta is None
            or grad_norm_rel_delta <= max_grad_norm_rel_delta
        )
    )
    return {
        "model_kind": model_kind,
        "scope": scope,
        "seed": seed,
        "dtype": str(dtype).replace("torch.", ""),
        "batch_size": batch_size,
        "dim": dim,
        "hidden_dim": hidden_dim,
        "num_blocks": num_blocks,
        "replacement_count": len(replacements),
        "replacements": [item.__dict__ for item in replacements],
        "payload_bytes": payload_bytes,
        "bf16_bytes": bf16_bytes,
        "payload_ratio_vs_bf16": payload_bytes / bf16_bytes if bf16_bytes else 0.0,
        "baseline_loss": baseline_step["loss"],
        "int8_loss": int8_step["loss"],
        "loss_rel_delta": loss_rel_delta,
        "baseline_grad_norm": baseline_grad_norm,
        "int8_grad_norm": int8_grad_norm,
        "grad_norm_rel_delta": grad_norm_rel_delta,
        "output_rel_l2": output_rel_l2,
        "output_cosine": output_cosine,
        "block_output_rel_l2_max": block_output_rel_l2_max,
        "block_output_deltas": block_output_deltas,
        "thresholds": {
            "max_output_rel_l2": max_output_rel_l2,
            "max_loss_rel_delta": max_loss_rel_delta,
            "max_grad_norm_rel_delta": max_grad_norm_rel_delta,
        },
        "gate_pass": gate_pass,
    }


def run_probe(
    *,
    scope: str = "mlp",
    seed: int = 0,
    dtype: torch.dtype = torch.bfloat16,
    batch_size: int = 4,
    dim: int = 16,
    hidden_dim: int = 64,
    num_blocks: int = 2,
    max_output_rel_l2: float = 0.03,
    max_loss_rel_delta: float = 0.05,
    max_grad_norm_rel_delta: float = 0.05,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    baseline = ProbeModel(dim=dim, hidden_dim=hidden_dim, num_blocks=num_blocks).to(dtype=dtype)
    int8_model = copy.deepcopy(baseline)

    replacements = replace_frozen_base_linears_with_int8(int8_model, scope=scope)

    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    x = torch.randn(batch_size, dim, generator=generator, dtype=dtype)
    target = torch.randn(batch_size, dim, generator=generator, dtype=dtype)

    baseline_step = _run_step(baseline, (x,), target)
    int8_step = _run_step(int8_model, (x,), target)

    return _finish_result(
        model_kind="toy",
        scope=scope,
        seed=seed,
        dtype=dtype,
        batch_size=batch_size,
        dim=dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        replacements=replacements,
        baseline_step=baseline_step,
        int8_step=int8_step,
        max_output_rel_l2=max_output_rel_l2,
        max_loss_rel_delta=max_loss_rel_delta,
        max_grad_norm_rel_delta=max_grad_norm_rel_delta,
    )


def run_anima_probe(
    *,
    scope: str = "mlp",
    seed: int = 0,
    dtype: torch.dtype = torch.bfloat16,
    batch_size: int = 2,
    dim: int = 24,
    hidden_dim: int = 48,
    num_blocks: int = 2,
    max_output_rel_l2: float = 0.03,
    max_loss_rel_delta: float = 0.05,
    max_grad_norm_rel_delta: float = 0.05,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    baseline = TinyAnimaProbe(
        dim=dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dtype=dtype,
    )
    int8_model = copy.deepcopy(baseline)
    baseline.train()
    int8_model.train()

    replacements = replace_frozen_base_linears_with_int8(int8_model.anima, scope=scope)

    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    x = torch.randn(batch_size, 4, 1, 4, 4, generator=generator, dtype=dtype)
    context = torch.randn(batch_size, 5, dim, generator=generator, dtype=dtype)
    timesteps = torch.rand(batch_size, generator=generator, dtype=dtype)
    target = torch.randn(batch_size, 4, 1, 4, 4, generator=generator, dtype=dtype)

    model_inputs = (x, timesteps, context)
    baseline_step = _run_step(baseline, model_inputs, target, capture_blocks=True)
    int8_step = _run_step(int8_model, model_inputs, target, capture_blocks=True)

    return _finish_result(
        model_kind="anima",
        scope=scope,
        seed=seed,
        dtype=dtype,
        batch_size=batch_size,
        dim=dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        replacements=replacements,
        baseline_step=baseline_step,
        int8_step=int8_step,
        max_output_rel_l2=max_output_rel_l2,
        max_loss_rel_delta=max_loss_rel_delta,
        max_grad_norm_rel_delta=max_grad_norm_rel_delta,
    )


def _cuda_peak_stats(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {}
    torch.cuda.synchronize(device)
    return {
        "max_memory_allocated": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved": int(torch.cuda.max_memory_reserved(device)),
    }


def _prepare_cuda_peak_stats(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)


def _cleanup_after_checkpoint_run(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def _load_checkpoint_probe_model(
    *,
    dit_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    attn_mode: str,
    adapter_kind: str,
    adapter_state: dict[str, torch.Tensor] | None,
    adapter_seed: int,
    lora_rank: int,
    lora_alpha: float,
    lora_target_scope: str,
    gradient_checkpointing: bool,
) -> nn.Module:
    from library.anima.weights import load_anima_model

    anima = load_anima_model(
        device=device,
        dit_path=str(dit_path),
        attn_mode=attn_mode,
        loading_device=device,
        dit_weight_dtype=dtype,
    )
    anima.to(device=device, dtype=dtype).requires_grad_(False)
    anima.reset_mod_guidance()
    if gradient_checkpointing:
        anima.enable_gradient_checkpointing()
    if adapter_kind == "lora":
        network = _create_checkpoint_lora_network(
            anima,
            seed=adapter_seed,
            device=device,
            dtype=dtype,
            network_dim=lora_rank,
            network_alpha=lora_alpha,
            target_scope=lora_target_scope,
        )
        model = CheckpointLoraProbe(anima, network)
    else:
        if adapter_state is None:
            raise ValueError("input adapter probe requires adapter_state")
        model = CheckpointAnimaProbe(
            anima,
            dtype=dtype,
            device=device,
            adapter_state=adapter_state,
        )
    model.train()
    return model


def _run_checkpoint_model_step(
    *,
    dit_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    attn_mode: str,
    adapter_kind: str,
    adapter_state: dict[str, torch.Tensor],
    adapter_seed: int,
    lora_rank: int,
    lora_alpha: float,
    lora_target_scope: str,
    gradient_checkpointing: bool,
    model_inputs: tuple[torch.Tensor, ...],
    target: torch.Tensor,
    scope: str,
    use_int8: bool,
    capture_blocks: bool,
    forward_only: bool,
) -> tuple[dict[str, Any], list[Any], dict[str, int]]:
    _prepare_cuda_peak_stats(device)
    model = _load_checkpoint_probe_model(
        dit_path=dit_path,
        device=device,
        dtype=dtype,
        attn_mode=attn_mode,
        adapter_kind=adapter_kind,
        adapter_state=adapter_state,
        adapter_seed=adapter_seed,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_target_scope=lora_target_scope,
        gradient_checkpointing=gradient_checkpointing and not forward_only,
    )
    replacements = []
    try:
        if use_int8:
            if adapter_kind == "lora":
                replacements = patch_lora_frozen_base_forwards_with_int8(
                    model.network,
                    scope=scope,
                )
            else:
                replacements = replace_frozen_base_linears_with_int8(
                    model.anima,
                    scope=scope,
                )
        step = _run_step(
            model,
            model_inputs,
            target,
            capture_blocks=capture_blocks,
            forward_only=forward_only,
        )
        peak_stats = _cuda_peak_stats(device)
        return step, replacements, peak_stats
    finally:
        del model
        _cleanup_after_checkpoint_run(device)


def run_checkpoint_probe(
    *,
    scope: str = "mlp",
    seed: int = 0,
    dtype: torch.dtype = torch.bfloat16,
    batch_size: int = 1,
    dit_path: Path = DEFAULT_DIT_PATH,
    data_dir: Path = DEFAULT_DATA_DIR,
    cache_index: int = 0,
    text_variant: int | str = 0,
    device: str | torch.device | None = None,
    attn_mode: str = "torch",
    adapter_kind: str = "input",
    lora_rank: int = 4,
    lora_alpha: float = 4.0,
    lora_target_scope: str = "mlp",
    gradient_checkpointing: bool = True,
    capture_blocks: bool = True,
    forward_only: bool = False,
    max_output_rel_l2: float = 0.03,
    max_loss_rel_delta: float = 0.05,
    max_grad_norm_rel_delta: float = 0.05,
) -> dict[str, Any]:
    if adapter_kind not in {"input", "lora"}:
        raise ValueError("adapter_kind must be 'input' or 'lora'")
    device_obj = _resolve_device(str(device or _default_device_name()))
    dit_path = Path(dit_path)
    data_dir = Path(data_dir)
    if not dit_path.exists():
        raise FileNotFoundError(f"DiT checkpoint not found: {dit_path}")

    pair = select_cached_batch_pair(data_dir, cache_index)
    model_inputs, target, batch_metadata = _load_checkpoint_batch(
        pair,
        batch_size=batch_size,
        seed=seed,
        device=device_obj,
        dtype=dtype,
        text_variant=text_variant,
    )
    adapter_state = _make_adapter_state(seed + 101)
    adapter_seed = seed + 101

    baseline_step, _baseline_replacements, baseline_peak = _run_checkpoint_model_step(
        dit_path=dit_path,
        device=device_obj,
        dtype=dtype,
        attn_mode=attn_mode,
        adapter_kind=adapter_kind,
        adapter_state=adapter_state,
        adapter_seed=adapter_seed,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_target_scope=lora_target_scope,
        gradient_checkpointing=gradient_checkpointing,
        model_inputs=model_inputs,
        target=target,
        scope=scope,
        use_int8=False,
        capture_blocks=capture_blocks,
        forward_only=forward_only,
    )
    int8_step, replacements, int8_peak = _run_checkpoint_model_step(
        dit_path=dit_path,
        device=device_obj,
        dtype=dtype,
        attn_mode=attn_mode,
        adapter_kind=adapter_kind,
        adapter_state=adapter_state,
        adapter_seed=adapter_seed,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_target_scope=lora_target_scope,
        gradient_checkpointing=gradient_checkpointing,
        model_inputs=model_inputs,
        target=target,
        scope=scope,
        use_int8=True,
        capture_blocks=capture_blocks,
        forward_only=forward_only,
    )

    result = _finish_result(
        model_kind="checkpoint",
        scope=scope,
        seed=seed,
        dtype=dtype,
        batch_size=batch_size,
        dim=2048,
        hidden_dim=8192,
        num_blocks=28,
        replacements=replacements,
        baseline_step=baseline_step,
        int8_step=int8_step,
        max_output_rel_l2=max_output_rel_l2,
        max_loss_rel_delta=max_loss_rel_delta,
        max_grad_norm_rel_delta=max_grad_norm_rel_delta,
    )
    result.update(
        {
            "dit_path": str(dit_path),
            "data_dir": str(data_dir),
            "cache_index": cache_index,
            "device": str(device_obj),
            "attn_mode": attn_mode,
            "adapter_kind": adapter_kind,
            "lora_rank": lora_rank if adapter_kind == "lora" else None,
            "lora_alpha": lora_alpha if adapter_kind == "lora" else None,
            "lora_target_scope": lora_target_scope if adapter_kind == "lora" else None,
            "gradient_checkpointing": bool(gradient_checkpointing and not forward_only),
            "capture_blocks": bool(capture_blocks),
            "forward_only": bool(forward_only),
            "batch": batch_metadata,
            "cuda_peak": {
                "baseline": baseline_peak,
                "int8": int8_peak,
            },
        }
    )
    return result


def _metric_summary(values: list[float | None]) -> dict[str, float | None]:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return {"min": None, "p50": None, "max": None}
    return {
        "min": clean[0],
        "p50": clean[len(clean) // 2],
        "max": clean[-1],
    }


def summarize_repeated_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    cache_indices = sorted(
        {
            int(item["cache_index"])
            for item in results
            if item.get("cache_index") is not None
        }
    )
    return {
        "gate_pass_all": all(bool(item.get("gate_pass")) for item in results),
        "gate_pass_count": sum(1 for item in results if item.get("gate_pass")),
        "run_count": len(results),
        "cache_indices": cache_indices,
        "seed_values": sorted({int(item["seed"]) for item in results if item.get("seed") is not None}),
        "replacement_count": results[0].get("replacement_count") if results else 0,
        "payload_ratio_vs_bf16": results[0].get("payload_ratio_vs_bf16") if results else 0.0,
        "output_rel_l2": _metric_summary([item.get("output_rel_l2") for item in results]),
        "loss_rel_delta": _metric_summary([item.get("loss_rel_delta") for item in results]),
        "grad_norm_rel_delta": _metric_summary(
            [item.get("grad_norm_rel_delta") for item in results]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", choices=("toy", "anima", "checkpoint"), default="toy")
    parser.add_argument(
        "--scope",
        default="mlp",
        help=(
            "Comma-separated scopes. Common values: mlp, attention, all, "
            "self_attn_qkv, self_attn_out, cross_attn_q, cross_attn_kv, "
            "cross_attn_out, attention_out."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--repeat-seeds",
        type=int,
        default=1,
        help="Run consecutive seeds starting at --seed and return an aggregate summary.",
    )
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--dit-path", type=Path, default=DEFAULT_DIT_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-index", type=int, default=0)
    parser.add_argument(
        "--repeat-caches",
        type=int,
        default=1,
        help="Checkpoint probes only: run consecutive cache indices starting at --cache-index.",
    )
    parser.add_argument("--text-variant", default="0")
    parser.add_argument("--device", default=_default_device_name())
    parser.add_argument("--attn-mode", default="torch")
    parser.add_argument(
        "--adapter-kind",
        choices=("input", "lora"),
        default="input",
        help=(
            "Checkpoint probe trainable surface. input keeps the legacy 1x1 "
            "Conv adapter; lora applies the real LoRA monkey-patch graph to "
            "block MLP/attention Linears only."
        ),
    )
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=4.0)
    parser.add_argument(
        "--lora-target-scope",
        default="mlp",
        help=(
            "Scope of block Linear modules that receive trainable LoRA in "
            "--adapter-kind lora checkpoint probes. Defaults to mlp; use "
            "mlp,cross_attn_q or all for wider experiments."
        ),
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable block gradient checkpointing for checkpoint probes.",
    )
    parser.add_argument(
        "--capture-blocks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture per-block output deltas for anima/checkpoint probes.",
    )
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Skip backward/grad metrics; useful when a full checkpoint probe OOMs.",
    )
    parser.add_argument("--max-output-rel-l2", type=float, default=0.03)
    parser.add_argument("--max-loss-rel-delta", type=float, default=0.05)
    parser.add_argument("--max-grad-norm-rel-delta", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    return parser.parse_args()


def _run_from_args(
    args: argparse.Namespace,
    *,
    seed: int,
    cache_index: int | None = None,
) -> dict[str, Any]:
    dtype = _dtype_from_name(args.dtype)
    if args.model_kind == "checkpoint":
        batch_size = args.batch_size if args.batch_size is not None else 1
        return run_checkpoint_probe(
            scope=args.scope,
            seed=seed,
            dtype=dtype,
            batch_size=batch_size,
            dit_path=args.dit_path,
            data_dir=args.data_dir,
            cache_index=args.cache_index if cache_index is None else cache_index,
            text_variant=args.text_variant,
            device=args.device,
            attn_mode=args.attn_mode,
            adapter_kind=args.adapter_kind,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_target_scope=args.lora_target_scope,
            gradient_checkpointing=args.gradient_checkpointing,
            capture_blocks=args.capture_blocks,
            forward_only=args.forward_only,
            max_output_rel_l2=args.max_output_rel_l2,
            max_loss_rel_delta=args.max_loss_rel_delta,
            max_grad_norm_rel_delta=args.max_grad_norm_rel_delta,
        )
    runner = run_anima_probe if args.model_kind == "anima" else run_probe
    dim = args.dim if args.dim is not None else (24 if args.model_kind == "anima" else 16)
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else (
        dim * 2 if args.model_kind == "anima" else 64
    )
    batch_size = args.batch_size if args.batch_size is not None else (
        2 if args.model_kind == "anima" else 4
    )
    return runner(
        scope=args.scope,
        seed=seed,
        dtype=dtype,
        batch_size=batch_size,
        dim=dim,
        hidden_dim=hidden_dim,
        num_blocks=args.num_blocks,
        max_output_rel_l2=args.max_output_rel_l2,
        max_loss_rel_delta=args.max_loss_rel_delta,
        max_grad_norm_rel_delta=args.max_grad_norm_rel_delta,
    )


def main() -> int:
    args = parse_args()
    if args.repeat_seeds < 1:
        raise ValueError("--repeat-seeds must be >= 1")
    if args.repeat_caches < 1:
        raise ValueError("--repeat-caches must be >= 1")
    if args.repeat_caches != 1 and args.model_kind != "checkpoint":
        raise ValueError("--repeat-caches is only supported for --model-kind checkpoint")

    if args.repeat_seeds == 1 and args.repeat_caches == 1:
        result = _run_from_args(args, seed=args.seed)
    else:
        results = []
        for cache_offset in range(args.repeat_caches):
            cache_index = args.cache_index + cache_offset
            for seed_offset in range(args.repeat_seeds):
                results.append(
                    _run_from_args(
                        args,
                        seed=args.seed + seed_offset,
                        cache_index=cache_index,
                    )
                )
        result = {
            "model_kind": args.model_kind,
            "scope": args.scope,
            "adapter_kind": args.adapter_kind if args.model_kind == "checkpoint" else None,
            "seed_start": args.seed,
            "repeat_seeds": args.repeat_seeds,
            "cache_index_start": args.cache_index,
            "repeat_caches": args.repeat_caches,
            "summary": summarize_repeated_results(results),
            "gate_pass": all(bool(item.get("gate_pass")) for item in results),
            "results": results,
        }
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
