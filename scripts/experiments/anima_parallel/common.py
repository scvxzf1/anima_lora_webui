"""Shared model, cached-batch, checkpoint, and hardware helpers."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from scripts.experiments.int8_linear_equivalence_probe import (
    CachedBatchPair,
    _load_checkpoint_batch,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIT = Path(
    "/home/scv/nvme0n1p1/ComfyUI/models/diffusion_models/anima/"
    "Anima-2.9B-preview-v1-sha.safetensors"
)
DEFAULT_LATENT = Path(
    "post_image_dataset/lora/"
    "pixiv_壱珂_40462352_2022-07-26 09_58_15_100011311_p0_0896x1200_anima.npz"
)
DEFAULT_TEXT = Path(
    "output/runs/model-family-fused-flash-smoke-20260903/cache/anima/"
    "pixiv_壱珂_40462352_2022-07-26 09_58_15_100011311_p0_anima_te.safetensors"
)
_BLOCK_RE = re.compile(r"lora_unet_blocks_(\d+)_")


def init_distributed() -> tuple[int, int, int, torch.device, dist.ProcessGroup]:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 2:
        raise RuntimeError(f"Anima parallel probe requires exactly 2 ranks, got {world}")
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    gloo_group = dist.new_group(backend="gloo")
    return rank, local_rank, world, device, gloo_group


def load_cached_batches(
    latent_path: Path,
    text_path: Path,
    *,
    count: int,
    seed: int,
    dtype: torch.dtype,
) -> list[tuple[tuple[torch.Tensor, ...], torch.Tensor, dict[str, Any]]]:
    for path in (latent_path, text_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    suffix = "_anima.npz"
    stem = latent_path.name[: -len(suffix)] if latent_path.name.endswith(suffix) else latent_path.stem
    stem = re.sub(r"_\d{3,5}x\d{3,5}$", "", stem)
    pair = CachedBatchPair(latent_path=latent_path, text_path=text_path, base_stem=stem)
    return [
        _load_checkpoint_batch(
            pair,
            batch_size=1,
            seed=seed + index * 17,
            device=torch.device("cpu"),
            dtype=dtype,
            text_variant=0,
        )
        for index in range(count)
    ]


def to_device_batch(batch, device: torch.device, dtype: torch.dtype):
    inputs, target, metadata = batch
    moved = tuple(
        value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
        for value in inputs
    )
    return moved, target.to(device=device, dtype=dtype), metadata


def load_model(dit_path: Path, device: torch.device, attn_mode: str):
    from library.anima.weights import load_anima_model

    model = load_anima_model(
        device=device,
        dit_path=str(dit_path),
        attn_mode=attn_mode,
        loading_device=torch.device("cpu"),
        dit_weight_dtype=torch.bfloat16,
    )
    model.requires_grad_(False)
    model.reset_mod_guidance()
    return model


def create_mlp_lora(model, *, seed: int, rank_dim: int, alpha: float):
    from networks.lora_anima.factory import create_network

    torch.manual_seed(seed)
    network = create_network(
        1.0,
        rank_dim,
        alpha,
        None,
        [],
        model,
        exclude_patterns=[r"^(?!blocks\.\d+\.mlp\.layer[12]$).*"],
        train_llm_adapter="false",
        lora_fp32_compute="false",
        model_family="anima",
    )
    network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
    network.train()
    return network


def lora_block_index(lora) -> int:
    match = _BLOCK_RE.search(lora.lora_name)
    if match is None:
        raise ValueError(f"cannot resolve block index from {lora.lora_name}")
    return int(match.group(1))


def lora_state_for_blocks(network, block_indices: set[int]) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for lora in network.unet_loras:
        if lora_block_index(lora) not in block_indices:
            continue
        for key, value in lora.state_dict().items():
            state[f"{lora.lora_name}.{key}"] = value.detach().cpu().contiguous()
    return state


def gather_objects(value: Any, *, rank: int, group: dist.ProcessGroup) -> list[Any] | None:
    gathered = [None for _ in range(dist.get_world_size())] if rank == 0 else None
    dist.gather_object(value, gathered, dst=0, group=group)
    return gathered


def save_network_state(
    network,
    state: dict[str, torch.Tensor],
    path: Path,
    *,
    mode: str,
    steps: int,
) -> None:
    current = network.state_dict()
    missing = sorted(set(current) - set(state))
    if missing:
        raise RuntimeError(f"consolidated LoRA state is missing {missing[:5]}")
    for key, value in state.items():
        module_name, field = key.rsplit(".", 1)
        module = network.get_submodule(module_name)
        if field == "weight":
            setattr(module, field, torch.nn.Parameter(value, requires_grad=True))
        else:
            setattr(module, field, value)
    path.parent.mkdir(parents=True, exist_ok=True)
    network.save_weights(
        str(path),
        torch.bfloat16,
        {
            "ss_output_name": path.stem,
            "ss_steps": str(steps),
            "ss_parallel_probe_mode": mode,
            "ss_network_dim": str(network.cfg.lora_dim),
            "ss_network_alpha": str(network.cfg.alpha),
            "ss_network_module": "networks.lora_anima",
        },
    )


def hardware_record(rank: int, local_rank: int, device: torch.device) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    return {
        "rank": rank,
        "local_rank": local_rank,
        "device": str(device),
        "name": props.name,
        "total_memory_bytes": props.total_memory,
        "capability": f"{props.major}.{props.minor}",
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
