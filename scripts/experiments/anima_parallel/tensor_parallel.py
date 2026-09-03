"""One-dimensional head/feature tensor parallelism for Anima blocks."""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType

import torch

from .collectives import copy_forward_reduce_backward, reduce_exact, reduce_forward


@dataclass(frozen=True)
class ShardSpec:
    dimension: int | None


def _replace_parameter(module, name: str, value: torch.Tensor, *, trainable: bool) -> None:
    setattr(module, name, torch.nn.Parameter(value.contiguous(), requires_grad=trainable))


def _plain_chunk(value: torch.Tensor, rank: int, world: int, dim: int) -> torch.Tensor:
    if value.shape[dim] % world:
        raise ValueError(f"shape {tuple(value.shape)} is not divisible by TP={world} on dim={dim}")
    return value.detach().chunk(world, dim=dim)[rank].clone()


def _fused_chunk(value: torch.Tensor, rank: int, world: int, groups: int) -> torch.Tensor:
    pieces = value.detach().chunk(groups, dim=0)
    return torch.cat([piece.chunk(world, dim=0)[rank] for piece in pieces], dim=0).clone()


def _lora_by_module(network) -> dict[int, object]:
    return {id(lora.org_module_ref[0]): lora for lora in network.unet_loras}


def _wrap_colwise(module) -> None:
    inner = module.forward

    def forward(_module, value):
        return inner(copy_forward_reduce_backward(value))

    module.forward = MethodType(forward, module)


def _wrap_rowwise(module) -> None:
    inner = module.forward

    def forward(_module, value):
        return reduce_forward(inner(value))

    module.forward = MethodType(forward, module)


def _shard_colwise(
    module,
    lora,
    *,
    rank: int,
    world: int,
    fused_groups: int = 1,
) -> dict[str, ShardSpec]:
    weight = (
        _fused_chunk(module.weight, rank, world, fused_groups)
        if fused_groups > 1
        else _plain_chunk(module.weight, rank, world, 0)
    )
    _replace_parameter(module, "weight", weight, trainable=False)
    module.out_features = weight.shape[0]
    specs: dict[str, ShardSpec] = {}
    if lora is not None:
        up = (
            _fused_chunk(lora.lora_up.weight, rank, world, fused_groups)
            if fused_groups > 1
            else _plain_chunk(lora.lora_up.weight, rank, world, 0)
        )
        _replace_parameter(lora.lora_up, "weight", up, trainable=True)
        lora.lora_up.out_features = up.shape[0]
        specs[f"{lora.lora_name}.lora_up.weight"] = ShardSpec(0)
        specs[f"{lora.lora_name}.lora_down.weight"] = ShardSpec(None)
    _wrap_colwise(module)
    return specs


def _shard_rowwise(module, lora, *, rank: int, world: int) -> dict[str, ShardSpec]:
    weight = _plain_chunk(module.weight, rank, world, 1)
    _replace_parameter(module, "weight", weight, trainable=False)
    module.in_features = weight.shape[1]
    specs: dict[str, ShardSpec] = {}
    if lora is not None:
        down = _plain_chunk(lora.lora_down.weight, rank, world, 1)
        _replace_parameter(lora.lora_down, "weight", down, trainable=True)
        lora.lora_down.in_features = down.shape[1]
        specs[f"{lora.lora_name}.lora_down.weight"] = ShardSpec(1)
        specs[f"{lora.lora_name}.lora_up.weight"] = ShardSpec(None)
    _wrap_rowwise(module)
    return specs


def parallelize_anima_blocks(model, network, *, rank: int, world: int) -> dict[str, ShardSpec]:
    if world != 2:
        raise ValueError("the probe currently supports TP2 only")
    loras = _lora_by_module(network)
    specs: dict[str, ShardSpec] = {}
    for block in model.blocks:
        self_attn = block.self_attn
        specs.update(
            _shard_colwise(
                self_attn.qkv_proj,
                loras.get(id(self_attn.qkv_proj)),
                rank=rank,
                world=world,
                fused_groups=3,
            )
        )
        specs.update(
            _shard_rowwise(
                self_attn.output_proj,
                loras.get(id(self_attn.output_proj)),
                rank=rank,
                world=world,
            )
        )
        self_attn.n_heads //= world
        self_attn._inner_dim //= world

        cross_attn = block.cross_attn
        specs.update(
            _shard_colwise(
                cross_attn.q_proj,
                loras.get(id(cross_attn.q_proj)),
                rank=rank,
                world=world,
            )
        )
        specs.update(
            _shard_colwise(
                cross_attn.kv_proj,
                loras.get(id(cross_attn.kv_proj)),
                rank=rank,
                world=world,
                fused_groups=2,
            )
        )
        specs.update(
            _shard_rowwise(
                cross_attn.output_proj,
                loras.get(id(cross_attn.output_proj)),
                rank=rank,
                world=world,
            )
        )
        cross_attn.n_heads //= world
        cross_attn._inner_dim //= world

        specs.update(
            _shard_colwise(
                block.mlp.layer1,
                loras.get(id(block.mlp.layer1)),
                rank=rank,
                world=world,
            )
        )
        specs.update(
            _shard_rowwise(
                block.mlp.layer2,
                loras.get(id(block.mlp.layer2)),
                rank=rank,
                world=world,
            )
        )
    return specs


def synchronize_replicated_lora_gradients(network, specs: dict[str, ShardSpec]) -> None:
    for name, parameter in network.named_parameters():
        spec = specs.get(name)
        if spec is None or spec.dimension is not None or parameter.grad is None:
            continue
        parameter.grad.copy_(reduce_exact(parameter.grad))


def consolidate_tp_state(
    states: list[dict[str, torch.Tensor]], specs: dict[str, ShardSpec]
) -> dict[str, torch.Tensor]:
    if len(states) != 2:
        raise ValueError("TP2 consolidation needs two rank states")
    result: dict[str, torch.Tensor] = {}
    for key in states[0]:
        spec = specs.get(key)
        if spec is None or spec.dimension is None:
            result[key] = states[0][key]
        else:
            result[key] = torch.cat([state[key] for state in states], dim=spec.dimension)
    return result
