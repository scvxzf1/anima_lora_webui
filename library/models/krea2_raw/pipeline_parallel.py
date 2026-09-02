"""Experimental Krea-2 pipeline-parallel planning primitives.

The regular trainer still uses Accelerate data parallelism.  This module keeps
the PP contract explicit while the 1F1B schedule is integrated into the train
loop: it validates the topology, computes deterministic contiguous block
ownership, and exposes a small stage wrapper for standalone probes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from torch import Tensor, nn


PIPELINE_PARALLEL_SCHEDULES = ("1f1b",)
PIPELINE_PARALLEL_SPLITS = ("balanced",)
MAX_PIPELINE_PARALLEL_STAGES = 2
MAX_PIPELINE_PARALLEL_MICROBATCHES = 1024


def _get(config: Mapping[str, Any] | object, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Krea2PipelineParallelConfig:
    """Normalized PP settings shared by CLI, preflight, and probes."""

    enabled: bool = False
    stages: int = 2
    microbatches: int = 4
    schedule: str = "1f1b"
    split: str = "balanced"

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any] | object
    ) -> "Krea2PipelineParallelConfig":
        return cls(
            enabled=_bool(_get(config, "pipeline_parallel", False)),
            stages=_int(_get(config, "pipeline_parallel_stages", 2), 2),
            microbatches=_int(_get(config, "pipeline_parallel_microbatches", 4), 4),
            schedule=str(_get(config, "pipeline_parallel_schedule", "1f1b") or "1f1b")
            .strip()
            .lower(),
            split=str(_get(config, "pipeline_parallel_split", "balanced") or "balanced")
            .strip()
            .lower(),
        )


@dataclass(frozen=True)
class Krea2PipelinePlan:
    """Contiguous main-block ownership for one PP topology."""

    num_blocks: int
    stages: int
    ranges: tuple[tuple[int, int], ...]

    def range_for_stage(self, stage_index: int) -> tuple[int, int]:
        if not 0 <= stage_index < self.stages:
            raise ValueError(
                f"stage_index must be in [0, {self.stages}), got {stage_index}"
            )
        return self.ranges[stage_index]

    def indices_for_stage(self, stage_index: int) -> tuple[int, ...]:
        start, end = self.range_for_stage(stage_index)
        return tuple(range(start, end))


def validate_krea2_pipeline_config(
    config: Mapping[str, Any] | object,
    *,
    world_size: int | None = None,
    num_blocks: int = 28,
) -> Krea2PipelineParallelConfig:
    """Validate PP settings without initializing a process group.

    ``world_size`` is optional so WebUI preflight can validate a saved config;
    the training process should always pass the actual distributed world size.
    """

    normalized = Krea2PipelineParallelConfig.from_config(config)
    if not normalized.enabled:
        return normalized

    family = str(_get(config, "model_family", "") or "").strip().lower()
    if family not in {"krea2", "krea2_raw"}:
        raise ValueError(
            "pipeline_parallel is currently supported only for model_family="
            "krea2_raw (alias: krea2)"
        )
    if normalized.stages != MAX_PIPELINE_PARALLEL_STAGES:
        raise ValueError(
            "pipeline_parallel_stages must be 2 in the current dual-GPU implementation"
        )
    if normalized.stages > num_blocks:
        raise ValueError(
            f"pipeline_parallel_stages={normalized.stages} exceeds Krea-2 block count {num_blocks}"
        )
    if (
        normalized.microbatches < 1
        or normalized.microbatches > MAX_PIPELINE_PARALLEL_MICROBATCHES
    ):
        raise ValueError(
            "pipeline_parallel_microbatches must be between 1 and "
            f"{MAX_PIPELINE_PARALLEL_MICROBATCHES}"
        )
    if normalized.schedule not in PIPELINE_PARALLEL_SCHEDULES:
        raise ValueError(
            "pipeline_parallel_schedule must be one of: "
            + ", ".join(PIPELINE_PARALLEL_SCHEDULES)
        )
    if normalized.split not in PIPELINE_PARALLEL_SPLITS:
        raise ValueError(
            "pipeline_parallel_split must be one of: "
            + ", ".join(PIPELINE_PARALLEL_SPLITS)
        )
    if world_size is not None and int(world_size) != normalized.stages:
        raise ValueError(
            "pipeline_parallel_stages must equal the distributed world size: "
            f"stages={normalized.stages}, world_size={world_size}"
        )

    unsupported = []
    if _int(_get(config, "blocks_to_swap", 0), 0) > 0:
        unsupported.append("blocks_to_swap")
    if _bool(_get(config, "torch_compile", False)):
        unsupported.append("torch_compile")
    if (
        str(_get(config, "selective_checkpoint", "off") or "off").strip().lower()
        != "off"
    ):
        unsupported.append("selective_checkpoint")
    if _bool(_get(config, "cpu_offload_checkpointing", False)):
        unsupported.append("cpu_offload_checkpointing")
    if _bool(_get(config, "unsloth_offload_checkpointing", False)):
        unsupported.append("unsloth_offload_checkpointing")
    if not _bool(_get(config, "network_train_unet_only", True), True):
        unsupported.append("network_train_unet_only=false")
    if unsupported:
        raise ValueError(
            "Krea-2 pipeline_parallel currently cannot be combined with: "
            + ", ".join(unsupported)
        )
    return normalized


def make_krea2_pipeline_plan(
    *,
    stages: int,
    num_blocks: int = 28,
) -> Krea2PipelinePlan:
    """Split blocks into balanced contiguous ranges, preserving order."""

    if stages < 1:
        raise ValueError(f"stages must be positive, got {stages}")
    if num_blocks < stages:
        raise ValueError(f"num_blocks={num_blocks} must be >= stages={stages}")
    base, remainder = divmod(num_blocks, stages)
    widths = [base + (1 if index < remainder else 0) for index in range(stages)]
    # Krea-2 stage 0 also owns text fusion and input/timestep projections.  For
    # the supported two-stage topology, move one main block to stage 1 as the
    # initial load-balancing heuristic (28 blocks -> 13/15).  Hardware probes
    # can replace this with measured partitioning later without changing the
    # serialized ``balanced`` contract.
    if stages == 2 and widths[0] > 1:
        widths[0] -= 1
        widths[1] += 1
    ranges: list[tuple[int, int]] = []
    start = 0
    for width in widths:
        end = start + width
        ranges.append((start, end))
        start = end
    return Krea2PipelinePlan(num_blocks, stages, tuple(ranges))


class Krea2BlockStage(nn.Module):
    """Run one contiguous range of Krea-2 blocks.

    The wrapper intentionally exposes only the block-level contract.  The
    caller owns text fusion, timestep projection, positional encoding, and the
    final output layer, which keeps this class usable in CPU/Gloo stage probes.
    """

    def __init__(self, blocks: list[nn.Module] | tuple[nn.Module, ...]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        combined: Tensor,
        tvec: Tensor,
        freqs: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        for block in self.blocks:
            combined = block(combined, tvec, freqs, mask)
        return combined


def build_krea2_block_stage(
    model: nn.Module,
    *,
    stage_index: int,
    stages: int,
) -> tuple[Krea2PipelinePlan, Krea2BlockStage]:
    """Extract a stage wrapper from a Krea-2 model's ``blocks`` list."""

    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise TypeError("Krea-2 pipeline model must expose a blocks ModuleList")
    plan = make_krea2_pipeline_plan(stages=stages, num_blocks=len(blocks))
    start, end = plan.range_for_stage(stage_index)
    return plan, Krea2BlockStage(tuple(blocks[start:end]))


__all__ = [
    "Krea2BlockStage",
    "Krea2PipelineParallelConfig",
    "Krea2PipelinePlan",
    "MAX_PIPELINE_PARALLEL_MICROBATCHES",
    "MAX_PIPELINE_PARALLEL_STAGES",
    "PIPELINE_PARALLEL_SCHEDULES",
    "PIPELINE_PARALLEL_SPLITS",
    "build_krea2_block_stage",
    "make_krea2_pipeline_plan",
    "validate_krea2_pipeline_config",
]
