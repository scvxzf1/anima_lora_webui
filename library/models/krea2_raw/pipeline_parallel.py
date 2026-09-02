"""Experimental Krea-2 pipeline-parallel planning primitives.

The regular trainer still uses Accelerate data parallelism.  This module keeps
the future PP contract explicit before the 1F1B schedule is integrated into the
train loop: it validates the topology, computes deterministic contiguous block
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


def _bool(value: Any, default: bool, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{field} must be a boolean, got {value!r}")


def _int(value: Any, default: int, *, field: str) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return int(value.strip(), 10)
        except ValueError as exc:
            raise ValueError(f"{field} must be an integer, got {value!r}") from exc
    raise ValueError(f"{field} must be an integer, got {value!r}")


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
        schedule = _get(config, "pipeline_parallel_schedule", "1f1b")
        split = _get(config, "pipeline_parallel_split", "balanced")
        return cls(
            enabled=_bool(
                _get(config, "pipeline_parallel", False),
                False,
                field="pipeline_parallel",
            ),
            stages=_int(
                _get(config, "pipeline_parallel_stages", 2),
                2,
                field="pipeline_parallel_stages",
            ),
            microbatches=_int(
                _get(config, "pipeline_parallel_microbatches", 4),
                4,
                field="pipeline_parallel_microbatches",
            ),
            schedule=str("1f1b" if schedule is None else schedule).strip().lower(),
            split=str("balanced" if split is None else split).strip().lower(),
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
    block_count = _int(num_blocks, 28, field="num_blocks")

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
    if normalized.stages > block_count:
        raise ValueError(
            f"pipeline_parallel_stages={normalized.stages} exceeds Krea-2 block count {block_count}"
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
    parsed_world_size = (
        None if world_size is None else _int(world_size, 1, field="world_size")
    )
    if parsed_world_size is not None and parsed_world_size != normalized.stages:
        raise ValueError(
            "pipeline_parallel_stages must equal the distributed world size: "
            f"stages={normalized.stages}, world_size={parsed_world_size}"
        )

    unsupported = []
    if (
        _int(
            _get(config, "blocks_to_swap", 0),
            0,
            field="blocks_to_swap",
        )
        > 0
    ):
        unsupported.append("blocks_to_swap")
    if _bool(
        _get(config, "torch_compile", False),
        False,
        field="torch_compile",
    ):
        unsupported.append("torch_compile")
    if (
        str(_get(config, "selective_checkpoint", "off") or "off").strip().lower()
        != "off"
    ):
        unsupported.append("selective_checkpoint")
    if _bool(
        _get(config, "cpu_offload_checkpointing", False),
        False,
        field="cpu_offload_checkpointing",
    ):
        unsupported.append("cpu_offload_checkpointing")
    if _bool(
        _get(config, "unsloth_offload_checkpointing", False),
        False,
        field="unsloth_offload_checkpointing",
    ):
        unsupported.append("unsloth_offload_checkpointing")
    if not _bool(
        _get(config, "network_train_unet_only", True),
        True,
        field="network_train_unet_only",
    ):
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

    This probe wrapper registers the same block objects that remain registered
    on the source model; it does not transfer parameter ownership.  Calling
    ``to()`` on it therefore also moves those source-model blocks and must not
    be used as rank placement.  ``block_range`` and ``state_dict_key_map`` keep
    global ownership/key metadata explicit until a real stage-local model
    builder exists.

    The caller still owns text fusion, timestep projection, positional
    encoding, and the final output layer.
    """

    def __init__(
        self,
        blocks: list[nn.Module] | tuple[nn.Module, ...],
        *,
        stage_index: int = 0,
        block_range: tuple[int, int] | None = None,
    ):
        super().__init__()
        block_items = tuple(blocks)
        start, end = block_range or (0, len(block_items))
        if start < 0 or end < start or end - start != len(block_items):
            raise ValueError(
                "block_range must be a non-negative half-open range matching "
                f"the borrowed block count, got range=({start}, {end}), "
                f"blocks={len(block_items)}"
            )
        self.stage_index = stage_index
        self.block_range = (start, end)
        self.blocks = nn.ModuleList(block_items)

    @property
    def global_block_indices(self) -> tuple[int, ...]:
        return tuple(range(*self.block_range))

    def global_state_dict_key(self, local_key: str) -> str:
        """Map a wrapper-local ``blocks.N`` key back to the source-model key."""

        parts = local_key.split(".", 2)
        if len(parts) < 2 or parts[0] != "blocks":
            raise ValueError(
                f"stage state key must start with 'blocks.N', got {local_key!r}"
            )
        try:
            local_index = int(parts[1])
        except ValueError as exc:
            raise ValueError(
                f"stage state key has invalid block index: {local_key!r}"
            ) from exc
        if not 0 <= local_index < len(self.blocks):
            raise ValueError(
                f"stage state key block index is out of range: {local_key!r}"
            )
        suffix = f".{parts[2]}" if len(parts) == 3 else ""
        return f"blocks.{self.block_range[0] + local_index}{suffix}"

    def state_dict_key_map(self) -> dict[str, str]:
        return {key: self.global_state_dict_key(key) for key in self.state_dict()}

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
    """Build a borrowing probe wrapper for one planned global block range."""

    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise TypeError("Krea-2 pipeline model must expose a blocks ModuleList")
    plan = make_krea2_pipeline_plan(stages=stages, num_blocks=len(blocks))
    start, end = plan.range_for_stage(stage_index)
    return plan, Krea2BlockStage(
        tuple(blocks[start:end]),
        stage_index=stage_index,
        block_range=(start, end),
    )


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
