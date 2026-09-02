"""Krea-2 compatibility facade for shared pipeline-parallel primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from torch import Tensor, nn

from library.models.family_registry import normalize_registered_family
from library.models.pipeline_parallel import (
    BorrowedBlockStage,
    MAX_PIPELINE_PARALLEL_MICROBATCHES,
    MAX_PIPELINE_PARALLEL_STAGES,
    PIPELINE_PARALLEL_SCHEDULES,
    PIPELINE_PARALLEL_SPLITS,
    PipelineParallelConfig,
    make_pipeline_plan,
    validate_pipeline_parallel_config,
)


@dataclass(frozen=True)
class Krea2PipelineParallelConfig(PipelineParallelConfig):
    """Backward-compatible name for the normalized shared PP config."""


@dataclass(frozen=True)
class Krea2PipelinePlan:
    """Backward-compatible Krea-2 plan type."""

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


class Krea2BlockStage(BorrowedBlockStage):
    """Run one contiguous range of borrowed Krea-2 blocks for probes."""

    def __init__(
        self,
        blocks: list[nn.Module] | tuple[nn.Module, ...],
        *,
        stage_index: int = 0,
        block_range: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(
            blocks,
            source_block_container="blocks",
            stage_index=stage_index,
            block_range=block_range,
        )

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


def _model_family(config: Mapping[str, Any] | object) -> Any:
    if isinstance(config, Mapping):
        return config.get("model_family", "")
    return getattr(config, "model_family", "")


def validate_krea2_pipeline_config(
    config: Mapping[str, Any] | object,
    *,
    world_size: int | None = None,
    num_blocks: int = 28,
) -> Krea2PipelineParallelConfig:
    """Preserve the Krea-only legacy validator on top of the shared contract."""

    normalized = Krea2PipelineParallelConfig.from_config(config)
    if normalized.enabled:
        try:
            canonical = normalize_registered_family(
                _model_family(config),
                source="pipeline_parallel model_family",
                allow_aliases=True,
            )
        except ValueError as exc:
            raise ValueError(
                "pipeline_parallel is currently supported only for model_family="
                "krea2_raw (alias: krea2)"
            ) from exc
        if canonical != "krea2_raw":
            raise ValueError(
                "pipeline_parallel is currently supported only for model_family="
                "krea2_raw (alias: krea2)"
            )
    validated = validate_pipeline_parallel_config(
        config,
        world_size=world_size,
        num_blocks=num_blocks,
    )
    return Krea2PipelineParallelConfig(**validated.__dict__)


def make_krea2_pipeline_plan(
    *,
    stages: int,
    num_blocks: int = 28,
) -> Krea2PipelinePlan:
    plan = make_pipeline_plan(
        family="krea2_raw",
        stages=stages,
        num_blocks=num_blocks,
    )
    return Krea2PipelinePlan(plan.num_blocks, plan.stages, plan.ranges)


def build_krea2_block_stage(
    model: nn.Module,
    *,
    stage_index: int,
    stages: int,
) -> tuple[Krea2PipelinePlan, Krea2BlockStage]:
    blocks = getattr(model, "blocks", None)
    if not isinstance(blocks, nn.ModuleList):
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
