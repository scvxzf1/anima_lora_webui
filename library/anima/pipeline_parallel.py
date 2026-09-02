"""Anima block-stage adapter for pipeline-parallel probes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch import Tensor, nn

from library.models.pipeline_parallel import (
    BorrowedBlockStage,
    PipelinePlan,
    build_pipeline_block_stage,
)


class AnimaBlockStage(BorrowedBlockStage):
    """Run a contiguous Anima block range while preserving global key metadata."""

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
        hidden: Tensor,
        embedding: Tensor,
        crossattn_emb: Tensor,
        attn_params: Any,
        rope_cos_sin: tuple[Tensor, Tensor] | None = None,
        adaln_lora: Tensor | None = None,
        use_fp32: bool = False,
        *,
        block_embeddings: Sequence[Tensor] | None = None,
    ) -> Tensor:
        """Run blocks; callers may supply per-block modulation embeddings."""

        embeddings = block_embeddings or (embedding,) * len(self.blocks)
        if len(embeddings) != len(self.blocks):
            raise ValueError(
                "block_embeddings must match the borrowed Anima block count: "
                f"{len(embeddings)} != {len(self.blocks)}"
            )
        for block, block_embedding in zip(self.blocks, embeddings, strict=True):
            hidden = block(
                hidden,
                block_embedding,
                crossattn_emb,
                attn_params,
                rope_cos_sin=rope_cos_sin,
                adaln_lora_B_T_3D=adaln_lora,
                use_fp32=use_fp32,
            )
        return hidden


def build_anima_block_stage(
    model: nn.Module,
    *,
    stage_index: int,
    stages: int,
) -> tuple[PipelinePlan, AnimaBlockStage]:
    plan, stage = build_pipeline_block_stage(
        model,
        family="anima",
        stage_index=stage_index,
        stages=stages,
    )
    if not isinstance(stage, AnimaBlockStage):
        raise TypeError("pipeline stage dispatch returned a non-Anima adapter")
    return plan, stage


__all__ = ["AnimaBlockStage", "build_anima_block_stage"]
