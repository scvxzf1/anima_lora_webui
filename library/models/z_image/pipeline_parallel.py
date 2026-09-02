"""Z-Image main-layer adapter for pipeline-parallel probes."""

from __future__ import annotations

from torch import Tensor, nn

from library.models.pipeline_parallel import (
    BorrowedBlockStage,
    PipelinePlan,
    build_pipeline_block_stage,
)


class ZImageBlockStage(BorrowedBlockStage):
    """Run a contiguous range from ``model.layers``."""

    def __init__(
        self,
        blocks: list[nn.Module] | tuple[nn.Module, ...],
        *,
        stage_index: int = 0,
        block_range: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(
            blocks,
            source_block_container="layers",
            stage_index=stage_index,
            block_range=block_range,
        )

    def forward(
        self,
        hidden: Tensor,
        attn_mask: Tensor,
        freqs_cis: Tensor,
        adaln_input: Tensor | None = None,
        noise_mask: Tensor | None = None,
        adaln_noisy: Tensor | None = None,
        adaln_clean: Tensor | None = None,
    ) -> Tensor:
        for block in self.blocks:
            hidden = block(
                hidden,
                attn_mask,
                freqs_cis,
                adaln_input,
                noise_mask,
                adaln_noisy,
                adaln_clean,
            )
        return hidden


def build_z_image_block_stage(
    model: nn.Module,
    *,
    stage_index: int,
    stages: int,
) -> tuple[PipelinePlan, ZImageBlockStage]:
    plan, stage = build_pipeline_block_stage(
        model,
        family="z_image",
        stage_index=stage_index,
        stages=stages,
    )
    if not isinstance(stage, ZImageBlockStage):
        raise TypeError("pipeline stage dispatch returned a non-Z-Image adapter")
    return plan, stage


__all__ = ["ZImageBlockStage", "build_z_image_block_stage"]
