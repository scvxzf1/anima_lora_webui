"""Functional-loss hook installation on DiT cross-attn projections."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def post_process_network(trainer, args, accelerator, network, text_encoders, unet) -> None:
    trainer._network = (
        network  # composer reads _network for ortho / balance regularizers
    )
    trainer._func_loss = None
    trainer._func_hooks = []
    trainer._func_captures = {}
    trainer._func_blocks = []
    if getattr(args, "functional_loss_weight", 0.0) > 0.0 and getattr(
        args, "inversion_dir", None
    ):
        blocks_str = getattr(args, "functional_loss_blocks", "8,12,16,20")
        try:
            trainer._func_blocks = sorted(
                int(b.strip()) for b in blocks_str.split(",") if b.strip()
            )
        except ValueError as e:
            raise ValueError(
                f"functional_loss_blocks must be comma-separated integers, got {blocks_str!r}"
            ) from e

        def _make_hook(block_idx: int):
            def _hook(_module, _inputs, output):
                # Save the cross_attn.output_proj output for this block.
                # Hook fires twice per step (main forward + inversion forward);
                # the main forward runs first, we snapshot before second forward overwrites.
                trainer._func_captures[block_idx] = output

            return _hook

        blocks_list = unet.blocks  # nn.ModuleList of 28 Anima DiT blocks
        num_blocks = len(blocks_list)
        for bi in trainer._func_blocks:
            if not (0 <= bi < num_blocks):
                raise ValueError(
                    f"functional_loss_blocks contains out-of-range index {bi} (model has {num_blocks} blocks)"
                )
            module = blocks_list[bi].cross_attn.output_proj
            h = module.register_forward_hook(_make_hook(bi))
            trainer._func_hooks.append(h)
        logger.info(
            f"Functional loss enabled: hooks on cross_attn.output_proj at blocks {trainer._func_blocks}, "
            f"weight={args.functional_loss_weight}, num_runs={args.functional_loss_num_runs}"
        )
