"""CLI entry helpers for train.py __main__."""

from __future__ import annotations

import logging
import os
import sys
from typing import Callable

from library.config import schema as _config_schema
from library.config.io import read_config_from_file
from library.datasets import base as _datasets_base
from library.training import verify_command_line_training_args
from library.training.bootstrap import NETWORK_KWARG_ALLOWLIST

logger = logging.getLogger(__name__)


def build_network_extras() -> dict[str, _config_schema.ConfigKey]:
    return {
        k: _config_schema.ConfigKey(name=k, type="str", source="network_module")
        for k in NETWORK_KWARG_ALLOWLIST
    }


def run_training_cli(
    *,
    setup_parser: Callable,
    trainer_factory: Callable,
    install_stop_signal_handlers: Callable[[], None],
    install_crash_reporter: Callable[[list[str]], None],
    argv: list[str] | None = None,
) -> None:
    install_stop_signal_handlers()
    install_crash_reporter(list(argv if argv is not None else sys.argv))
    parser = setup_parser()
    _config_schema.populate_schema(parser, extras=build_network_extras())

    args = parser.parse_args(argv[1:] if argv is not None else None)
    verify_command_line_training_args(args)
    args = read_config_from_file(args, parser)

    from library.models.pipeline_parallel import (
        PipelineParallelConfig,
        validate_pipeline_parallel_config,
    )

    pipeline_config = PipelineParallelConfig.from_config(args)
    if pipeline_config.enabled:

        try:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
        except ValueError as exc:
            raise ValueError("WORLD_SIZE must be an integer for pipeline_parallel") from exc
        validate_pipeline_parallel_config(args, world_size=world_size)
        raise RuntimeError(
            "Pipeline-parallel stage planning is available for this model "
            "family, but the 1F1B schedule is not wired into the main trainer "
            "yet. Refusing to fall back to ordinary Accelerate data parallelism."
        )

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"  # backward compatibility

    artist = getattr(args, "artist_filter", None)
    if artist:
        _datasets_base.set_artist_filter(artist)
        slug = artist.lstrip("@")
        args.output_dir = "output/ckpt-artist"
        args.output_name = f"{args.output_name}_{slug}"
        logger.info(
            f"artist_filter active: '{artist}' → output_dir={args.output_dir}, "
            f"output_name={args.output_name}"
        )

    trainer = trainer_factory()
    trainer.train(args)
