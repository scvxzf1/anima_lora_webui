"""Explicit experimental launchers for matched Anima parallelism probes."""

from __future__ import annotations

import argparse

from scripts.tasks._common import PY, run


def cmd_anima_pipeline_bench(extra):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", default="pp")
    args, _ = parser.parse_known_args(extra)
    module = "scripts.experiments.anima_pipeline_bench"
    if args.mode == "single" or "--help" in extra or "-h" in extra:
        run([PY, "-m", module, *extra])
    else:
        run(
            [
                PY,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                "-m",
                module,
                *extra,
            ]
        )
