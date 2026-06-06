"""Compatibility front door for training CLI argparse declarations.

Upstream moved the argparse surface from ``library.training.cli_args`` into
``library.config.cli_args``. This branch still has local training/WebUI
diagnostic flags in the old module, so this file intentionally re-exports that
complete implementation instead of keeping a divergent copy.
"""

from library.training.cli_args import *  # noqa: F401,F403
