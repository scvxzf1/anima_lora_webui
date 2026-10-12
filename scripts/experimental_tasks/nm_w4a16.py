"""Explicit static-plan / future GPU-probe launcher."""

from scripts.tasks._common import PY, run


def cmd_nm_w4a16(extra):
    run([PY, "-m", "bench.nm_w4a16", *extra])
