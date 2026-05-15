"""Web UI launcher."""

from __future__ import annotations

import sys

from scripts.tasks._common import PY, ROOT, run


def cmd_web(extra: list[str]):
    run([PY, "-m", "web", *extra], cwd=str(ROOT))
