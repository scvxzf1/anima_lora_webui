"""Windows-quiet subprocess helpers."""

from __future__ import annotations

import subprocess
import sys


def no_window_kwargs() -> dict:
    """Return subprocess kwargs that hide short-lived console windows on Windows."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}
