"""Web UI launcher."""

from __future__ import annotations

import sys
import shutil

from scripts.tasks._common import PY, ROOT, run


def cmd_web(extra: list[str]):
    run([PY, "-m", "web", *extra], cwd=str(ROOT))


def _frontend_command(command: str, extra: list[str]):
    pnpm = shutil.which("pnpm")
    if not pnpm:
        raise SystemExit("pnpm is required; see web/frontend-next/README.md")
    run([pnpm, "--dir", str(ROOT / "web" / "frontend-next"), command, *extra], cwd=str(ROOT))


def cmd_frontend_build(extra: list[str]):
    _frontend_command("build", extra)


def cmd_frontend_check(extra: list[str]):
    _frontend_command("typecheck", [])
    _frontend_command("test", extra)


def cmd_frontend_e2e(extra: list[str]):
    _frontend_command("test:e2e", extra)
