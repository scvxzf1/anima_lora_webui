"""Mount pool directories into a run's dataset_cache with fallbacks."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def mount_dir(src: Path, dst: Path) -> str:
    """Link or copy ``src`` onto ``dst``.

    Preference: symlink → copy. Returns ``link_mode`` string.
    File-level hardlink of whole trees is not portable for directories; copy is
    the safe fallback when symlink is unavailable.
    """
    src = Path(src).resolve()
    dst = Path(dst)
    if not src.exists():
        dst.mkdir(parents=True, exist_ok=True)
        return "copy"

    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        elif dst.is_dir():
            shutil.rmtree(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst, target_is_directory=True)
        return "symlink"
    except OSError:
        pass

    shutil.copytree(src, dst, dirs_exist_ok=True)
    return "copy"
