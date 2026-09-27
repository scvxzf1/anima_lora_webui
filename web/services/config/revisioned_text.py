"""Optimistic concurrency for editable config text files."""

from __future__ import annotations

import fcntl
import hashlib
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from web.services.atomic_io import atomic_write_text


class RevisionConflictError(ValueError):
    pass


def text_revision(content: str, *, exists: bool = True) -> str:
    marker = b"present\0" if exists else b"missing\0"
    return hashlib.sha256(marker + content.encode("utf-8")).hexdigest()


def file_revision(path: Path) -> str:
    if not path.exists():
        return text_revision("", exists=False)
    return text_revision(path.read_text(encoding="utf-8"))


@contextmanager
def locked_text_file(path: Path) -> Iterator[None]:
    # Lock a stable name, not the target inode replaced by atomic_write_text.
    lock_dir = Path(tempfile.gettempdir()) / f"anima-config-locks-{os.getuid()}"
    lock_dir.mkdir(mode=0o700, exist_ok=True)
    key = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()
    fd = os.open(lock_dir / key, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def check_revision(path: Path, expected_revision: str | None) -> None:
    if expected_revision is not None and file_revision(path) != expected_revision:
        raise RevisionConflictError("文件已在其他位置修改，请核对新版本后再保存")


def write_text(
    path: Path, content: str, expected_revision: str | None = None, *, exclusive: bool = False
) -> str:
    with locked_text_file(path):
        if exclusive and path.exists():
            raise FileExistsError(path)
        check_revision(path, expected_revision)
        atomic_write_text(path, content)
        return text_revision(content)
