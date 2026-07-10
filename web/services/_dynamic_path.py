"""Small path-like proxy for runtime-resolved service directories/files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable


class DynamicPath:
    """Resolve a fresh ``Path`` on each access while remaining path-like."""

    def __init__(self, resolver: Callable[[], Path]):
        self._resolver = resolver

    def _path(self) -> Path:
        path = self._resolver()
        return path if isinstance(path, Path) else Path(path)

    def __fspath__(self) -> str:
        return os.fspath(self._path())

    def __str__(self) -> str:
        return str(self._path())

    def __repr__(self) -> str:
        return f"DynamicPath({self._path()!r})"

    def __truediv__(self, key):
        return self._path() / key

    def __rtruediv__(self, key):
        return Path(key) / self._path()

    @property
    def parent(self) -> Path:
        return self._path().parent

    @property
    def name(self) -> str:
        return self._path().name

    def resolve(self, *args, **kwargs) -> Path:
        return self._path().resolve(*args, **kwargs)

    def with_name(self, *args, **kwargs) -> Path:
        return self._path().with_name(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._path(), name)
