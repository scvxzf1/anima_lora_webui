"""Bounded, race-tolerant file selection for WebUI image galleries."""

from __future__ import annotations

from heapq import heappush, heapreplace
import os
from pathlib import Path
import stat as stat_module
from typing import Collection


RecentFile = tuple[Path, os.stat_result]


def select_recent_files(
    directory: Path,
    *,
    suffixes: Collection[str],
    limit: int,
    min_mtime: float | None = None,
    offset: int = 0,
    before_key: tuple[float, str, str] | None = None,
) -> tuple[list[RecentFile], int]:
    """Return one newest-first page, retaining at most ``offset + limit`` candidates.

    Directory scans still count every eligible file for API ``total`` fields,
    but expensive image metadata work can be restricted to the returned list.
    Files that disappear or become unreadable during the scan are skipped.
    """

    try:
        if not directory.is_dir():
            return [], 0
    except OSError:
        return [], 0

    normalized_suffixes = {str(value).lower() for value in suffixes}
    page_limit = max(0, int(limit))
    offset = max(0, int(offset))
    bounded_limit = offset + page_limit if page_limit else 0
    heap: list[tuple[tuple[float, str, str], Path, os.stat_result]] = []
    total = 0

    try:
        for path in directory.iterdir():
            if path.suffix.lower() not in normalized_suffixes:
                continue
            try:
                stat_result = path.lstat()
            except OSError:
                continue
            if not stat_module.S_ISREG(stat_result.st_mode):
                continue
            if min_mtime is not None and stat_result.st_mtime < min_mtime:
                continue

            total += 1
            if bounded_limit == 0:
                continue
            sort_key = (float(stat_result.st_mtime), path.name, path.as_posix())
            if before_key is not None and sort_key >= before_key:
                continue
            item = (sort_key, path, stat_result)
            if len(heap) < bounded_limit:
                heappush(heap, item)
            elif sort_key > heap[0][0]:
                heapreplace(heap, item)
    except OSError:
        # Iterators may fail after yielding some entries when a mounted or
        # external directory changes underneath the request.
        pass

    selected = sorted(heap, key=lambda item: item[0], reverse=True)
    return [(path, stat_result) for _, path, stat_result in selected[offset:offset + page_limit]], total
