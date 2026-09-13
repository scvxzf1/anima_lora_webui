"""Bounded, stat-keyed indexes for read-only history logs."""

from array import array
from bisect import bisect_left, bisect_right
from functools import lru_cache
import json
from pathlib import Path

STRIDE = 256


def fingerprint(path: Path) -> tuple:
    stat = path.stat()
    return str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@lru_cache(maxsize=8)
def index_for(key: tuple) -> tuple[array, int]:
    offsets = array("Q")
    total = 0
    with open(key[0], "rb") as handle:
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            if not raw.strip():
                continue
            if total % STRIDE == 0:
                offsets.append(offset)
            total += 1
    return offsets, total


def records(key: tuple, start: int, stop: int):
    offsets, total = index_for(key)
    if start >= total:
        return
    block = start // STRIDE
    index = block * STRIDE
    with open(key[0], "rb") as handle:
        handle.seek(offsets[block])
        for raw in handle:
            if not raw.strip():
                continue
            if index >= stop:
                break
            if index >= start:
                try:
                    value = json.loads(raw.decode("utf-8", errors="replace"))
                except (ValueError, UnicodeError):
                    value = None
                if isinstance(value, dict):
                    yield index, value
            index += 1


@lru_cache(maxsize=4)
def matches_for(key: tuple, query: str) -> array:
    total = index_for(key)[1]
    return array("Q", (index for index, value in records(key, 0, total)
                       if query in str(value.get("line", value.get("message", value.get("text", value)))).casefold()))


def search(key: tuple, query: str, cursor: int, direction: str):
    matches = matches_for(key, query)
    if not matches:
        return None
    ordinal = (bisect_left(matches, cursor) % len(matches) if direction == "forward"
               else (bisect_right(matches, cursor) - 1) % len(matches))
    index = matches[ordinal]
    record = next(records(key, index, index + 1), (index, None))[1]
    return index, record, ordinal + 1, len(matches)
