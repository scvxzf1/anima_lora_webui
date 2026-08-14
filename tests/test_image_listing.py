from __future__ import annotations

import os
from pathlib import Path

import pytest

from web.services.image_listing import select_recent_files


def _write_file(path: Path, *, mtime: float) -> None:
    path.write_bytes(path.name.encode("utf-8"))
    os.utime(path, (mtime, mtime))


def test_select_recent_files_returns_bounded_newest_and_full_total(tmp_path) -> None:
    for index in range(6):
        _write_file(tmp_path / f"image-{index}.png", mtime=100 + index)
    _write_file(tmp_path / "ignored.txt", mtime=999)

    selected, total = select_recent_files(
        tmp_path,
        suffixes={".png"},
        limit=3,
    )

    assert total == 6
    assert [path.name for path, _ in selected] == ["image-5.png", "image-4.png", "image-3.png"]


def test_select_recent_files_counts_without_retaining_candidates(tmp_path) -> None:
    for index in range(4):
        _write_file(tmp_path / f"image-{index}.png", mtime=100 + index)

    selected, total = select_recent_files(
        tmp_path,
        suffixes={".png"},
        limit=0,
        min_mtime=102,
    )

    assert selected == []
    assert total == 2


def test_select_recent_files_skips_symlinks_directories_and_stat_failures(
    tmp_path,
    monkeypatch,
) -> None:
    regular = tmp_path / "regular.png"
    disappearing = tmp_path / "disappearing.png"
    _write_file(regular, mtime=100)
    _write_file(disappearing, mtime=200)
    (tmp_path / "folder.png").mkdir()

    link = tmp_path / "link.png"
    try:
        link.symlink_to(regular)
    except OSError:
        pytest.skip("platform does not support symlinks")

    original_lstat = Path.lstat

    def flaky_lstat(path: Path):
        if path == disappearing:
            raise FileNotFoundError(path)
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    selected, total = select_recent_files(
        tmp_path,
        suffixes={".png"},
        limit=10,
    )

    assert total == 1
    assert [path.name for path, _ in selected] == ["regular.png"]
