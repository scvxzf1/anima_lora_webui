from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from web.services.tagging import tag_dictionary


def _write_dictionary_sqlite(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE tags (name TEXT, cn_name TEXT, post_count INTEGER)"
        )
        connection.executemany(
            "INSERT INTO tags (name, cn_name, post_count) VALUES (?, ?, ?)",
            [
                ("blue_hair", "蓝发", 100),
                ("1girl", "1个女孩", 200),
                ("ignored", "", 300),
            ],
        )
        connection.commit()
    finally:
        connection.close()
    return path.stat().st_size


def test_tag_dictionary_manifest_is_fixed_to_verified_source() -> None:
    assert tag_dictionary.SOURCE_COMMIT == "bc2953723a76e1841e9564297c6812723223ecb0"
    assert tag_dictionary.SOURCE_SIZE == 23_937_024
    assert tag_dictionary.SOURCE_SHA256 == (
        "08671c8ca1f0342baa5e9e6cfd8ab64d9a703165a2c4dbd7212c6115680ef1a8"
    )
    assert tag_dictionary.SOURCE_COMMIT in tag_dictionary.SOURCE_URL
    assert tag_dictionary.SOURCE_URL.startswith("https://raw.githubusercontent.com/")


def test_tag_dictionary_parses_special_path_and_orders_by_usage(tmp_path: Path) -> None:
    source = tmp_path / "dictionary ?# source.sqlite"
    _write_dictionary_sqlite(source)

    entries = tag_dictionary._parse_sqlite(source)

    assert list(entries) == ["1girl", "blue hair"]
    assert entries == {"1girl": "1个女孩", "blue hair": "蓝发"}


def test_tag_dictionary_publish_status_and_bidirectional_translation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "tag-dictionary"
    stage = root / ".download-test.sqlite"
    source_size = _write_dictionary_sqlite(stage)
    monkeypatch.setenv(tag_dictionary.ROOT_ENV, str(root))
    monkeypatch.setattr(tag_dictionary, "SOURCE_SIZE", source_size)
    entries = tag_dictionary._parse_sqlite(stage)
    meta = {
        "source_name": tag_dictionary.SOURCE_NAME,
        "source_commit": tag_dictionary.SOURCE_COMMIT,
        "source_sha256": tag_dictionary.SOURCE_SHA256,
        "entry_count": len(entries),
        "installed_at": 123,
    }

    tag_dictionary._publish_dictionary(stage, entries, meta)
    service = tag_dictionary.TagDictionaryService()

    status = service.status()
    assert status["installed"] is True
    assert status["entry_count"] == 2
    assert not stage.exists()
    assert (root / "source.sqlite").stat().st_size == source_size

    translated = asyncio.run(service.translate(["blue_hair", "unknown"], "zh"))
    assert translated["translations"] == ["蓝发", "unknown"]
    assert translated["matched"] == 1

    reversed_result = asyncio.run(service.translate(["蓝发", "未知"], "en"))
    assert reversed_result["translations"] == ["blue hair", "未知"]
    assert reversed_result["matched"] == 1

    (root / "active.json").write_text("not-json", encoding="utf-8")
    assert service.status()["installed"] is False


def test_tag_dictionary_missing_or_partial_install_is_not_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "tag-dictionary"
    root.mkdir()
    monkeypatch.setenv(tag_dictionary.ROOT_ENV, str(root))
    (root / "active.json").write_text(
        json.dumps({"meta": {}, "entries": {"1girl": "1个女孩"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (root / "meta.json").write_text(
        json.dumps(
            {
                "source_commit": tag_dictionary.SOURCE_COMMIT,
                "source_sha256": tag_dictionary.SOURCE_SHA256,
                "entry_count": 1,
                "installed_at": 123,
            }
        ),
        encoding="utf-8",
    )

    service = tag_dictionary.TagDictionaryService()
    assert service.status()["installed"] is False

    (root / "active.json").unlink()
    with pytest.raises(RuntimeError, match="先下载"):
        asyncio.run(service.translate(["1girl"], "zh"))


@pytest.mark.parametrize(
    ("tags", "target", "message"),
    [
        ([], "zh", "非空数组"),
        (["1girl"], "fr", "只支持 zh 或 en"),
        ([""], "zh", "不能为空"),
        (["tag"] * 501, "zh", "最多翻译 500"),
    ],
)
def test_tag_dictionary_rejects_invalid_translation_payload(tags, target, message) -> None:
    service = tag_dictionary.TagDictionaryService()
    with pytest.raises(ValueError, match=message):
        asyncio.run(service.translate(tags, target))


def test_tag_dictionary_invalid_root_finishes_download_as_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(tag_dictionary.ROOT_ENV, "../outside")
    service = tag_dictionary.TagDictionaryService()
    job = tag_dictionary._DownloadJob(id="invalid-root")

    asyncio.run(service._run_download(job))

    assert job.state == "error"
    assert "不能包含" in job.error
