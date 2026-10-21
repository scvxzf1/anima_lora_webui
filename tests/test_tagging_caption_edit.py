from __future__ import annotations

import json
from pathlib import Path

import pytest

from web.services.tagging import caption_edit


def _image(tmp_path: Path, name: str = "a.png") -> Path:
    path = tmp_path / "dataset" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image")
    return path


def test_tag_parsing_accepts_ascii_and_fullwidth_separators_and_newlines() -> None:
    assert caption_edit.split_tags("1girl, solo\nsmile，blue eyes,, 1GIRL") == [
        "1girl",
        "solo",
        "smile",
        "blue eyes",
    ]


def test_merge_keeps_existing_order_then_appends_new_tags() -> None:
    assert caption_edit.merge_tags(["1girl", "solo"], ["Solo", "smile"]) == ["1girl", "solo", "smile"]


def test_trigger_is_prepended_once_case_insensitively() -> None:
    assert caption_edit.apply_trigger("MyStyle", ["1girl", "mystyle"]) == ["MyStyle", "1girl"]
    assert caption_edit.apply_trigger("", ["1girl"]) == ["1girl"]


def test_structured_sidecar_wins_over_plain_text(tmp_path) -> None:
    image = _image(tmp_path)
    assert caption_edit.sidecar_for(image) == (image.with_suffix(".txt"), "text")

    image.with_suffix(".txt").write_text("plain\n", encoding="utf-8")
    assert caption_edit.sidecar_for(image) == (image.with_suffix(".txt"), "text")

    image.with_suffix(".json").write_text('{"tags": ["json"]}', encoding="utf-8")
    assert caption_edit.sidecar_for(image) == (image.with_suffix(".json"), "json")


def test_effective_caption_requires_real_content(tmp_path) -> None:
    image = _image(tmp_path)
    assert caption_edit.has_effective_caption(image) is False

    image.with_suffix(".txt").write_text("   \n", encoding="utf-8")
    assert caption_edit.has_effective_caption(image) is False

    image.with_suffix(".txt").write_text("1girl\n", encoding="utf-8")
    assert caption_edit.has_effective_caption(image) is True


def test_trigger_only_structured_caption_counts_as_effective(tmp_path) -> None:
    image = _image(tmp_path)
    image.with_suffix(".json").write_text('{"meta": {"trigger": "MyStyle"}}', encoding="utf-8")

    assert caption_edit.has_effective_caption(image) is True
    assert caption_edit.read_tags(image) == []


def test_damaged_structured_caption_is_not_effective_but_cannot_be_overwritten(tmp_path) -> None:
    image = _image(tmp_path)
    image.with_suffix(".json").write_text("{not json", encoding="utf-8")

    assert caption_edit.has_effective_caption(image) is False
    with pytest.raises(caption_edit.CorruptCaptionError):
        caption_edit.write_tags(image, ["1girl"])
    assert image.with_suffix(".json").read_text(encoding="utf-8") == "{not json"


def test_structured_write_keeps_unknown_fields_and_meta(tmp_path) -> None:
    image = _image(tmp_path)
    image.with_suffix(".json").write_text(
        json.dumps(
            {
                "sample": "keep",
                "meta": {"source": "api"},
                "quality": ["best"],
                "ai_output": {"tags": ["old"]},
            }
        ),
        encoding="utf-8",
    )

    written = caption_edit.write_tags(image, ["1girl", "solo"], trigger="MyStyle")
    payload = json.loads(image.with_suffix(".json").read_text(encoding="utf-8"))

    assert written["kind"] == "json"
    assert payload["tags"] == ["MyStyle", "1girl", "solo"]
    assert payload["sample"] == "keep"
    assert payload["meta"] == {"source": "api", "trigger": "MyStyle"}
    for absorbed in ("quality", "ai_output"):
        assert absorbed not in payload


def test_structured_write_keeps_natural_language_unless_already_present(tmp_path) -> None:
    image = _image(tmp_path)
    image.with_suffix(".json").write_text(
        json.dumps({"tags": ["1girl"], "nl": "a girl standing"}), encoding="utf-8"
    )

    kept = caption_edit.write_tags(image, ["1girl", "solo"])
    assert kept["text"] == "1girl, solo. a girl standing"
    assert json.loads(image.with_suffix(".json").read_text(encoding="utf-8"))["nl"] == "a girl standing"

    absorbed = caption_edit.write_tags(image, ["1girl", "a girl standing at night"])
    assert absorbed["text"] == "1girl, a girl standing at night"
    assert "nl" not in json.loads(image.with_suffix(".json").read_text(encoding="utf-8"))


def test_empty_tags_still_persist_the_sidecar(tmp_path) -> None:
    image = _image(tmp_path)
    caption_edit.write_tags(image, ["1girl"])
    assert image.with_suffix(".txt").exists()

    caption_edit.write_tags(image, [])
    assert image.with_suffix(".txt").exists()
    assert image.with_suffix(".txt").read_text(encoding="utf-8") == ""
    assert caption_edit.has_effective_caption(image) is False


def test_multi_tag_trigger_is_prepended_once() -> None:
    trigger = "masterpiece, best quality"
    assert caption_edit.apply_trigger(trigger, ["1girl"]) == ["masterpiece", "best quality", "1girl"]
    assert caption_edit.apply_trigger(trigger, ["best quality", "1girl"]) == [
        "masterpiece",
        "best quality",
        "1girl",
    ]
    assert caption_edit.apply_trigger("", ["1girl"]) == ["1girl"]


def test_configured_text_suffix_drives_detection_and_reads(tmp_path) -> None:
    image = _image(tmp_path)
    image.with_suffix(".tags").write_text("keep, mine\n", encoding="utf-8")

    # The default suffix must not see it, and the configured one must.
    assert caption_edit.has_effective_caption(image) is False
    assert caption_edit.has_effective_caption(image, text_suffix=".tags") is True
    assert caption_edit.read_tags(image, text_suffix=".tags") == ["keep", "mine"]


def test_configured_text_suffix_is_used_for_new_sidecars(tmp_path) -> None:
    image = _image(tmp_path)

    written = caption_edit.write_tags(image, ["1girl"], text_suffix=".caption")

    assert written["path"] == image.with_suffix(".caption")
    assert image.with_suffix(".caption").read_text(encoding="utf-8") == "1girl\n"
