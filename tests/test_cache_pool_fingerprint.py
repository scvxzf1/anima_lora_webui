"""Tests for cache_pool light/content fingerprints."""

from __future__ import annotations

from pathlib import Path

from library.cache_pool.fingerprint import (
    build_preprocess_signature,
    compute_fingerprint,
    scan_input_inventory,
)


def _touch_image(path: Path, data: bytes = b"\x89PNG\r\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_light_fingerprint_stable_for_same_inputs(tmp_path: Path) -> None:
    src = tmp_path / "src"
    img = src / "a.png"
    cap = src / "a.txt"
    _touch_image(img, b"img-a")
    cap.write_text("1girl, smile", encoding="utf-8")

    inv = scan_input_inventory(src, recursive=True, path_pattern=None, caption_mode="txt")
    sig = build_preprocess_signature(
        {
            "resolution": 1024,
            "drop_lowres_images": True,
            "min_pixels": 500000,
            "model_family": "anima",
        },
    )
    fp1 = compute_fingerprint(
        mode="light",
        source_dir=src,
        inventory=inv,
        preprocess_signature=sig,
        normalized_source=str(src.resolve()),
    )
    inv2 = scan_input_inventory(src, recursive=True, path_pattern=None, caption_mode="txt")
    fp2 = compute_fingerprint(
        mode="light",
        source_dir=src,
        inventory=inv2,
        preprocess_signature=sig,
        normalized_source=str(src.resolve()),
    )
    assert fp1 == fp2
    assert len(fp1) >= 16


def test_light_fingerprint_changes_when_caption_mtime_or_bytes_change(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _touch_image(src / "a.png", b"img-a")
    cap = src / "a.txt"
    cap.write_text("1girl", encoding="utf-8")
    sig = build_preprocess_signature({"resolution": 1024, "model_family": "anima"})
    inv1 = scan_input_inventory(src, recursive=True, path_pattern=None, caption_mode="txt")
    fp1 = compute_fingerprint(
        mode="light",
        source_dir=src,
        inventory=inv1,
        preprocess_signature=sig,
        normalized_source=str(src.resolve()),
    )
    cap.write_text("1girl, smile", encoding="utf-8")
    inv2 = scan_input_inventory(src, recursive=True, path_pattern=None, caption_mode="txt")
    fp2 = compute_fingerprint(
        mode="light",
        source_dir=src,
        inventory=inv2,
        preprocess_signature=sig,
        normalized_source=str(src.resolve()),
    )
    assert fp1 != fp2


def test_content_mode_hashes_file_bytes_not_just_mtime(tmp_path: Path) -> None:
    src = tmp_path / "src"
    img = src / "a.png"
    _touch_image(img, b"img-a")
    (src / "a.txt").write_text("tags", encoding="utf-8")
    sig = build_preprocess_signature({"resolution": 1024, "model_family": "anima"})
    inv1 = scan_input_inventory(src, recursive=True, path_pattern=None, caption_mode="txt")
    fp1 = compute_fingerprint(
        mode="content",
        source_dir=src,
        inventory=inv1,
        preprocess_signature=sig,
        normalized_source=str(src.resolve()),
    )
    img.write_bytes(b"img-b")
    inv2 = scan_input_inventory(src, recursive=True, path_pattern=None, caption_mode="txt")
    fp2 = compute_fingerprint(
        mode="content",
        source_dir=src,
        inventory=inv2,
        preprocess_signature=sig,
        normalized_source=str(src.resolve()),
    )
    assert fp1 != fp2


def test_adapter_method_not_in_signature() -> None:
    a = build_preprocess_signature(
        {"resolution": 1024, "network_module": "networks.lora", "learning_rate": 1e-4}
    )
    b = build_preprocess_signature(
        {"resolution": 1024, "network_module": "networks.lokr", "learning_rate": 1e-3}
    )
    assert a == b


def test_caption_cache_content_settings_are_in_signature() -> None:
    base = {
        "resolution": 1024,
        "caption_source_mode": "auto",
        "caption_shuffle_variants": 4,
        "caption_tag_dropout_rate": 0.1,
    }
    signature = build_preprocess_signature(base)

    assert signature["schema_version"] == "2"
    assert signature["caption_source_mode"] == "auto"
    assert signature["caption_shuffle_variants"] == 4
    assert signature["caption_tag_dropout_rate"] == 0.1
    assert signature != build_preprocess_signature({**base, "caption_shuffle_variants": 2})
    assert signature != build_preprocess_signature({**base, "caption_tag_dropout_rate": 0.25})


def test_runtime_caption_dropout_does_not_change_preprocess_signature() -> None:
    base = {"resolution": 1024, "caption_shuffle_variants": 4}

    assert build_preprocess_signature(
        {**base, "caption_dropout_rate": 0.0}
    ) == build_preprocess_signature({**base, "caption_dropout_rate": 0.5})


def test_captions_json_included_when_present(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _touch_image(src / "a.png", b"img-a")
    captions = src / "captions.json"
    captions.write_text('{"a.png": ["1girl"]}', encoding="utf-8")
    inv = scan_input_inventory(
        src, recursive=True, path_pattern=None, caption_mode="captions_json"
    )
    kinds = {item["kind"] for item in inv}
    assert "image" in kinds
    assert "caption_index" in kinds


def test_caption_inventory_tracks_all_supported_source_layouts(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _touch_image(src / "nested" / "a.png", b"img-a")
    (src / "nested" / "a.caption").write_text("text", encoding="utf-8")
    (src / "nested" / "a.json").write_text('{"tags": ["json"]}', encoding="utf-8")
    (src / "nested" / "captions.json").write_text(
        '{"a.png": ["indexed"]}', encoding="utf-8"
    )

    inventory = scan_input_inventory(
        src,
        recursive=True,
        path_pattern=None,
        caption_mode="auto",
        caption_extension=".caption",
    )

    caption_paths = {
        item["relpath"] for item in inventory if item["kind"] != "image"
    }
    assert caption_paths == {
        "nested/a.caption",
        "nested/a.json",
        "nested/captions.json",
    }
