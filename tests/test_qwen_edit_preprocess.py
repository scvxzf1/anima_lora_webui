from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from scripts.qwen_image_2_1.preprocess_edit_cache import build_edit_datasets
from scripts.tasks.qwen_edit_preprocess import run_edit_preprocess, uses_qwen_edit
from scripts.tasks.qwen_edit_preprocess import snapshot_edit_captions, split_edit_options
from scripts.tasks import preprocess


def _dataset(tmp_path: Path, *, caption: str = "make the sky blue\n") -> Path:
    target = tmp_path / "target"
    reference = tmp_path / "reference"
    for folder in (target, reference):
        (folder / "sub").mkdir(parents=True)
        Image.new("RGB", (64, 64)).save(folder / "sub" / "one.png")
    (target / "sub" / "one.txt").write_text(caption, encoding="utf-8")
    config = tmp_path / "dataset.toml"
    config.write_text(
        '[general]\ncaption_extension = ".txt"\n'
        '[[datasets]]\nresolution = 256\nbucket_reso_steps = 32\n'
        '[[datasets.subsets]]\n'
        f'image_dir = "{target}"\nreference_image_dir = "{reference}"\n'
        'recursive = true\n',
        encoding="utf-8",
    )
    return config


def test_edit_preprocess_uses_training_infos_and_buckets(tmp_path: Path) -> None:
    config = _dataset(tmp_path)
    groups = build_edit_datasets(str(config))
    info = next(iter(groups[0].image_data.values()))
    assert info.caption == "make the sky blue"
    assert info.reference_image_path == str(tmp_path / "reference" / "sub" / "one.png")
    assert info.bucket_reso == (256, 256)
    assert groups[0].datasets[0].image_to_subset[info.image_key].model_family == "qwen_image_2_1"


def test_edit_preprocess_rejects_missing_pair_and_caption_variants(tmp_path: Path) -> None:
    config = _dataset(tmp_path, caption="first\nsecond\n")
    with pytest.raises(ValueError, match="caption variants"):
        build_edit_datasets(str(config))
    (tmp_path / "target" / "sub" / "one.txt").write_text("first\n", encoding="utf-8")
    (tmp_path / "reference" / "sub" / "one.png").unlink()
    with pytest.raises(ValueError, match="no paired reference"):
        build_edit_datasets(str(config))


def test_edit_task_dispatch_preserves_t2i(tmp_path: Path) -> None:
    config = _dataset(tmp_path)
    overrides = {"dataset_config": str(config)}
    assert uses_qwen_edit("qwen_image_2_1", overrides)
    assert not uses_qwen_edit("anima", overrides)
    calls = []
    run_edit_preprocess(
        overrides, run=calls.append, python="python", path=lambda key, default: default,
        dtype="bfloat16", extra=[],
    )
    assert calls[0][:3] == ["python", "-m", "scripts.qwen_image_2_1.preprocess_edit_cache"]
    assert "--dataset_config" in calls[0]


def test_edit_overwrite_is_not_forwarded_to_resize():
    resize, runtime = split_edit_options(["--overwrite", "--offload=on", "--min_pixels", "0"])
    assert resize == ["--min_pixels", "0"]
    assert runtime == ["--offload", "on"]


def test_edit_caption_snapshot_preserves_custom_suffix_and_relative_path(tmp_path):
    source, resized = tmp_path / "source", tmp_path / "resized"
    (source / "nested").mkdir(parents=True)
    (resized / "nested").mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(resized / "nested" / "one.png")
    (source / "nested" / "one.prompt").write_text("change color\n", encoding="utf-8")
    (source / "excluded.prompt").write_text("excluded", encoding="utf-8")
    snapshot_edit_captions(str(source), str(resized), ".prompt")
    assert (resized / "nested" / "one.prompt").read_text() == "change color\n"
    assert not (resized / "excluded.prompt").exists()


def test_edit_preprocess_reuses_complete_caches_without_loading_models(monkeypatch) -> None:
    from scripts.qwen_image_2_1 import preprocess_edit_cache as edit

    class CompleteGroup:
        image_data = {"one": object()}

        def is_latents_cache_complete(self):
            return True

        def is_text_encoder_outputs_cache_complete(self):
            return True

    monkeypatch.setattr(edit, "build_edit_datasets", lambda _path: [CompleteGroup()])
    edit.cache_edit("unused.toml", vae_path="unused", qwen3_path="unused", device="cpu")


def test_edit_task_without_references_fails_closed(tmp_path: Path, monkeypatch) -> None:
    config = _dataset(tmp_path)
    config.write_text(config.read_text(encoding="utf-8").replace(
        f'reference_image_dir = "{tmp_path / "reference"}"\n', ""
    ), encoding="utf-8")
    overrides = {"model_family": "qwen_image_2_1", "qwen_image_2_1_task": "edit",
                 "dataset_config": str(config)}
    monkeypatch.setattr(preprocess, "_path_overrides_value", lambda: overrides)
    with pytest.raises(ValueError, match="reference_image_dir"):
        preprocess.cmd_preprocess([])
    with pytest.raises(ValueError, match="reference_image_dir"):
        preprocess.cmd_preprocess_te([])
    with pytest.raises(ValueError, match="reference_image_dir"):
        preprocess.cmd_preprocess_vae([])


def test_edit_dispatch_runtime_options_and_rebuild_policy(tmp_path: Path, monkeypatch) -> None:
    config = _dataset(tmp_path)
    overrides = {"model_family": "qwen_image_2_1", "qwen_image_2_1_task": "edit",
                 "dataset_config": str(config), "force_rebuild_preprocess_cache": True}
    calls: list[list[str]] = []
    monkeypatch.setattr(preprocess, "_path_overrides_value", lambda: overrides)
    monkeypatch.setattr(preprocess, "_preprocess_rows", lambda: [{"source_image_dir": str(tmp_path / "target"),
        "resized_image_dir": str(tmp_path / "resized"), "lora_cache_dir": str(tmp_path / "cache")}])
    monkeypatch.setattr(preprocess, "_run_caption_backup", lambda _row: None)
    monkeypatch.setattr(preprocess, "_preprocess_precision_dtype", lambda: "bfloat16")
    monkeypatch.setattr(preprocess, "run", calls.append)
    preprocess.cmd_preprocess(["--device=cpu", "--offload", "off"])
    assert len(calls) == 2
    assert "--no_copy_captions" not in calls[0]
    assert "--device" not in calls[0] and "--device=cpu" not in calls[0]
    assert "--offload" not in calls[0] and "off" not in calls[0]
    assert calls[1][calls[1].index("--device") + 1] == "cpu"
    assert calls[1][calls[1].index("--offload") + 1] == "off"
    assert "--overwrite" in calls[1]


def test_edit_rejects_partial_rebuild_and_individual_tasks(tmp_path: Path, monkeypatch) -> None:
    config = _dataset(tmp_path)
    overrides = {"model_family": "qwen_image_2_1", "qwen_image_2_1_task": "edit",
                 "dataset_config": str(config), "reuse_vae_latents": False}
    calls: list[list[str]] = []
    monkeypatch.setattr(preprocess, "_path_overrides_value", lambda: overrides)
    monkeypatch.setattr(preprocess, "_preprocess_rows", lambda: [{}])
    monkeypatch.setattr(preprocess, "_run_caption_backup", lambda _row: None)
    monkeypatch.setattr(preprocess, "run", calls.append)
    with pytest.raises(ValueError, match="cannot rebuild only VAE or TE"):
        preprocess.cmd_preprocess([])
    with pytest.raises(ValueError, match="full `preprocess` task"):
        preprocess.cmd_preprocess_te([])
    with pytest.raises(ValueError, match="full `preprocess` task"):
        preprocess.cmd_preprocess_vae([])
    assert calls == []
