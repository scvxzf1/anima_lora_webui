"""Web runtime trigger-clone materialization tests"""

from __future__ import annotations

from types import SimpleNamespace

from library.training.stage_schedule import active_subset_indices_for_step

from tests import training_resume_test_support as _training_resume_support

globals().update(
    {
        name: value
        for name, value in vars(_training_resume_support).items()
        if name == "Path" or not name.startswith("__")
    }
)

def test_web_runtime_trigger_clone_materializes_extra_subset(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)

    source_root = tmp_path / "image_dataset" / "mixed"
    nested = source_root / "nested"
    nested.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(source_root / "top.png")
    Image.new("RGB", (8, 8), color=(60, 40, 20)).save(nested / "sample.png")
    captions = {
        "top.png": ["1girl, solo, silver hair, purple eyes, white dress"],
        "nested/sample.png": ["1girl, solo, forest, moonlight"],
    }
    (source_root / "captions.json").write_text(json.dumps(captions, ensure_ascii=False), encoding="utf-8")
    second_source = tmp_path / "image_dataset" / "second"
    second_source.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(10, 80, 40)).save(second_source / "second.png")
    (second_source / "second.txt").write_text("second character", encoding="utf-8")

    (tmp_path / "configs" / "datasets" / "522.toml").write_text(
        "\n".join(
            [
                "stage_schedule_enabled = true",
                'stage_schedule = [{name = "mixed", subset_index = 0, start_pct = 0.0, end_pct = 0.5}, {name = "second", subset_index = 1, start_pct = 0.5, end_pct = 1.0}]',
                "",
                "[[datasets]]",
                'caption_source_mode = "captions_json"',
                "",
                "[[datasets.subsets]]",
                'image_dir = "old/mixed_resized"',
                'cache_dir = "old/mixed_lora"',
                "recursive = true",
                'custom_attributes = {source_dir = "image_dataset/mixed", nl_tag_mix = {enabled = true, tag_ratio = 1.0}, trigger_clone = {enabled = true, prompt = "my_character", num_repeats = 3}}',
                "num_repeats = 5",
                "",
                "[[datasets]]",
                "",
                "[[datasets.subsets]]",
                'image_dir = "old/second_resized"',
                'cache_dir = "old/second_lora"',
                'custom_attributes = {source_dir = "image_dataset/second"}',
                "num_repeats = 2",
            ]
        ),
        encoding="utf-8",
    )

    class FixedDatetime:
        @classmethod
        def now(cls):
            return datetime(2026, 5, 23, 12, 8, 0)

        @classmethod
        def fromtimestamp(cls, value):
            return datetime.fromtimestamp(value)

    monkeypatch.setattr(training_service, "datetime", FixedDatetime)

    runtime = training_service._prepare_web_runtime_config(
        "522",
        "default",
        "imported",
        source_config_file="configs/imported/522.toml",
    )

    run_dir = tmp_path / "output" / "runs" / "522-20260523-120800"
    clone_source = run_dir / "dataset_cache" / "dataset-01" / "trigger-clone-source"
    assert (clone_source / "top.png").is_file()
    assert (clone_source / "nested" / "sample.png").is_file()
    materialized = json.loads((clone_source / "captions.json").read_text(encoding="utf-8"))
    assert materialized == {
        "nested/sample.png": ["my_character"],
        "top.png": ["my_character"],
    }
    manifest = json.loads((clone_source / "trigger-clone-results.json").read_text(encoding="utf-8"))
    assert manifest["source_dir"].endswith("dataset-01/source")
    assert manifest["num_repeats"] == 3
    assert manifest["total"] == 2

    dataset_cfg = toml.loads((run_dir / "dataset.runtime.toml").read_text(encoding="utf-8"))
    assert len(dataset_cfg["datasets"]) == 3
    original_subset = dataset_cfg["datasets"][0]["subsets"][0]
    clone_dataset = dataset_cfg["datasets"][1]
    clone_subset = clone_dataset["subsets"][0]
    assert original_subset["num_repeats"] == 5
    assert clone_subset["num_repeats"] == 3
    assert clone_subset["image_dir"].endswith("dataset-01/trigger-clone-resized")
    assert clone_subset["cache_dir"].endswith("dataset-01/trigger-clone-lora")
    assert clone_subset["custom_attributes"]["source_dir"].endswith("dataset-01/trigger-clone-source")
    assert clone_dataset["caption_source_mode"] == "captions_json"
    assert clone_dataset["prefer_json_caption"] is False
    assert "trigger_clone" not in original_subset["custom_attributes"]
    assert runtime["dataset_dirs"][1]["source_dir"].endswith("dataset-01/trigger-clone-source")
    assert runtime["dataset_dirs"][2]["source_dir"] == "image_dataset/second"

    runtime_cfg = toml.loads((run_dir / "config.runtime.toml").read_text(encoding="utf-8"))
    assert runtime_cfg["stage_schedule_target_groups"] == [[0, 1], [2]]
    runtime_cfg["max_train_steps"] = 100
    args = SimpleNamespace(**runtime_cfg)
    assert active_subset_indices_for_step(args, 0) == {0, 1}
    assert active_subset_indices_for_step(args, 50) == {2}

def test_clone_runtime_dataset_rows_preserves_trigger_clone_materialized_dirs(tmp_path):
    old_group = tmp_path / "old" / "dataset_cache" / "dataset-01"
    source = old_group / "trigger-clone-source"
    resized = old_group / "trigger-clone-resized"
    lora = old_group / "trigger-clone-lora"
    for path in (source, resized, lora):
        path.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(source / "sample.png")
    (source / "captions.json").write_text(json.dumps({"sample.png": ["my_character"]}), encoding="utf-8")
    (resized / "sample.png").write_bytes(b"resized")
    (lora / "sample_anima_te.safetensors").write_bytes(b"cache")

    cloned = training_service._clone_runtime_dataset_rows(
        [{
            "source_dir": source.as_posix(),
            "image_dir": resized.as_posix(),
            "cache_dir": lora.as_posix(),
            "num_repeats": 3,
            "settings": {"caption_source_mode": "captions_json"},
        }],
        tmp_path / "new" / "dataset_cache",
        copy_existing=True,
    )

    new_group = tmp_path / "new" / "dataset_cache" / "dataset-01"
    assert (new_group / "trigger-clone-source" / "captions.json").is_file()
    assert (new_group / "trigger-clone-resized" / "sample.png").is_file()
    assert (new_group / "trigger-clone-lora" / "sample_anima_te.safetensors").is_file()
    assert cloned[0]["source_dir"].endswith("dataset-01/trigger-clone-source")
    assert cloned[0]["image_dir"].endswith("dataset-01/trigger-clone-resized")
    assert cloned[0]["cache_dir"].endswith("dataset-01/trigger-clone-lora")
