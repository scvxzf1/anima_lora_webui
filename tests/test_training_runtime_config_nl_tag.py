"""Web runtime nl-tag mix materialization tests"""

from __future__ import annotations

from tests import training_resume_test_support as _training_resume_support

globals().update(
    {
        name: value
        for name, value in vars(_training_resume_support).items()
        if name == "Path" or not name.startswith("__")
    }
)

def test_web_runtime_config_materializes_nl_tag_mix_source(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)

    source_root = tmp_path / "image_dataset" / "mixed"
    source_root.mkdir(parents=True)
    for idx in range(6):
        name = f"nl_named_tag_caption_{idx:02d}.png"
        Image.new("RGB", (8, 8), color=(idx, 20, 40)).save(source_root / name)
        (source_root / f"nl_named_tag_caption_{idx:02d}.txt").write_text(
            "1girl, solo, silver hair, purple eyes, white dress, standing, forest, moonlight\n",
            encoding="utf-8",
        )
    Image.new("RGB", (8, 8), color=(60, 20, 40)).save(source_root / "ambiguous_caption.png")
    (source_root / "ambiguous_caption.txt").write_text("quiet portrait\n", encoding="utf-8")
    for idx in range(3):
        name = f"tag_named_nl_caption_{idx:02d}.png"
        Image.new("RGB", (8, 8), color=(40, 20, idx)).save(source_root / name)
        (source_root / f"tag_named_nl_caption_{idx:02d}.txt").write_text(
            "A high-quality anime-style illustration of a calm original female character. "
            "She stands in a moonlit fantasy forest with soft luminous lighting and delicate painterly shading.\n",
            encoding="utf-8",
        )

    (tmp_path / "configs" / "datasets" / "522.toml").write_text(
        "\n".join(
            [
                "[[datasets]]",
                "resolution = 768",
                "",
                "[[datasets.subsets]]",
                'image_dir = "old/mixed_resized"',
                'cache_dir = "old/mixed_lora"',
                'custom_attributes = {source_dir = "image_dataset/mixed", nl_tag_mix = {enabled = true, tag_ratio = 0.7}}',
                "num_repeats = 2",
            ]
        ),
        encoding="utf-8",
    )

    class FixedDatetime:
        @classmethod
        def now(cls):
            return datetime(2026, 5, 23, 12, 0, 0)

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

    run_dir = tmp_path / "output" / "runs" / "522-20260523-120000"
    mixed_source = run_dir / "dataset_cache" / "dataset-01" / "source"
    manifest = json.loads((mixed_source / "results.json").read_text(encoding="utf-8"))
    assert runtime["data_dirs"]["source_image_dir"] == "output/runs/522-20260523-120000/dataset_cache/dataset-01/source"
    assert manifest["tag_ratio"] == 0.7
    assert manifest["classification_method"] == "caption_text_v1"
    assert manifest["available_tag_count"] == 7
    assert manifest["available_nl_count"] == 3
    assert manifest["actual_tag_count"] == 7
    assert manifest["actual_nl_count"] == 3
    assert manifest["total"] == 10
    assert len(list(mixed_source.glob("*.png"))) == 10
    by_stem = {item["stem"]: item for item in manifest["items"]}
    assert by_stem["nl_named_tag_caption_00"]["source"] == "tag"
    assert by_stem["tag_named_nl_caption_00"]["source"] == "nl"
    assert by_stem["ambiguous_caption"]["source"] == "tag"
    assert by_stem["ambiguous_caption"]["classification"]["reason"] == "ambiguous_caption_default_tag"
    assert (mixed_source / "nl_named_tag_caption_00.txt").read_text(encoding="utf-8").startswith("1girl")
    assert (mixed_source / "tag_named_nl_caption_00.txt").read_text(encoding="utf-8").startswith("A high-quality")

    runtime_cfg = toml.loads((run_dir / "config.runtime.toml").read_text(encoding="utf-8"))
    assert runtime_cfg["source_image_dir"].endswith("dataset-01/source")
    dataset_cfg = toml.loads((run_dir / "dataset.runtime.toml").read_text(encoding="utf-8"))
    attrs = dataset_cfg["datasets"][0]["subsets"][0]["custom_attributes"]
    assert attrs["source_dir"].endswith("dataset-01/source")
    assert "nl_tag_mix" not in attrs

def test_web_runtime_nl_tag_mix_preserves_captions_json_source(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)

    source_root = tmp_path / "image_dataset" / "mixed"
    source_root.mkdir(parents=True)
    captions = {}
    for idx in range(2):
        name = f"sample_{idx:02d}.png"
        Image.new("RGB", (8, 8), color=(idx, 20, 40)).save(source_root / name)
        captions[name] = [
            "1girl, solo, silver hair, purple eyes, white dress, standing",
            "1girl, solo, forest, moonlight",
        ]
    (source_root / "captions.json").write_text(json.dumps(captions, ensure_ascii=False), encoding="utf-8")

    (tmp_path / "configs" / "datasets" / "522.toml").write_text(
        "\n".join(
            [
                "[[datasets]]",
                'caption_source_mode = "captions_json"',
                "",
                "[[datasets.subsets]]",
                'image_dir = "old/mixed_resized"',
                'cache_dir = "old/mixed_lora"',
                'custom_attributes = {source_dir = "image_dataset/mixed", nl_tag_mix = {enabled = true, tag_ratio = 1.0}}',
                "num_repeats = 1",
            ]
        ),
        encoding="utf-8",
    )

    class FixedDatetime:
        @classmethod
        def now(cls):
            return datetime(2026, 5, 23, 12, 5, 0)

        @classmethod
        def fromtimestamp(cls, value):
            return datetime.fromtimestamp(value)

    monkeypatch.setattr(training_service, "datetime", FixedDatetime)

    training_service._prepare_web_runtime_config(
        "522",
        "default",
        "imported",
        source_config_file="configs/imported/522.toml",
    )

    run_dir = tmp_path / "output" / "runs" / "522-20260523-120500"
    mixed_source = run_dir / "dataset_cache" / "dataset-01" / "source"
    materialized = json.loads((mixed_source / "captions.json").read_text(encoding="utf-8"))
    manifest = json.loads((mixed_source / "results.json").read_text(encoding="utf-8"))

    assert sorted(materialized) == ["sample_00.png", "sample_01.png"]
    assert materialized["sample_00.png"] == captions["sample_00.png"]
    assert len(list(mixed_source.glob("*.png"))) == 2
    assert not list(mixed_source.glob("*.txt"))
    assert manifest["caption_source_mode"] == "captions_json"
    assert manifest["actual_tag_count"] == 2
    assert all(item["caption_source_mode"] == "captions_json" for item in manifest["items"])
    assert all(item["captions"][0].endswith("captions.json") for item in manifest["items"])

def test_web_runtime_nl_tag_mix_reweights_captions_json_entries(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)

    source_root = tmp_path / "image_dataset" / "mixed"
    source_root.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(source_root / "sample.png")
    tag_caption = "1girl, solo, black hair, pink eyes, school uniform, grass"
    nl_caption_a = (
        "A girl lies on green grass. She looks toward the viewer while holding a phone. "
        "The image uses a direct overhead composition."
    )
    nl_caption_b = (
        "The scene shows a relaxed schoolgirl outdoors. Soft daylight falls across "
        "her uniform and the surrounding field."
    )
    (source_root / "captions.json").write_text(
        json.dumps({"sample.png": [tag_caption, nl_caption_a, nl_caption_b]}, ensure_ascii=False),
        encoding="utf-8",
    )

    (tmp_path / "configs" / "datasets" / "522.toml").write_text(
        "\n".join(
            [
                "[[datasets]]",
                'caption_source_mode = "captions_json"',
                "",
                "[[datasets.subsets]]",
                'image_dir = "old/mixed_resized"',
                'cache_dir = "old/mixed_lora"',
                'custom_attributes = {source_dir = "image_dataset/mixed", nl_tag_mix = {enabled = true, tag_ratio = 0.7}}',
                "num_repeats = 1",
            ]
        ),
        encoding="utf-8",
    )

    class FixedDatetime:
        @classmethod
        def now(cls):
            return datetime(2026, 5, 23, 12, 7, 0)

        @classmethod
        def fromtimestamp(cls, value):
            return datetime.fromtimestamp(value)

    monkeypatch.setattr(training_service, "datetime", FixedDatetime)

    training_service._prepare_web_runtime_config(
        "522",
        "default",
        "imported",
        source_config_file="configs/imported/522.toml",
    )

    run_dir = tmp_path / "output" / "runs" / "522-20260523-120700"
    mixed_source = run_dir / "dataset_cache" / "dataset-01" / "source"
    materialized = json.loads((mixed_source / "captions.json").read_text(encoding="utf-8"))
    manifest = json.loads((mixed_source / "results.json").read_text(encoding="utf-8"))

    assert materialized["sample.png"] == [tag_caption, tag_caption, nl_caption_a]
    assert manifest["available_tag_caption_count"] == 1
    assert manifest["available_nl_caption_count"] == 2
    assert manifest["actual_tag_caption_count"] == 2
    assert manifest["actual_nl_caption_count"] == 1
    assert manifest["items"][0]["selected_caption_indices"] == [0, 0, 1]
    assert manifest["items"][0]["actual_caption_counts"] == {"tag": 2, "nl": 1}

def test_web_runtime_nl_tag_mix_preserves_recursive_captions_json_source(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)

    source_root = tmp_path / "image_dataset" / "mixed"
    nested = source_root / "nested"
    nested.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(nested / "sample.png")
    captions = {
        "nested/sample.png": [
            "1girl, solo, silver hair, purple eyes, white dress, standing",
            "1girl, solo, forest, moonlight",
        ]
    }
    (source_root / "captions.json").write_text(json.dumps(captions, ensure_ascii=False), encoding="utf-8")

    (tmp_path / "configs" / "datasets" / "522.toml").write_text(
        "\n".join(
            [
                "[[datasets]]",
                'caption_source_mode = "captions_json"',
                "",
                "[[datasets.subsets]]",
                'image_dir = "old/mixed_resized"',
                'cache_dir = "old/mixed_lora"',
                'recursive = true',
                'custom_attributes = {source_dir = "image_dataset/mixed", nl_tag_mix = {enabled = true, tag_ratio = 1.0}}',
                "num_repeats = 1",
            ]
        ),
        encoding="utf-8",
    )

    class FixedDatetime:
        @classmethod
        def now(cls):
            return datetime(2026, 5, 23, 12, 6, 0)

        @classmethod
        def fromtimestamp(cls, value):
            return datetime.fromtimestamp(value)

    monkeypatch.setattr(training_service, "datetime", FixedDatetime)

    training_service._prepare_web_runtime_config(
        "522",
        "default",
        "imported",
        source_config_file="configs/imported/522.toml",
    )

    run_dir = tmp_path / "output" / "runs" / "522-20260523-120600"
    mixed_source = run_dir / "dataset_cache" / "dataset-01" / "source"
    materialized = json.loads((mixed_source / "captions.json").read_text(encoding="utf-8"))
    manifest = json.loads((mixed_source / "results.json").read_text(encoding="utf-8"))

    assert materialized == captions
    assert (mixed_source / "nested" / "sample.png").is_file()
    assert not (mixed_source / "sample.png").exists()
    assert manifest["actual_tag_count"] == 1
    assert manifest["items"][0]["target"].endswith("dataset-01/source/nested/sample.png")
    assert manifest["items"][0]["captions"][0].endswith("captions.json")

