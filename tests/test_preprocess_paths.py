from __future__ import annotations

from PIL import Image
import toml
import torch

from library.preprocess import text as preprocess_text
from preprocess import cache_text_embeddings
from preprocess import resize_images
from scripts.tasks import preprocess
from scripts.tasks import _common, utilities
from scripts.experimental_tasks import training as experimental_training


def test_preprocess_vae_uses_configured_vae_path(monkeypatch):
    commands: list[list[str]] = []

    def fake_path(key: str, default: str) -> str:
        return {
            "resized_image_dir": "D:/data/resized",
            "lora_cache_dir": "D:/data/lora_cache",
            "vae": "D:/models/VAE/qwen_image_vae.safetensors",
        }.get(key, default)

    monkeypatch.setattr(preprocess, "_path", fake_path)
    monkeypatch.setattr(preprocess, "run", commands.append)

    preprocess.cmd_preprocess_vae([])

    cmd = commands[0]
    assert (
        cmd[cmd.index("--vae") + 1]
        == "D:/models/VAE/qwen_image_vae.safetensors"
    )


def test_preprocess_te_uses_configured_model_paths(monkeypatch):
    commands: list[list[str]] = []

    def fake_path(key: str, default: str) -> str:
        return {
            "source_image_dir": "D:/data/source",
            "lora_cache_dir": "D:/data/lora_cache",
            "qwen3": "D:/models/text_encoder/qwen_3_06b_base.safetensors",
            "pretrained_model_name_or_path": "D:/models/anima/anima_base.safetensors",
        }.get(key, default)

    monkeypatch.setattr(preprocess, "_path", fake_path)
    monkeypatch.setattr(preprocess, "run", commands.append)
    monkeypatch.setattr(preprocess, "_min_pixels_args", lambda: [])

    preprocess.cmd_preprocess_te([])

    cmd = commands[0]
    assert (
        cmd[cmd.index("--qwen3") + 1]
        == "D:/models/text_encoder/qwen_3_06b_base.safetensors"
    )
    assert cmd[cmd.index("--dit") + 1] == "D:/models/anima/anima_base.safetensors"


def test_easycontrol_preprocess_uses_configured_model_paths(monkeypatch):
    commands: list[list[str]] = []

    def fake_path(key: str, default: str) -> str:
        return {
            "vae": "D:/models/VAE/qwen_image_vae.safetensors",
            "qwen3": "D:/models/text_encoder/qwen_3_06b_base.safetensors",
            "pretrained_model_name_or_path": "D:/models/anima/anima_base.safetensors",
        }.get(key, default)

    monkeypatch.setattr(experimental_training, "_path", fake_path)
    monkeypatch.setattr(experimental_training, "run", commands.append)

    experimental_training.cmd_easycontrol_preprocess([])

    vae_cmd, te_cmd = commands
    assert (
        vae_cmd[vae_cmd.index("--vae") + 1]
        == "D:/models/VAE/qwen_image_vae.safetensors"
    )
    assert (
        te_cmd[te_cmd.index("--qwen3") + 1]
        == "D:/models/text_encoder/qwen_3_06b_base.safetensors"
    )
    assert (
        te_cmd[te_cmd.index("--dit") + 1]
        == "D:/models/anima/anima_base.safetensors"
    )


def test_inference_base_uses_configured_model_paths(monkeypatch):
    monkeypatch.setattr(
        _common,
        "_PATH_OVERRIDES_CACHE",
        {
            "pretrained_model_name_or_path": "D:/models/anima/anima_base.safetensors",
            "qwen3": "D:/models/text_encoder/qwen_3_06b_base.safetensors",
            "vae": "D:/models/VAE/qwen_image_vae.safetensors",
        },
    )

    cmd = _common.build_inference_base()

    assert cmd[cmd.index("--dit") + 1] == "D:/models/anima/anima_base.safetensors"
    assert (
        cmd[cmd.index("--text_encoder") + 1]
        == "D:/models/text_encoder/qwen_3_06b_base.safetensors"
    )
    assert (
        cmd[cmd.index("--vae") + 1]
        == "D:/models/VAE/qwen_image_vae.safetensors"
    )


def test_path_overrides_use_anima_runtime_config(tmp_path, monkeypatch):
    runtime_config = tmp_path / "runs" / "522-20260523-114514" / "config.runtime.toml"
    runtime_config.parent.mkdir(parents=True)
    runtime_config.write_text(
        toml.dumps({
            "dataset_config": "output/runs/522-20260523-114514/dataset.runtime.toml",
            "output_dir": "output/runs/522-20260523-114514/training_output",
            "source_image_dir": "image_dataset/source",
            "resized_image_dir": "output/runs/522-20260523-114514/dataset_cache/dataset-01/resized",
            "lora_cache_dir": "output/runs/522-20260523-114514/dataset_cache/dataset-01/lora",
            "general": {"ignored": True},
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANIMA_RUNTIME_CONFIG", str(runtime_config))
    monkeypatch.setattr(_common, "_PATH_OVERRIDES_CACHE", None)
    monkeypatch.setattr(_common, "_PATH_OVERRIDES_CACHE_KEY", None)

    overrides = _common._path_overrides()

    assert overrides["dataset_config"].endswith("dataset.runtime.toml")
    assert overrides["output_dir"].endswith("training_output")
    assert overrides["resized_image_dir"].endswith("dataset-01/resized")
    assert "general" not in overrides


def test_task_run_adds_project_root_to_pythonpath(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run(cmd, cwd=None, env=None, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env
        captured["kwargs"] = kwargs

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(_common.subprocess, "run", fake_run)

    _common.run([_common.PY, "scripts/preprocess/resize_images.py"])

    env = captured["env"]
    assert isinstance(env, dict)
    pythonpath = env.get("PYTHONPATH", "")
    assert pythonpath.split(_common.os.pathsep)[0] == str(_common.ROOT)


def test_distill_mod_uses_configured_paths(monkeypatch):
    commands: list[list[str]] = []

    def fake_path(key: str, default: str) -> str:
        return {
            "lora_cache_dir": "D:/data/lora_cache",
            "pretrained_model_name_or_path": "D:/models/anima/anima_base.safetensors",
        }.get(key, default)

    monkeypatch.setattr(utilities, "_path", fake_path)
    monkeypatch.setattr(utilities, "run", commands.append)
    monkeypatch.setattr(utilities, "bespoke_preset_flags", lambda preset: [])

    utilities.cmd_distill_mod([])

    cmd = commands[0]
    assert cmd[cmd.index("--data_dir") + 1] == "D:/data/lora_cache"
    assert cmd[cmd.index("--dit_path") + 1] == "D:/models/anima/anima_base.safetensors"


def test_resize_bucket_args_use_dataset_no_upscale(tmp_path, monkeypatch):
    dataset_path = tmp_path / "configs" / "datasets" / "no_upscale.toml"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text(
        "\n".join(
            [
                "[[datasets]]",
                "resolution = 768",
                "min_bucket_reso = 256",
                "max_bucket_reso = 768",
                "bucket_reso_steps = 32",
                "bucket_no_upscale = true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(preprocess, "ROOT", tmp_path)
    monkeypatch.setattr(
        _common,
        "_PATH_OVERRIDES_CACHE",
        {"dataset_config": "configs/datasets/no_upscale.toml"},
    )

    args = preprocess._resize_bucket_args()

    assert args == [
        "--resolution",
        "768",
        "--min_bucket_reso",
        "256",
        "--max_bucket_reso",
        "768",
        "--bucket_reso_steps",
        "32",
        "--bucket_no_upscale",
    ]


def test_resize_bucket_args_use_runtime_preprocess_attrs(tmp_path, monkeypatch):
    dataset_path = tmp_path / "runs" / "demo" / "dataset.runtime.toml"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text(
        "\n".join(
            [
                "[[datasets]]",
                "batch_size = 1",
                "",
                "[[datasets.subsets]]",
                'image_dir = "post_image_dataset/a_resized"',
                'cache_dir = "post_image_dataset/a_cache"',
                (
                    'custom_attributes = {source_dir = "image_dataset/a", '
                    'preprocess = {resolution = 768, min_bucket_reso = 256, '
                    'max_bucket_reso = 768, bucket_reso_steps = 32, '
                    'bucket_no_upscale = true}}'
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(preprocess, "ROOT", tmp_path)
    monkeypatch.setattr(
        _common,
        "_PATH_OVERRIDES_CACHE",
        {"dataset_config": "runs/demo/dataset.runtime.toml"},
    )

    rows = preprocess._preprocess_rows()
    args = preprocess._resize_bucket_args(rows[0])

    assert rows[0]["source_image_dir"] == "image_dataset/a"
    assert args == [
        "--resolution",
        "768",
        "--min_bucket_reso",
        "256",
        "--max_bucket_reso",
        "768",
        "--bucket_reso_steps",
        "32",
        "--bucket_no_upscale",
    ]


def test_runtime_dataset_config_supplies_json_caption_flag(tmp_path, monkeypatch):
    dataset_path = tmp_path / "runs" / "demo" / "dataset.runtime.toml"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text(
        "\n".join(
            [
                "[general]",
                "prefer_json_caption = true",
                "",
                "[[datasets]]",
                "batch_size = 1",
                "",
                "[[datasets.subsets]]",
                'image_dir = "post_image_dataset/a_resized"',
                'cache_dir = "post_image_dataset/a_cache"',
                'custom_attributes = {source_dir = "image_dataset/a"}',
                "",
            ]
        ),
        encoding="utf-8",
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(preprocess, "ROOT", tmp_path)
    monkeypatch.setattr(
        _common,
        "_PATH_OVERRIDES_CACHE",
        {
            "dataset_config": "runs/demo/dataset.runtime.toml",
            "qwen3": "D:/models/qwen3.safetensors",
            "pretrained_model_name_or_path": "D:/models/anima.safetensors",
        },
    )
    monkeypatch.setattr(preprocess, "run", commands.append)

    preprocess.cmd_preprocess_te([])

    assert "--prefer_json_caption" in commands[0]


def test_cache_text_embeddings_keeps_uncaptioned_images(tmp_path):
    captioned = tmp_path / "captioned.png"
    missing = tmp_path / "missing.png"
    empty = tmp_path / "empty.png"
    small = tmp_path / "small.png"
    for path, size in [
        (captioned, (800, 800)),
        (missing, (800, 800)),
        (empty, (800, 800)),
        (small, (32, 32)),
    ]:
        Image.new("RGB", size, color=(128, 128, 128)).save(path)
    captioned.with_suffix(".txt").write_text("tag one, tag two\nignored", encoding="utf-8")
    empty.with_suffix(".txt").write_text("\n", encoding="utf-8")

    entries, skipped_small, missing_captions, empty_caption_files, samples = (
        cache_text_embeddings._collect_image_caption_entries(
            [captioned, missing, empty, small],
            min_pixels=500_000,
        )
    )

    assert skipped_small == 1
    assert missing_captions == 1
    assert empty_caption_files == 1
    assert [(path.name, caption) for path, caption in entries] == [
        ("captioned.png", "tag one, tag two"),
        ("missing.png", ""),
        ("empty.png", ""),
    ]
    assert samples == ["missing.png", "empty.png"]


def test_cache_text_embeddings_writes_missing_caption_caches(tmp_path, monkeypatch):
    captioned = tmp_path / "captioned.png"
    missing = tmp_path / "missing.png"
    empty = tmp_path / "empty.png"
    for path in (captioned, missing, empty):
        Image.new("RGB", (800, 800), color=(128, 128, 128)).save(path)
    captioned.with_suffix(".txt").write_text("tag one\n", encoding="utf-8")
    empty.with_suffix(".txt").write_text("\n", encoding="utf-8")

    seen_captions: list[str] = []

    def fake_encode_batch(
        captions,
        _tokenize_strategy,
        _encoding_strategy,
        _text_encoder,
        _llm_adapter,
        _device,
    ):
        seen_captions.extend(captions)
        n = len(captions)
        return (
            torch.zeros((n, 2, 3), dtype=torch.bfloat16),
            torch.ones((n, 2), dtype=torch.int32),
            torch.zeros((n, 2), dtype=torch.long),
            torch.ones((n, 2), dtype=torch.int32),
            None,
        )

    monkeypatch.setattr(preprocess_text, "_encode_batch", fake_encode_batch)

    stats = preprocess_text.cache_text_embeddings(
        tmp_path,
        object(),
        object(),
        object(),
        device=torch.device("cpu"),
        cache_dir=tmp_path / "cache",
        batch_size=8,
        min_pixels=500_000,
        verbose=False,
    )

    assert stats.written == 3
    assert seen_captions == ["tag one", "", ""]
    for path in (captioned, missing, empty):
        assert (tmp_path / "cache" / f"{path.stem}_anima_te.safetensors").is_file()


def test_preprocess_runs_all_dataset_config_rows(tmp_path, monkeypatch):
    dataset_path = tmp_path / "configs" / "datasets" / "multi.toml"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text(
        "\n".join(
            [
                "[[datasets]]",
                "resolution = 768",
                "min_bucket_reso = 256",
                "max_bucket_reso = 768",
                "bucket_reso_steps = 32",
                "bucket_no_upscale = true",
                "",
                "[[datasets.subsets]]",
                'image_dir = "post_image_dataset/a_resized"',
                'cache_dir = "post_image_dataset/a_cache"',
                'custom_attributes = {source_dir = "image_dataset/a"}',
                "",
                "[[datasets]]",
                "resolution = 1024",
                "min_bucket_reso = 384",
                "max_bucket_reso = 1344",
                "bucket_reso_steps = 64",
                "bucket_no_upscale = false",
                "",
                "[[datasets.subsets]]",
                'image_dir = "post_image_dataset/b_resized"',
                'cache_dir = "post_image_dataset/b_cache"',
                'custom_attributes = {source_dir = "image_dataset/b"}',
            ]
        ),
        encoding="utf-8",
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(preprocess, "ROOT", tmp_path)
    monkeypatch.setattr(
        _common,
        "_PATH_OVERRIDES_CACHE",
        {
            "dataset_config": "configs/datasets/multi.toml",
            "vae": "D:/models/vae.safetensors",
            "qwen3": "D:/models/qwen3.safetensors",
            "pretrained_model_name_or_path": "D:/models/anima.safetensors",
        },
    )
    monkeypatch.setattr(preprocess, "run", commands.append)

    preprocess.cmd_preprocess([])

    assert len(commands) == 6
    resize_a, vae_a, te_a, resize_b, vae_b, te_b = commands
    assert resize_a[1:3] == ["-m", "scripts.preprocess.resize_images"]
    assert resize_a[resize_a.index("--src") + 1] == "image_dataset/a"
    assert resize_a[resize_a.index("--dst") + 1] == "post_image_dataset/a_resized"
    assert resize_a[resize_a.index("--resolution") + 1] == "768"
    assert "--bucket_no_upscale" in resize_a
    assert vae_a[1:3] == ["-m", "scripts.preprocess.cache_latents"]
    assert vae_a[vae_a.index("--dir") + 1] == "post_image_dataset/a_resized"
    assert vae_a[vae_a.index("--cache_dir") + 1] == "post_image_dataset/a_cache"
    assert vae_a[vae_a.index("--vae") + 1] == "D:/models/vae.safetensors"
    assert te_a[1:3] == ["-m", "scripts.preprocess.cache_text_embeddings"]
    assert te_a[te_a.index("--dir") + 1] == "image_dataset/a"
    assert te_a[te_a.index("--cache_dir") + 1] == "post_image_dataset/a_cache"
    assert te_a[te_a.index("--qwen3") + 1] == "D:/models/qwen3.safetensors"
    assert te_a[te_a.index("--dit") + 1] == "D:/models/anima.safetensors"

    assert resize_b[resize_b.index("--src") + 1] == "image_dataset/b"
    assert resize_b[resize_b.index("--dst") + 1] == "post_image_dataset/b_resized"
    assert resize_b[resize_b.index("--resolution") + 1] == "1024"
    assert "--bucket_no_upscale" not in resize_b
    assert vae_b[vae_b.index("--cache_dir") + 1] == "post_image_dataset/b_cache"
    assert te_b[te_b.index("--dir") + 1] == "image_dataset/b"


def test_resize_process_image_does_not_upscale_when_disabled(tmp_path):
    src = tmp_path / "source"
    dst = tmp_path / "resized"
    src.mkdir()
    image_path = src / "small.png"
    Image.new("RGB", (700, 900), color=(255, 0, 0)).save(image_path)

    resize_images.process_image(
        image_path,
        dst,
        ((1024, 1024), 256, 1024, 64, True, True),
        copy_captions=False,
    )

    with Image.open(dst / "small.png") as image:
        assert image.size == (640, 896)
