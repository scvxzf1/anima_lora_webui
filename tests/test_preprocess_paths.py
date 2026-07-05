from __future__ import annotations

from PIL import Image
import toml
import torch
from safetensors.torch import load_file

from library.preprocess import text as preprocess_text
from preprocess import cache_text_embeddings
from preprocess import resize_images
from scripts.tasks import preprocess
from scripts.tasks import _common, utilities
from scripts.experimental_tasks import training as experimental_training


def _set_path_overrides_cache(
    monkeypatch,
    overrides: dict,
    *,
    runtime_config: str = "",
    preset: str = "default",
    method: str = "",
    methods_subdir: str = "methods",
) -> None:
    monkeypatch.setattr(_common, "_PATH_OVERRIDES_CACHE", overrides)
    monkeypatch.setattr(
        _common,
        "_PATH_OVERRIDES_CACHE_KEY",
        (runtime_config, preset, method, methods_subdir),
    )


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


def test_preprocess_cache_batch_sizes_follow_memory_profile(monkeypatch):
    commands: list[list[str]] = []
    monkeypatch.delenv("ANIMA_RUNTIME_CONFIG", raising=False)
    _set_path_overrides_cache(
        monkeypatch,
        {
            "preprocess_memory_profile": "low_vram",
            "preprocess_text_cache_batch_size": "2",
            "preprocess_precision_preference": "fp16",
            "vae": "D:/models/vae.safetensors",
            "qwen3": "D:/models/qwen3.safetensors",
            "pretrained_model_name_or_path": "D:/models/anima.safetensors",
        },
    )
    monkeypatch.setattr(preprocess, "run", commands.append)
    monkeypatch.setattr(preprocess, "_run_caption_backup", lambda row: None)
    monkeypatch.setattr(preprocess, "_build_caption_index_best_effort", lambda: None)

    preprocess.cmd_preprocess([])

    _, vae_cmd, te_cmd = commands
    assert vae_cmd[vae_cmd.index("--batch_size") + 1] == "1"
    assert vae_cmd[vae_cmd.index("--dtype") + 1] == "float16"
    assert te_cmd[te_cmd.index("--batch_size") + 1] == "2"
    assert te_cmd[te_cmd.index("--dtype") + 1] == "float16"


def test_preprocess_dtype_defaults_to_bfloat16(monkeypatch):
    commands: list[list[str]] = []
    monkeypatch.delenv("ANIMA_RUNTIME_CONFIG", raising=False)
    _set_path_overrides_cache(
        monkeypatch,
        {
            "vae": "D:/models/vae.safetensors",
            "qwen3": "D:/models/qwen3.safetensors",
            "pretrained_model_name_or_path": "D:/models/anima.safetensors",
        },
    )
    monkeypatch.setattr(preprocess, "run", commands.append)
    monkeypatch.setattr(preprocess, "_run_caption_backup", lambda row: None)
    monkeypatch.setattr(preprocess, "_build_caption_index_best_effort", lambda: None)

    preprocess.cmd_preprocess([])

    _, vae_cmd, te_cmd = commands
    assert vae_cmd[vae_cmd.index("--batch_size") + 1] == "2"
    assert te_cmd[te_cmd.index("--batch_size") + 1] == "16"
    assert vae_cmd[vae_cmd.index("--dtype") + 1] == "bfloat16"
    assert te_cmd[te_cmd.index("--dtype") + 1] == "bfloat16"


def test_preprocess_dtype_falls_back_to_training_mixed_precision(monkeypatch):
    commands: list[list[str]] = []
    monkeypatch.delenv("ANIMA_RUNTIME_CONFIG", raising=False)
    _set_path_overrides_cache(
        monkeypatch,
        {
            "mixed_precision": "fp16",
            "vae": "D:/models/vae.safetensors",
            "qwen3": "D:/models/qwen3.safetensors",
            "pretrained_model_name_or_path": "D:/models/anima.safetensors",
        },
    )
    monkeypatch.setattr(preprocess, "run", commands.append)
    monkeypatch.setattr(preprocess, "_run_caption_backup", lambda row: None)
    monkeypatch.setattr(preprocess, "_build_caption_index_best_effort", lambda: None)

    preprocess.cmd_preprocess([])

    _, vae_cmd, te_cmd = commands
    assert vae_cmd[vae_cmd.index("--dtype") + 1] == "float16"
    assert te_cmd[te_cmd.index("--dtype") + 1] == "float16"


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
    _set_path_overrides_cache(
        monkeypatch,
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


def test_path_overrides_cache_is_keyed_by_preset(monkeypatch):
    from library.config import io as config_io

    calls: list[str] = []

    def fake_load_path_overrides(*, preset: str, method=None, methods_subdir: str = "methods"):
        calls.append(preset)
        return {"preset_marker": preset}

    monkeypatch.delenv("ANIMA_RUNTIME_CONFIG", raising=False)
    monkeypatch.delenv("METHOD", raising=False)
    monkeypatch.delenv("METHODS_SUBDIR", raising=False)
    monkeypatch.setattr(config_io, "load_path_overrides", fake_load_path_overrides)
    monkeypatch.setattr(_common, "_PATH_OVERRIDES_CACHE", None)
    monkeypatch.setattr(_common, "_PATH_OVERRIDES_CACHE_KEY", None)

    monkeypatch.setenv("PRESET", "default")
    first = _common._path_overrides()
    monkeypatch.setenv("PRESET", "low_vram_blockswap")
    second = _common._path_overrides()

    assert first["preset_marker"] == "default"
    assert second["preset_marker"] == "low_vram_blockswap"
    assert calls == ["default", "low_vram_blockswap"]


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
    _set_path_overrides_cache(
        monkeypatch,
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
    _set_path_overrides_cache(
        monkeypatch,
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


def test_resize_bucket_args_disable_bucket_when_dataset_requests_square_resize(tmp_path, monkeypatch):
    dataset_path = tmp_path / "configs" / "datasets" / "square.toml"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text(
        "\n".join(
            [
                "[[datasets]]",
                "resolution = 768",
                "enable_bucket = false",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(preprocess, "ROOT", tmp_path)
    _set_path_overrides_cache(
        monkeypatch,
        {"dataset_config": "configs/datasets/square.toml"},
    )

    args = preprocess._resize_bucket_args()

    assert args == ["--resolution", "768", "--no_enable_bucket"]


def test_preprocess_forwards_path_pattern_to_resize_and_cache_steps(tmp_path, monkeypatch):
    dataset_path = tmp_path / "configs" / "datasets" / "filtered.toml"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text(
        "\n".join(
            [
                "[[datasets]]",
                "resolution = 768",
                "",
                "[[datasets.subsets]]",
                'image_dir = "post_image_dataset/a_resized"',
                'cache_dir = "post_image_dataset/a_cache"',
                'path_pattern = "char_a/*"',
                'custom_attributes = {source_dir = "image_dataset/a"}',
            ]
        ),
        encoding="utf-8",
    )
    commands: list[list[str]] = []
    backups: list[tuple[str, str]] = []
    monkeypatch.setattr(preprocess, "ROOT", tmp_path)
    _set_path_overrides_cache(
        monkeypatch,
        {
            "dataset_config": "configs/datasets/filtered.toml",
            "vae": "D:/models/vae.safetensors",
            "qwen3": "D:/models/qwen3.safetensors",
            "pretrained_model_name_or_path": "D:/models/anima.safetensors",
        },
    )
    monkeypatch.setattr(preprocess, "run", commands.append)
    monkeypatch.setattr(
        preprocess,
        "_run_caption_backup",
        lambda row: backups.append((row["source_image_dir"], row.get("path_pattern"))),
    )
    monkeypatch.setattr(preprocess, "_build_caption_index_best_effort", lambda: None)

    preprocess.cmd_preprocess([])
    preprocess.cmd_preprocess_pe([])

    assert backups == [("image_dataset/a", "char_a/*")]
    resize_cmd, vae_cmd, te_cmd, pe_cmd = commands
    for cmd in (resize_cmd, vae_cmd, te_cmd, pe_cmd):
        assert cmd[cmd.index("--path_pattern") + 1] == "char_a/*"


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
    backups: list[str] = []
    monkeypatch.setattr(preprocess, "ROOT", tmp_path)
    _set_path_overrides_cache(
        monkeypatch,
        {
            "dataset_config": "runs/demo/dataset.runtime.toml",
            "qwen3": "D:/models/qwen3.safetensors",
            "pretrained_model_name_or_path": "D:/models/anima.safetensors",
        },
    )
    monkeypatch.setattr(preprocess, "run", commands.append)
    monkeypatch.setattr(
        preprocess,
        "_run_caption_backup",
        lambda row: backups.append(row["source_image_dir"]),
    )

    preprocess.cmd_preprocess_te([])

    assert backups == ["image_dataset/a"]
    assert "--prefer_json_caption" in commands[0]
    assert "--caption_source_mode" not in commands[0]


def test_runtime_dataset_config_supplies_caption_source_mode(tmp_path, monkeypatch):
    dataset_path = tmp_path / "runs" / "demo" / "dataset.runtime.toml"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text(
        "\n".join(
            [
                "[[datasets]]",
                'caption_source_mode = "captions_json"',
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
    _set_path_overrides_cache(
        monkeypatch,
        {
            "dataset_config": "runs/demo/dataset.runtime.toml",
            "qwen3": "D:/models/qwen3.safetensors",
            "pretrained_model_name_or_path": "D:/models/anima.safetensors",
        },
    )
    monkeypatch.setattr(preprocess, "run", commands.append)
    monkeypatch.setattr(preprocess, "_run_caption_backup", lambda row: None)

    preprocess.cmd_preprocess_te([])

    assert commands[0][commands[0].index("--caption_source_mode") + 1] == "captions_json"


def test_preprocess_config_uses_dataset_level_caption_sources(tmp_path, monkeypatch):
    dataset_path = tmp_path / "runs" / "demo" / "dataset.runtime.toml"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text(
        "\n".join(
            [
                "[general]",
                'caption_source_mode = "auto"',
                'caption_extension = ".txt"',
                "",
                "[[datasets]]",
                'caption_source_mode = "txt"',
                'caption_extension = ".caption"',
                "batch_size = 1",
                "",
                "[[datasets.subsets]]",
                'image_dir = "post_image_dataset/a_resized"',
                'cache_dir = "post_image_dataset/a_cache"',
                "",
                "[[datasets]]",
                'caption_source_mode = "captions_json"',
                "batch_size = 1",
                "",
                "[[datasets.subsets]]",
                'image_dir = "post_image_dataset/b_resized"',
                'cache_dir = "post_image_dataset/b_cache"',
            ]
        ),
        encoding="utf-8",
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(preprocess, "run", commands.append)

    preprocess.cmd_preprocess_config([
        "--dataset_config",
        str(dataset_path),
        "--src",
        "image_dataset/source",
        "--vae",
        "D:/models/vae.safetensors",
        "--qwen3",
        "D:/models/qwen3.safetensors",
        "--dit",
        "D:/models/anima.safetensors",
    ])

    assert len(commands) == 6
    te_a = commands[2]
    te_b = commands[5]
    assert te_a[te_a.index("--caption_source_mode") + 1] == "txt"
    assert te_a[te_a.index("--caption_extension") + 1] == ".caption"
    assert te_b[te_b.index("--caption_source_mode") + 1] == "captions_json"
    assert te_b[te_b.index("--caption_extension") + 1] == ".txt"


def test_preprocess_te_auto_forwards_diff_output_preservation_from_runtime_config(tmp_path, monkeypatch):
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
                'custom_attributes = {source_dir = "image_dataset/a"}',
                "",
            ]
        ),
        encoding="utf-8",
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(preprocess, "ROOT", tmp_path)
    _set_path_overrides_cache(
        monkeypatch,
        {
            "dataset_config": "runs/demo/dataset.runtime.toml",
            "qwen3": "D:/models/qwen3.safetensors",
            "pretrained_model_name_or_path": "D:/models/anima.safetensors",
            "diff_output_preservation_trigger": "sks",
            "diff_output_preservation_class": "woman",
        },
    )
    monkeypatch.setattr(preprocess, "run", commands.append)
    monkeypatch.setattr(preprocess, "_run_caption_backup", lambda row: None)

    preprocess.cmd_preprocess_te([])

    cmd = commands[0]
    assert cmd[cmd.index("--diff_output_preservation_trigger") + 1] == "sks"
    assert cmd[cmd.index("--diff_output_preservation_class") + 1] == "woman"


def test_preprocess_config_keeps_diff_output_preservation_out_of_resize(tmp_path, monkeypatch):
    dataset_path = tmp_path / "configs" / "datasets" / "dop.toml"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text(
        "\n".join(
            [
                "[[datasets]]",
                "resolution = 768",
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
    monkeypatch.setattr(preprocess, "run", commands.append)

    preprocess.cmd_preprocess_config([
        "--dataset_config",
        str(dataset_path),
        "--src",
        "image_dataset/source",
        "--vae",
        "D:/models/vae.safetensors",
        "--qwen3",
        "D:/models/qwen3.safetensors",
        "--dit",
        "D:/models/anima.safetensors",
        "--diff_output_preservation_trigger",
        "sks",
        "--diff_output_preservation_class",
        "woman",
    ])

    assert len(commands) == 3
    resize_cmd, vae_cmd, te_cmd = commands
    assert "--diff_output_preservation_trigger" not in resize_cmd
    assert "--diff_output_preservation_class" not in resize_cmd
    assert te_cmd[te_cmd.index("--diff_output_preservation_trigger") + 1] == "sks"
    assert te_cmd[te_cmd.index("--diff_output_preservation_class") + 1] == "woman"


def test_preprocess_config_forwards_subset_filter_scope(tmp_path, monkeypatch):
    dataset_path = tmp_path / "configs" / "datasets" / "filtered.toml"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text(
        "\n".join(
            [
                "[[datasets]]",
                "recursive = true",
                "",
                "[[datasets.subsets]]",
                'image_dir = "post_image_dataset/a_resized"',
                'cache_dir = "post_image_dataset/a_cache"',
                "recursive = false",
                'path_pattern = "char_a/*"',
            ]
        ),
        encoding="utf-8",
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(preprocess, "run", commands.append)

    preprocess.cmd_preprocess_config([
        "--dataset_config",
        str(dataset_path),
        "--src",
        "image_dataset/source",
        "--vae",
        "D:/models/vae.safetensors",
        "--qwen3",
        "D:/models/qwen3.safetensors",
        "--dit",
        "D:/models/anima.safetensors",
    ])

    assert len(commands) == 3
    for cmd in commands:
        assert "--recursive" not in cmd
        assert cmd[cmd.index("--path_pattern") + 1] == "char_a/*"


def test_caption_backup_dir_uses_cache_parent_and_stays_out_of_training_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(preprocess, "ROOT", tmp_path)
    row = {
        "source_image_dir": "image_dataset/a",
        "resized_image_dir": "output/runs/demo/dataset_cache/dataset-02/resized",
        "lora_cache_dir": "output/runs/demo/dataset_cache/dataset-02/lora",
    }

    backup_dir = preprocess._caption_backup_dir_for_row(row)

    assert backup_dir == tmp_path / "output/runs/demo/dataset_cache/dataset-02/caption_backup"
    assert backup_dir != tmp_path / row["resized_image_dir"]
    assert backup_dir != tmp_path / row["lora_cache_dir"]


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
        _cache_dtype,
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


def test_cache_text_embeddings_writes_captions_json_as_multi_source_variants(tmp_path, monkeypatch):
    image = tmp_path / "hero.png"
    Image.new("RGB", (800, 800), color=(128, 128, 128)).save(image)
    (tmp_path / "captions.json").write_text(
        '{"hero.png": ["caption one", "caption two"]}',
        encoding="utf-8",
    )

    seen_captions: list[str] = []

    def fake_encode_batch(
        captions,
        _tokenize_strategy,
        _encoding_strategy,
        _text_encoder,
        _llm_adapter,
        _device,
        _cache_dtype,
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
        caption_source_mode="auto",
        caption_shuffle_variants=0,
    )

    cache = load_file(str(tmp_path / "cache" / "hero_anima_te.safetensors"))
    assert stats.written == 1
    assert seen_captions == ["caption one", "caption two"]
    assert int(cache["num_variants"]) == 2
    assert int(cache["caption_multi_source"]) == 1


def test_cache_text_embeddings_writes_diff_output_prior_crossattn(tmp_path, monkeypatch):
    image = tmp_path / "hero.png"
    Image.new("RGB", (800, 800), color=(128, 128, 128)).save(image)
    image.with_suffix(".txt").write_text("sks woman, portrait\n", encoding="utf-8")

    seen_captions: list[list[str]] = []

    def fake_encode_batch(
        captions,
        _tokenize_strategy,
        _encoding_strategy,
        _text_encoder,
        _llm_adapter,
        _device,
        _cache_dtype,
    ):
        seen_captions.append(list(captions))
        n = len(captions)
        marker = 7 if captions == ["sks woman, portrait"] else 11
        return (
            torch.zeros((n, 2, 3), dtype=torch.bfloat16),
            torch.ones((n, 2), dtype=torch.int32),
            torch.zeros((n, 2), dtype=torch.long),
            torch.ones((n, 2), dtype=torch.int32),
            torch.full((n, 2, 4), marker, dtype=torch.bfloat16),
        )

    monkeypatch.setattr(preprocess_text, "_encode_batch", fake_encode_batch)

    stats = preprocess_text.cache_text_embeddings(
        tmp_path,
        object(),
        object(),
        object(),
        llm_adapter=object(),
        device=torch.device("cpu"),
        cache_dir=tmp_path / "cache",
        batch_size=8,
        min_pixels=500_000,
        verbose=False,
        diff_output_preservation_trigger="sks",
        diff_output_preservation_class="woman",
    )

    cache = load_file(str(tmp_path / "cache" / "hero_anima_te.safetensors"))
    assert stats.written == 1
    assert seen_captions == [
        ["sks woman, portrait"],
        ["woman woman, portrait"],
    ]
    assert torch.equal(cache["crossattn_emb"], torch.full((2, 4), 7, dtype=torch.bfloat16))
    assert torch.equal(cache["prior_crossattn_emb"], torch.full((2, 4), 11, dtype=torch.bfloat16))


def test_cache_text_embeddings_uses_text_encoder_dtype_for_saved_tensors(tmp_path, monkeypatch):
    image = tmp_path / "hero.png"
    Image.new("RGB", (800, 800), color=(128, 128, 128)).save(image)
    image.with_suffix(".txt").write_text("tag one\n", encoding="utf-8")

    def fake_encode_batch(
        captions,
        _tokenize_strategy,
        _encoding_strategy,
        _text_encoder,
        _llm_adapter,
        _device,
        cache_dtype,
    ):
        n = len(captions)
        return (
            torch.zeros((n, 2, 3), dtype=cache_dtype),
            torch.ones((n, 2), dtype=torch.int32),
            torch.zeros((n, 2), dtype=torch.long),
            torch.ones((n, 2), dtype=torch.int32),
            torch.full((n, 2, 4), 7, dtype=cache_dtype),
        )

    monkeypatch.setattr(preprocess_text, "_encode_batch", fake_encode_batch)

    class DummyTextEncoder:
        dtype = torch.float16

    stats = preprocess_text.cache_text_embeddings(
        tmp_path,
        object(),
        object(),
        DummyTextEncoder(),
        llm_adapter=object(),
        device=torch.device("cpu"),
        cache_dir=tmp_path / "cache",
        batch_size=8,
        min_pixels=500_000,
        verbose=False,
    )

    cache = load_file(str(tmp_path / "cache" / "hero_anima_te.safetensors"))
    assert stats.written == 1
    assert cache["prompt_embeds"].dtype == torch.float16
    assert cache["crossattn_emb"].dtype == torch.float16


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
    backups: list[str] = []
    monkeypatch.setattr(preprocess, "ROOT", tmp_path)
    _set_path_overrides_cache(
        monkeypatch,
        {
            "dataset_config": "configs/datasets/multi.toml",
            "vae": "D:/models/vae.safetensors",
            "qwen3": "D:/models/qwen3.safetensors",
            "pretrained_model_name_or_path": "D:/models/anima.safetensors",
        },
    )
    monkeypatch.setattr(preprocess, "run", commands.append)
    monkeypatch.setattr(
        preprocess,
        "_run_caption_backup",
        lambda row: backups.append(row["source_image_dir"]),
    )
    monkeypatch.setattr(preprocess, "_build_caption_index_best_effort", lambda: None)

    preprocess.cmd_preprocess([])

    assert backups == ["image_dataset/a", "image_dataset/b"]
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
