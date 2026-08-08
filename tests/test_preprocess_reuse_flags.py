"""Preprocess task wrappers honor VAE/TE reuse overwrite flags."""

from __future__ import annotations

from scripts.tasks import preprocess as preprocess_task


def test_vae_overwrite_when_reuse_disabled(monkeypatch):
    captured: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        captured.append(list(cmd))
        return 0

    monkeypatch.setattr(preprocess_task, "run", fake_run)
    monkeypatch.setattr(preprocess_task, "_preprocess_cache_batch_sizes", lambda: (1, 1))
    monkeypatch.setattr(preprocess_task, "_preprocess_precision_dtype", lambda: "bfloat16")
    monkeypatch.setattr(preprocess_task, "_path", lambda key, default: default)
    monkeypatch.setattr(preprocess_task, "_recursive_args", lambda row: [])
    monkeypatch.setattr(preprocess_task, "_path_pattern_args", lambda row: [])

    row = {
        "resized_image_dir": "post_image_dataset/resized",
        "lora_cache_dir": "post_image_dataset/lora",
        "reuse_vae_latents": False,
        "reuse_text_encoder_cache": True,
    }
    preprocess_task._run_preprocess_vae(row, [])
    assert captured, "expected cache_latents invocation"
    assert "--overwrite" in captured[0]


def test_te_overwrite_when_force_rebuild(monkeypatch):
    captured: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        captured.append(list(cmd))
        return 0

    monkeypatch.setattr(preprocess_task, "run", fake_run)
    monkeypatch.setattr(preprocess_task, "_preprocess_cache_batch_sizes", lambda: (1, 1))
    monkeypatch.setattr(preprocess_task, "_preprocess_precision_dtype", lambda: "bfloat16")
    monkeypatch.setattr(preprocess_task, "_path", lambda key, default: default)
    monkeypatch.setattr(preprocess_task, "_recursive_args", lambda row: [])
    monkeypatch.setattr(preprocess_task, "_path_pattern_args", lambda row: [])
    monkeypatch.setattr(preprocess_task, "_resolve_lowres_filter", lambda extra: ([], extra))
    monkeypatch.setattr(preprocess_task, "_caption_source_args", lambda *a, **k: [])
    monkeypatch.setattr(preprocess_task, "_caption_extension_args_for_row", lambda row: [])
    monkeypatch.setattr(preprocess_task, "_diff_output_preservation_args", lambda extra: [])
    monkeypatch.setattr(preprocess_task, "_run_caption_backup", lambda row: None)
    monkeypatch.setattr(preprocess_task, "_truthy", lambda v: bool(v))

    row = {
        "source_image_dir": "image_dataset",
        "lora_cache_dir": "post_image_dataset/lora",
        "force_rebuild_preprocess_cache": True,
    }
    preprocess_task._run_preprocess_te(row, [], "0", "0.0", backup_captions=False)
    assert captured, "expected cache_text_embeddings invocation"
    assert "--overwrite" in captured[0]


def test_no_overwrite_when_reuse_enabled(monkeypatch):
    captured: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        captured.append(list(cmd))
        return 0

    monkeypatch.setattr(preprocess_task, "run", fake_run)
    monkeypatch.setattr(preprocess_task, "_preprocess_cache_batch_sizes", lambda: (1, 1))
    monkeypatch.setattr(preprocess_task, "_preprocess_precision_dtype", lambda: "bfloat16")
    monkeypatch.setattr(preprocess_task, "_path", lambda key, default: default)
    monkeypatch.setattr(preprocess_task, "_recursive_args", lambda row: [])
    monkeypatch.setattr(preprocess_task, "_path_pattern_args", lambda row: [])

    row = {
        "resized_image_dir": "post_image_dataset/resized",
        "lora_cache_dir": "post_image_dataset/lora",
        "reuse_vae_latents": True,
    }
    preprocess_task._run_preprocess_vae(row, [])
    assert captured
    assert "--overwrite" not in captured[0]


def test_te_dispatches_to_krea2_script_when_model_family_krea2_raw(monkeypatch):
    """model_family=krea2_raw routes TE caching to scripts.krea2.preprocess_te_cache
    (writes ``_krea2_te.safetensors``); anima stays on cache_text_embeddings."""
    captured: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        captured.append(list(cmd))
        return 0

    monkeypatch.setattr(preprocess_task, "run", fake_run)
    monkeypatch.setattr(preprocess_task, "_preprocess_cache_batch_sizes", lambda: (1, 2))
    monkeypatch.setattr(preprocess_task, "_preprocess_precision_dtype", lambda: "bfloat16")
    monkeypatch.setattr(preprocess_task, "_path", lambda key, default: default)
    monkeypatch.setattr(preprocess_task, "_recursive_args", lambda row: ["--recursive"])
    monkeypatch.setattr(preprocess_task, "_model_family", lambda: "krea2_raw")
    monkeypatch.setattr(preprocess_task, "_run_caption_backup", lambda row: None)

    row = {
        "source_image_dir": "image_dataset",
        "resized_image_dir": "post_image_dataset/resized",
        "lora_cache_dir": "post_image_dataset/lora",
    }
    preprocess_task._run_preprocess_te(row, [], "0", "0.0", backup_captions=False)
    assert captured, "expected krea2 TE invocation"
    cmd = captured[0]
    assert "scripts.krea2.preprocess_te_cache" in cmd
    # Krea-2 TE runs on the source dir (captions.json master lives there; the
    # resize step does not mirror captions.json into the resized dir).
    assert "--dir" in cmd and cmd[cmd.index("--dir") + 1] == "image_dataset"
    # Krea-2 single-variant path must NOT carry anima-only flags.
    assert "--caption_shuffle_variants" not in cmd
    assert "--dit" not in cmd
    assert "--batch_size" in cmd and cmd[cmd.index("--batch_size") + 1] == "2"


def test_te_dispatches_to_anima_script_by_default(monkeypatch):
    captured: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        captured.append(list(cmd))
        return 0

    monkeypatch.setattr(preprocess_task, "run", fake_run)
    monkeypatch.setattr(preprocess_task, "_preprocess_cache_batch_sizes", lambda: (1, 1))
    monkeypatch.setattr(preprocess_task, "_preprocess_precision_dtype", lambda: "bfloat16")
    monkeypatch.setattr(preprocess_task, "_path", lambda key, default: default)
    monkeypatch.setattr(preprocess_task, "_recursive_args", lambda row: [])
    monkeypatch.setattr(preprocess_task, "_path_pattern_args", lambda row: [])
    monkeypatch.setattr(preprocess_task, "_resolve_lowres_filter", lambda extra: ([], extra))
    monkeypatch.setattr(preprocess_task, "_caption_source_args", lambda *a, **k: [])
    monkeypatch.setattr(preprocess_task, "_caption_extension_args_for_row", lambda row: [])
    monkeypatch.setattr(preprocess_task, "_diff_output_preservation_args", lambda extra: [])
    monkeypatch.setattr(preprocess_task, "_truthy", lambda v: bool(v))
    monkeypatch.setattr(preprocess_task, "_model_family", lambda: "anima")
    monkeypatch.setattr(preprocess_task, "_run_caption_backup", lambda row: None)

    row = {"source_image_dir": "image_dataset", "lora_cache_dir": "post_image_dataset/lora"}
    preprocess_task._run_preprocess_te(row, [], "0", "0.0", backup_captions=False)
    assert captured
    assert "scripts.preprocess.cache_text_embeddings" in captured[0]
