"""Configuration forwarding checks for Anima vocab packs."""

from scripts.tasks import preprocess


def test_anima_preprocess_forwards_configured_vocab_pack(monkeypatch):
    commands = []
    monkeypatch.setattr(
        preprocess,
        "_path_overrides_value",
        lambda: {
            "model_family": "anima",
            "source_image_dir": "image_dataset",
            "lora_cache_dir": "post_image_dataset/lora",
            "qwen3": "models/qwen3.safetensors",
            "pretrained_model_name_or_path": "models/anima.safetensors",
            "vocab_pack_path": "models/vocab_packs/anima_cjk_vocab_pack_jp_v1",
        },
    )
    monkeypatch.setattr(preprocess, "run", commands.append)
    monkeypatch.setattr(preprocess, "_run_caption_backup", lambda row: None)
    monkeypatch.setattr(preprocess, "_min_pixels_args", lambda: [])

    preprocess.cmd_preprocess_te([])

    command = commands[0]
    assert command[command.index("--vocab_pack_path") + 1] == (
        "models/vocab_packs/anima_cjk_vocab_pack_jp_v1"
    )
