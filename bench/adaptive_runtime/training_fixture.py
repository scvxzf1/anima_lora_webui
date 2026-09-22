"""Export a verified real Krea probe input into isolated train.py sidecars.

This is a one-image wiring smoke, not a multi-input quality calibration.
No encoders are loaded and no user cache files are modified.
"""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps
from safetensors.torch import save_file
import toml
import torch

from library.anima.strategy import AnimaLatentsCachingStrategy
from library.models.family_registry import get_model_family_spec
from library.models.krea2_raw.strategy import Krea2TextEncoderOutputsCachingStrategy
from library.training.auto_block_swap.process import write_result
from .inputs import load_krea_inputs


def export_fixture(source, destination):
    source, destination = Path(source), Path(destination)
    report = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
    if report.get("status") != "ok" or report.get("schema") != "adaptive_krea_inputs_v1":
        raise ValueError("Expected a completed real Krea input report")
    resolution = int(report["resolution"])
    values, digest = load_krea_inputs(source, resolution=resolution, device="cpu", dtype=torch.float32)
    image_path = Path(report["image"])
    caption_path = image_path.with_suffix(".txt")
    for actual, expected in ((digest, report["cache_sha256"]),
                             (hashlib.sha256(image_path.read_bytes()).hexdigest(), report["image_sha256"]),
                             (hashlib.sha256(caption_path.read_bytes()).hexdigest(), report["caption_sha256"])):
        if actual != expected:
            raise ValueError("Real input source hash changed")
    hidden = values["hidden"][0].bfloat16()
    if not torch.equal(hidden.float(), values["hidden"][0]):
        raise ValueError("Text cache conversion would lose source precision")
    destination.mkdir(parents=True, exist_ok=False)
    images, caches = destination / "images", destination / "cache"
    images.mkdir()
    caches.mkdir()
    exported = images / "sample.png"
    with Image.open(image_path) as image:
        # Same geometry as prepare_krea_inputs.encode; PNG is lossless after fit.
        image = ImageOps.fit(ImageOps.exif_transpose(image).convert("RGB"),
                             (resolution, resolution), method=Image.Resampling.LANCZOS)
        image.save(exported)
    exported.with_suffix(".txt").write_bytes(caption_path.read_bytes())
    latent_strategy = AnimaLatentsCachingStrategy(True, 1, False)
    latent_path = latent_strategy.get_latents_npz_path(
        str(exported), (resolution, resolution), cache_dir=str(caches), image_dir=str(images))
    latent_strategy.save_latents_to_disk(
        latent_path, values["latents"][0, :, 0], [resolution, resolution], [0, 0, resolution, resolution],
        key_reso_suffix=f"_{resolution // 8}x{resolution // 8}")
    text_strategy = Krea2TextEncoderOutputsCachingStrategy()
    text_path = text_strategy.get_outputs_npz_path(str(exported), cache_dir=str(caches), image_dir=str(images))
    family = get_model_family_spec("krea2_raw")
    save_file({"hiddens": hidden.contiguous(), "mask": values["mask"][0].contiguous(),
               "caption_dropout_rate": torch.tensor(0.0)}, text_path,
              metadata=family.text_cache.metadata(family.name))
    if not (text_strategy.is_disk_cached_outputs_expected(text_path)
            and latent_strategy.is_disk_cached_latents_expected((resolution, resolution), latent_path, False, False)):
        raise ValueError("Exported training sidecars failed production validation")
    dataset = {"general": {"caption_extension": ".txt"}, "datasets": [{
        "resolution": resolution, "batch_size": 1, "enable_bucket": False,
        "subsets": [{"image_dir": str(images.resolve()), "cache_dir": str(caches.resolve()),
                     "num_repeats": 3, "flip_aug": False, "caption_dropout_rate": 0.0}],
    }]}
    (destination / "dataset.toml").write_text(toml.dumps(dataset), encoding="utf-8")
    write_result(destination / "provenance.json", {
        "schema": "adaptive_training_fixture_v1", "source": str(source.resolve()),
        "source_sha256": digest, "source_image_sha256": report["image_sha256"],
        "scope": "one_real_image_training_entry_smoke_not_quality_calibration",
        "noise_reused": False, "resolution": resolution,
    })
    return destination


def smoke_config(fixture, output, weights):
    if Path(output).exists():
        raise FileExistsError("Smoke training requires a fresh output directory")
    root = Path(__file__).resolve().parents[2]
    config = toml.load(root / "configs/base.toml")
    config.pop("general", None)
    config.pop("datasets", None)
    config.update(
        model_family="krea2_raw", pretrained_model_name_or_path=str(Path(weights).resolve()),
        output_dir=str(Path(output).resolve()), output_name="adaptive-smoke", save_precision="float",
        dataset_config=str((Path(fixture) / "dataset.toml").resolve()),
        adaptive_precision="fp16_fp32", adaptive_fp32_modules=[], adaptive_loss_scale=1024.0,
        adaptive_oom_retry=True, adaptive_oom_retry_max_attempts=2, adaptive_oom_retry_timeout=600.0,
        adaptive_oom_retry_swap_increment=2, adaptive_oom_retry_max_swap=26,
        mixed_precision="fp16", base_compute="bf16", attn_mode="torch", torch_compile=False,
        compile_dynamic_seq=False, gradient_checkpointing=True, blocks_to_swap=26,
        block_swap_restore_mode="foreach", network_dim=4, network_alpha=4,
        max_train_steps=3, seed=20260922, learning_rate=1e-4, lr_scheduler="constant",
        lr_warmup_steps=0, cache_latents=True, cache_latents_to_disk=True,
        cache_text_encoder_outputs=True, cache_text_encoder_outputs_to_disk=True,
        cache_llm_adapter_outputs=False, masked_loss=False, skip_cache_check=False,
        max_data_loader_n_workers=0, persistent_data_loader_workers=False,
        dataloader_pin_memory=False, use_cmmd=False, gradient_accumulation_steps=1,
        log_every_n_steps=1, use_moe_style=False, route_per_layer=False, router_source="none",
        memory_probe_jsonl=str((Path(output) / "memory.jsonl").resolve()), memory_probe_max_steps=3,
    )
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    export_fixture(args.source, args.destination)
    (args.destination / "train.toml").write_text(
        toml.dumps(smoke_config(args.destination, args.training_output, args.weights)), encoding="utf-8")
    print(args.destination / "train.toml")


if __name__ == "__main__":
    main()
