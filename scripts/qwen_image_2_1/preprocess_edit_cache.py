"""Build Qwen Image 2.1 Edit caches from the training dataset blueprint."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from library.anima.text_strategies import (
    LatentsCachingStrategy,
    TextEncoderOutputsCachingStrategy,
    TextEncodingStrategy,
    TokenizeStrategy,
)
from library.config import loader as config_util
from library.models.qwen_image_2_1.strategy import (
    QwenImage21EditTextCache,
    QwenImage21EditTokenizeStrategy,
    QwenImage21LatentCache,
    QwenImage21TextEncodingStrategy,
)
from library.runtime.device import str_to_dtype
from library.models.qwen_image_2_1.cache_policy import (
    QWEN_TEXT_ENCODER_CACHE_POLICIES, resolve_cache_policy,
)


class _SingleProcess:
    num_processes = 1
    process_index = 0

    def __init__(self, device: torch.device):
        self.device = device

    def wait_for_everyone(self) -> None:
        """The cache CLI has exactly one process, so no barrier is needed."""
        return None


def build_edit_datasets(config_path: str):
    """Use the same image, caption, pairing, and bucket construction as training."""
    config = config_util.load_user_config(config_path)
    rows = [
        subset
        for dataset in config.get("datasets", [])
        for subset in dataset.get("subsets", [])
    ]
    if not rows or any(not subset.get("reference_image_dir") for subset in rows):
        raise ValueError("Qwen Image 2.1 Edit preprocessing requires paired reference_image_dir in every subset")
    for scope in (config.get("general", {}), *config.get("datasets", []), *rows):
        if any(scope.get(key) for key in ("flip_aug", "color_aug", "random_crop", "enable_wildcard")):
            raise ValueError("Qwen Image 2.1 Edit requires deterministic captions and image transforms")
        if any(float(scope.get(key) or 0) for key in ("caption_dropout_rate", "caption_tag_dropout_rate")):
            raise ValueError("Qwen Image 2.1 Edit does not support caption dropout")
    for subset in rows:
        if subset.get("is_reg"):
            raise ValueError("Qwen Image 2.1 Edit does not support regularization subsets")
    blueprint = config_util.BlueprintGenerator(
        config_util.ConfigSanitizer(support_dropout=True)
    ).generate(config, argparse.Namespace(model_family="qwen_image_2_1"))
    train, val = config_util.generate_dataset_group_by_blueprint(
        blueprint.dataset_group, constant_token_buckets=True
    )
    groups = [group for group in (train, val) if group is not None]
    for group in groups:
        if not group.image_data:
            continue
        for dataset in group.datasets:
            for info in dataset.image_data.values():
                if not info.reference_image_path or not info.caption.strip():
                    raise ValueError(f"Qwen Image 2.1 Edit requires a paired image and non-empty instruction: {info.absolute_path}")
                caption_path = Path(info.absolute_path).with_suffix(
                    dataset.image_to_subset[info.image_key].caption_extension
                )
                if caption_path.exists() and len(caption_path.read_text(encoding="utf-8").splitlines()) > 1:
                    raise ValueError(f"Qwen Image 2.1 Edit does not support caption variants: {caption_path}")
    return groups


def cache_edit(config_path: str, *, vae_path: str, qwen3_path: str,
               dtype: str = "bfloat16", device: str = "auto", offload: str = "auto",
               overwrite: bool = False, cache_policy: str = "auto") -> None:
    # Validate even when all caches exist; TE policy never changes VAE placement.
    resolve_cache_policy(cache_policy, device=device, offload=offload)
    groups = build_edit_datasets(config_path)
    if not any(group.image_data for group in groups):
        raise ValueError("Qwen Image 2.1 Edit dataset contains no training images")
    compute_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                                  "cpu" if device == "auto" else device)
    accelerator = _SingleProcess(compute_device)
    latent_cache = QwenImage21LatentCache(True, 1, False)
    LatentsCachingStrategy.set_strategy(latent_cache)
    text_cache = QwenImage21EditTextCache(True, 1, False)
    TextEncoderOutputsCachingStrategy.set_strategy(text_cache)
    if overwrite or any(not group.is_text_encoder_outputs_cache_complete() for group in groups):
        from library.models.qwen_image_2_1.text_encoder_runtime import text_encoder_for_cache

        TokenizeStrategy.set_strategy(QwenImage21EditTokenizeStrategy(qwen3_path))
        TextEncodingStrategy.set_strategy(QwenImage21TextEncodingStrategy())
        valid_text = text_cache.is_expected_for_info
        if overwrite:
            text_cache.is_expected_for_info = lambda *_args: False
        with text_encoder_for_cache(
            qwen3_path, dtype=str_to_dtype(dtype), device=device, offload=offload,
            cache_policy=cache_policy,
        ) as model:
            try:
                for group in groups:
                    group.new_cache_text_encoder_outputs([model], _SingleProcess(model._qwen_execution_device))
            finally:
                if overwrite:
                    text_cache.is_expected_for_info = valid_text
        del model
        gc.collect()
    else:
        print(f"Qwen3-VL cache policy: requested={cache_policy}; valid text caches reused, encoder not loaded", flush=True)
    needs_vae = overwrite or any(not group.is_latents_cache_complete() for group in groups)
    if needs_vae:
        from library.models.qwen_image_2_1.weights import load_qwen_image_2_1_vae

        vae = load_qwen_image_2_1_vae(vae_path, dtype=str_to_dtype(dtype), device="cpu")
        try:
            vae.to(compute_device, dtype=str_to_dtype(dtype)).requires_grad_(False).eval()
            valid_target = latent_cache.is_disk_cached_latents_expected
            valid_reference = latent_cache.is_edit_reference_cache_expected
            if overwrite:
                latent_cache.is_disk_cached_latents_expected = lambda *_args: False
                latent_cache.is_edit_reference_cache_expected = lambda *_args: False
            for group in groups:
                group.new_cache_latents(vae, accelerator)
        finally:
            if overwrite:
                latent_cache.is_disk_cached_latents_expected = valid_target
                latent_cache.is_edit_reference_cache_expected = valid_reference
            vae.to("cpu")
            del vae
            gc.collect()
            if compute_device.type == "cuda":
                torch.cuda.empty_cache()
    if not all(group.is_latents_cache_complete() and group.is_text_encoder_outputs_cache_complete()
               for group in groups):
        raise RuntimeError("Qwen Image 2.1 Edit preprocessing left incomplete caches")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_config", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--qwen3", required=True)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--offload", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--cache_policy", choices=QWEN_TEXT_ENCODER_CACHE_POLICIES, default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cache_edit(args.dataset_config, vae_path=args.vae, qwen3_path=args.qwen3,
               dtype=args.dtype, device=args.device, offload=args.offload,
               overwrite=args.overwrite, cache_policy=args.cache_policy)


if __name__ == "__main__":
    main()
