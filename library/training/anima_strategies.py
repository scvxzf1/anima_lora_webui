"""Anima text / latent strategy factories used by AnimaTrainer.

Family dispatch (stage 6): each ``get_*`` reads ``resolve_model_family(args)``
and returns the matching Anima or Krea-2-Raw strategy. Anima is the default and
keeps byte-identical behavior; Krea-2 dispatches to ``library.models.krea2_raw``
strategies (Qwen3-VL ChatML tokenize/encode + ``_krea2_te`` cache suffix).
"""

from __future__ import annotations

import torch

from library.anima import strategy as strategy_anima
from library.env import resolve_model_family
from library.models.family_registry import dispatch_model_family


def _krea2_tokenize_strategy(args):
    from library.models.krea2_raw.strategy import Krea2TokenizeStrategy

    return Krea2TokenizeStrategy()


def _z_image_tokenize_strategy(args):
    from library.models.z_image.strategy import ZImageTokenizeStrategy

    return ZImageTokenizeStrategy(args.qwen3)


def _anima_tokenize_strategy(args):
    return strategy_anima.AnimaTokenizeStrategy(
        qwen3_path=args.qwen3,
        t5_tokenizer_path=args.t5_tokenizer_path,
        qwen3_max_length=args.qwen3_max_token_length,
        t5_max_length=args.t5_max_token_length,
    )


def get_tokenize_strategy(args):
    factory = dispatch_model_family(
        resolve_model_family(args),
        operation="training tokenize strategy",
        handlers={
            "anima": _anima_tokenize_strategy,
            "krea2_raw": _krea2_tokenize_strategy,
            "z_image": _z_image_tokenize_strategy,
        },
    )
    return factory(args)


def get_tokenizers(tokenize_strategy):
    if isinstance(tokenize_strategy, strategy_anima.AnimaTokenizeStrategy):
        return [tokenize_strategy.qwen3_tokenizer]
    from library.models.krea2_raw.strategy import Krea2TokenizeStrategy

    if isinstance(tokenize_strategy, Krea2TokenizeStrategy):
        return [tokenize_strategy.tokenizer]
    from library.models.z_image.strategy import ZImageTokenizeStrategy

    if isinstance(tokenize_strategy, ZImageTokenizeStrategy):
        return [tokenize_strategy.tokenizer]
    raise TypeError(
        f"Unsupported tokenize strategy: {type(tokenize_strategy).__name__}"
    )


def get_latents_caching_strategy(args):
    # VAE / latent cache is shared (same AutoencoderKLQwenImage); anima's
    # caching strategy is family-agnostic.
    factory = dispatch_model_family(
        resolve_model_family(args),
        operation="latent cache strategy",
        handlers={
            "anima": strategy_anima.AnimaLatentsCachingStrategy,
            "krea2_raw": strategy_anima.AnimaLatentsCachingStrategy,
            "z_image": _z_image_latents_caching_strategy,
        },
    )
    return factory(
        args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
    )


def get_text_encoding_strategy(args):
    def krea2_factory():
        from library.models.krea2_raw.strategy import Krea2TextEncodingStrategy

        return Krea2TextEncodingStrategy()

    def z_image_factory():
        from library.models.z_image.strategy import ZImageTextEncodingStrategy

        return ZImageTextEncodingStrategy()

    factory = dispatch_model_family(
        resolve_model_family(args),
        operation="training text-encoding strategy",
        handlers={
            "anima": strategy_anima.AnimaTextEncodingStrategy,
            "krea2_raw": krea2_factory,
            "z_image": z_image_factory,
        },
    )
    return factory()


def get_text_encoder_outputs_caching_strategy(args, weight_dtype: torch.dtype):
    if not args.cache_text_encoder_outputs:
        return None

    def krea2_factory():
        from library.models.krea2_raw.strategy import (
            Krea2TextEncoderOutputsCachingStrategy,
        )

        return Krea2TextEncoderOutputsCachingStrategy(
            args.cache_text_encoder_outputs_to_disk,
            args.text_encoder_batch_size,
            args.skip_cache_check,
            use_shuffled_caption_variants=getattr(
                args, "use_shuffled_caption_variants", False
            ),
        )

    def anima_factory():
        return strategy_anima.AnimaTextEncoderOutputsCachingStrategy(
            args.cache_text_encoder_outputs_to_disk,
            args.text_encoder_batch_size,
            args.skip_cache_check,
            False,
            cache_llm_adapter_outputs=getattr(args, "cache_llm_adapter_outputs", False),
            use_shuffled_caption_variants=getattr(
                args, "use_shuffled_caption_variants", False
            ),
            diff_output_preservation_trigger=getattr(
                args, "diff_output_preservation_trigger", None
            ),
            diff_output_preservation_class=getattr(
                args, "diff_output_preservation_class", None
            ),
            cache_dtype=weight_dtype,
        )

    def z_image_factory():
        from library.models.z_image.strategy import (
            ZImageTextEncoderOutputsCachingStrategy,
        )

        return ZImageTextEncoderOutputsCachingStrategy(
            args.cache_text_encoder_outputs_to_disk,
            args.text_encoder_batch_size,
            args.skip_cache_check,
            use_shuffled_caption_variants=getattr(
                args, "use_shuffled_caption_variants", False
            ),
        )

    factory = dispatch_model_family(
        resolve_model_family(args),
        operation="training text cache strategy",
        handlers={
            "anima": anima_factory,
            "krea2_raw": krea2_factory,
            "z_image": z_image_factory,
        },
    )
    return factory()


def _z_image_latents_caching_strategy(*args):
    from library.models.z_image.strategy import ZImageLatentsCachingStrategy

    return ZImageLatentsCachingStrategy(*args)


def get_models_for_text_encoding(args, accelerator, text_encoders):
    if args.cache_text_encoder_outputs:
        return None  # no text encoders needed for encoding
    return text_encoders
