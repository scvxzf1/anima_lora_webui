"""Anima text / latent strategy factories used by AnimaTrainer."""

from __future__ import annotations

import torch

from library.anima import strategy as strategy_anima


def get_tokenize_strategy(args):
    return strategy_anima.AnimaTokenizeStrategy(
        qwen3_path=args.qwen3,
        t5_tokenizer_path=args.t5_tokenizer_path,
        qwen3_max_length=args.qwen3_max_token_length,
        t5_max_length=args.t5_max_token_length,
    )


def get_tokenizers(tokenize_strategy: strategy_anima.AnimaTokenizeStrategy):
    return [tokenize_strategy.qwen3_tokenizer]


def get_latents_caching_strategy(args):
    return strategy_anima.AnimaLatentsCachingStrategy(
        args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
    )


def get_text_encoding_strategy(args):
    return strategy_anima.AnimaTextEncodingStrategy()


def get_text_encoder_outputs_caching_strategy(args, weight_dtype: torch.dtype):
    if not args.cache_text_encoder_outputs:
        return None
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


def get_models_for_text_encoding(args, accelerator, text_encoders):
    if args.cache_text_encoder_outputs:
        return None  # no text encoders needed for encoding
    return text_encoders
