"""Text-encoder output caching helpers extracted from AnimaTrainer.

Preserves live-encoding vs disk-cache behavior and sample-prompt TE caching.
"""

from __future__ import annotations

import logging

import torch
from accelerate import Accelerator

from library import train_util
from library.anima import text_strategies
from library.datasets import DatasetGroup
from library.runtime.device import clean_memory_on_device
from library.training.train_bootstrap import sample_preview_enabled

logger = logging.getLogger(__name__)


def cache_text_encoder_outputs_if_needed(
    trainer,
    args,
    accelerator: Accelerator,
    text_encoders,
    dataset: DatasetGroup,
):
    if not args.cache_text_encoder_outputs:
        # Live-encoding mode (e.g. IP-Adapter cache_text_encoder_outputs=false):
        # move the text encoder to device for per-step encoding.
        text_encoders[0].to(accelerator.device)
        return

    # With caching on, the on-disk cache is guaranteed complete (asserted in
    # train(), including the LLM adapter's crossattn_emb outputs, which
    # preprocess writes). The dataset thus never needs encoding here — run
    # the pass with no model purely to populate
    # ImageInfo.text_encoder_outputs_npz (forms no batches).
    dataset.new_cache_text_encoder_outputs([None], accelerator)

    # The text encoder is in memory only to encode sample prompts when
    # sampling will actually run. It is None when no preview needs it.
    if text_encoders[0] is not None and sample_preview_enabled(args):
        logger.info(
            f"cache Text Encoder outputs for sample prompts: {args.sample_prompts}"
        )
        logger.info("move text encoder to gpu")
        text_encoders[0].to(accelerator.device)

        tokenize_strategy = text_strategies.TokenizeStrategy.get_strategy()
        text_encoding_strategy = text_strategies.TextEncodingStrategy.get_strategy()

        prompts = train_util.load_prompts(args.sample_prompts)
        trainer.sample_prompts_snapshot = [dict(prompt) for prompt in prompts]
        sample_prompts_te_outputs = {}
        with accelerator.autocast(), torch.no_grad():
            for prompt_dict in prompts:
                for p in [
                    prompt_dict.get("prompt", ""),
                    prompt_dict.get("negative_prompt", ""),
                ]:
                    if p not in sample_prompts_te_outputs:
                        logger.info(f"  cache TE outputs for: {p}")
                        tokens_and_masks = tokenize_strategy.tokenize(p)
                        sample_prompts_te_outputs[p] = (
                            text_encoding_strategy.encode_tokens(
                                tokenize_strategy,
                                text_encoders,
                                tokens_and_masks,
                            )
                        )
        trainer.sample_prompts_te_outputs = sample_prompts_te_outputs

        logger.info("move text encoder back to cpu")
        text_encoders[0].to("cpu")
        clean_memory_on_device(accelerator.device)

    accelerator.wait_for_everyone()

