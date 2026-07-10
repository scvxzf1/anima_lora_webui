"""Training-time sample preview dispatch used by AnimaTrainer.sample_images."""

from __future__ import annotations

from library.anima import (
    training as anima_train_utils,
    text_strategies,
)


def sample_images(
    trainer,
    accelerator,
    args,
    epoch,
    global_step,
    device,
    vae,
    tokenizer,
    text_encoder,
    unet,
    network=None,
):
    text_encoders = (
        text_encoder if isinstance(text_encoder, list) else [text_encoder]
    )  # compatibility
    te = trainer.get_models_for_text_encoding(args, accelerator, text_encoders)
    qwen3_te = te[0] if te is not None else None

    text_encoding_strategy = text_strategies.TextEncodingStrategy.get_strategy()
    tokenize_strategy = text_strategies.TokenizeStrategy.get_strategy()
    anima_train_utils.sample_images(
        accelerator,
        args,
        epoch,
        global_step,
        unet,
        vae,
        qwen3_te,
        tokenize_strategy,
        text_encoding_strategy,
        trainer.sample_prompts_te_outputs,
        sample_prompts_snapshot=trainer.sample_prompts_snapshot,
        network=network,
    )
