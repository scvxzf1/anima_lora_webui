"""CPU-side batch reshaping before the shared process_batch_inner path."""

from __future__ import annotations


def split_cached_text_encoder_outputs(batch: dict) -> dict:
    """Surface caption_dropout_rates / prior_crossattn_emb from the cache list.

    The cached text-encoder outputs list arrives as
    [..., optional prior_crossattn_emb, caption_dropout_rates] from the
    dataset (see strategy.py cache layout). Split the trailing aux tensors
    off so the inner path sees the canonical 4- or 5-element conds list.
    Doing it here on CPU avoids cloning prompt_embeds / crossattn_emb on
    the critical path before the H2D copy.
    """
    text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
    if text_encoder_outputs_list is None:
        return batch

    caption_dropout_rates = text_encoder_outputs_list[-1]
    encoder_outputs = text_encoder_outputs_list[:-1]
    prior_crossattn_emb = None
    if len(encoder_outputs) == 6:
        prior_crossattn_emb = encoder_outputs[-1]
        encoder_outputs = encoder_outputs[:-1]
    # Shallow copy so the original list (with rates appended) stays
    # intact for validation's per-sigma loop that reuses the batch.
    batch = {
        **batch,
        "text_encoder_outputs_list": encoder_outputs,
        "caption_dropout_rates": caption_dropout_rates,
    }
    if prior_crossattn_emb is not None:
        batch["prior_crossattn_emb"] = prior_crossattn_emb
    return batch
