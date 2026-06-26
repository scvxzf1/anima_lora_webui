"""Prior-preservation reference forward for no-extra-dataset regularization."""

from __future__ import annotations

from typing import Any, Mapping

import torch


def run_prior_preservation_forward(
    *,
    anima_call: Any,
    network: Any,
    noisy_model_input: torch.Tensor,
    timesteps: torch.Tensor,
    crossattn_emb: torch.Tensor,
    padding_mask: torch.Tensor,
    forward_kwargs: Mapping[str, Any],
) -> torch.Tensor:
    """Return base-model prediction with the trainable adapter temporarily zeroed.

    The caller supplies the exact same noisy latent and timestep used by the
    primary training forward. Only the text condition may differ, e.g. blank
    prompt preservation passes T5(""). The output stays 5D so it can be compared
    with the primary DiT prediction before train.py squeezes it.
    """
    if not hasattr(network, "set_multiplier"):
        raise ValueError(
            "prior preservation requires a network with set_multiplier(), "
            "so the adapter can be disabled for the base-model reference forward"
        )

    orig_mult = float(getattr(network, "multiplier", 1.0))
    network.set_multiplier(0.0)
    try:
        with torch.no_grad():
            return anima_call(
                noisy_model_input,
                timesteps,
                crossattn_emb,
                padding_mask=padding_mask,
                **forward_kwargs,
            )
    finally:
        network.set_multiplier(orig_mult)
