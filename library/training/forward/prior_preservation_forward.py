"""Prior-preservation reference forward for no-extra-dataset regularization."""

from __future__ import annotations

from typing import Any, Mapping

import torch


def _prepare_block_swap_before_reference_forward(anima_call: Any) -> None:
    """在同一步内的参考 forward 前重置 DiT 块交换状态。

    主训练 forward 结束后，前段交换块会停在 CPU，等待 backward hook 预取。
    prior-preservation 会在 backward 前额外跑一次 no-grad forward，因此必须先恢复
    块交换的 forward 起始布局；否则第二次 forward 会遇到 GPU activation + CPU
    block weight，torch.compile 下会表现为 fake tensor 设备不一致。
    """
    model = getattr(anima_call, "module", anima_call)
    if not getattr(model, "blocks_to_swap", 0):
        return
    prepare = getattr(model, "prepare_block_swap_before_forward", None)
    if not callable(prepare):
        return
    try:
        prepare(free_cache=False)
    except TypeError:
        prepare()


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

    _prepare_block_swap_before_reference_forward(anima_call)

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
