"""Two-stage, one-microbatch Anima pipeline probe."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch


@dataclass
class PipelineCommunication:
    payload_bytes: int = 0
    calls: int = 0
    seconds: float = 0.0

    def transfer(self, fn, tensor: torch.Tensor) -> None:
        torch.cuda.synchronize(tensor.device)
        start = perf_counter()
        fn(tensor)
        torch.cuda.synchronize(tensor.device)
        self.seconds += perf_counter() - start
        self.payload_bytes += tensor.numel() * tensor.element_size()
        self.calls += 1

    def as_dict(self) -> dict[str, float | int]:
        return {
            "payload_bytes_per_rank": self.payload_bytes,
            "wire_bytes_per_rank": self.payload_bytes,
            "collective_calls": self.calls,
            "quantize_seconds": 0.0,
            "collective_seconds": self.seconds,
            "dequantize_seconds": 0.0,
            "communication_seconds": self.seconds,
        }


def prepare_block_inputs(model, x, timesteps, context, padding_mask):
    from networks import attention_dispatch

    hidden, rope = model.prepare_embedded_sequence(x, padding_mask=padding_mask)
    if timesteps.ndim == 1:
        timesteps = timesteps.unsqueeze(1)
    embedding, adaln = model.t_embedder(timesteps)
    embedding = model.t_embedding_norm(embedding)
    rope = tuple(value.to(hidden.dtype) for value in rope)
    params = attention_dispatch.AttentionParams.create_attention_params(
        model.attn_mode,
        model.attn_softmax_scale,
        v100_flash_stability=model.v100_flash_stability,
        debug_finite_checks=model.debug_finite_checks,
    )
    kwargs = {"rope_cos_sin": rope, "adaln_lora_B_T_3D": adaln, "use_fp32": False}
    return hidden, embedding, params, kwargs


def run_local_blocks(model, hidden, embedding, context, params, kwargs):
    return model._run_blocks(hidden, embedding, context, params, **kwargs)


def finish_model(model, hidden, embedding, adaln):
    projected = model.final_layer(
        hidden,
        embedding,
        adaln_lora_B_T_3D=adaln,
        use_fp32=False,
    )
    return model.unpatchify(projected)
