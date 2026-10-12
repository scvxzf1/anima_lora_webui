"""Invocation-local, inference-only prefix KV reuse for official Qwen 2.1."""

from __future__ import annotations

import torch
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21KVCache,
    QwenImage21Transformer2DModel,
)

from library.models.qwen_image_2_1.family import forward_for_loss


_STATIC_ADAPTERS = {
    ("networks.lora_modules.lora", "LoRAModule"),
    ("networks.plugins.loha.module", "LoHaModule"),
    ("networks.plugins.lokr.module", "LoKrModule"),
}

_CACHE_PROCESSORS = {
    ("diffusers.models.transformers.transformer_qwenimage21", "QwenImage21AttnProcessor"),
    ("diffusers.models.transformers.transformer_qwenimage21", "QwenImage21FlexAttnProcessor"),
    ("library.models.qwen_image_2_1.attention_backend", "QwenImage21FlashAttnProcessor"),
    ("library.models.qwen_image_2_1.packed_qkv", "PackedQwenImage21AttnProcessor"),
}


def _condition_identity(hiddens, mask, references, slots):
    """Guard an invocation's fixed tensors against replacement or mutation."""
    refs = (references,) if isinstance(references, torch.Tensor) else tuple(references or ())
    signature = []
    for value in (hiddens, mask, *refs, slots):
        try:
            version = value._version if value is not None else None
        except RuntimeError:  # Inference tensors do not expose a version counter.
            version = None
        signature.append((id(value), version))
    return tuple(signature)


def supports_preview_kv(dit, forward) -> bool:
    """Fail closed for custom forwards, training/dropout and dynamic adapters."""
    if forward is not forward_for_loss or not isinstance(dit, QwenImage21Transformer2DModel):
        return False
    if dit.training or not dit.config.causal_condition:
        return False
    for module in dit.modules():
        if module._forward_hooks or module._forward_pre_hooks:
            return False
        processor = getattr(module, "processor", None)
        if processor is not None and (type(processor).__module__, type(processor).__name__) not in _CACHE_PROCESSORS:
            return False
        if not isinstance(module, torch.nn.Linear):
            continue
        owner = getattr(module.forward, "__self__", None)
        if owner is module:
            continue
        if owner is None or (type(owner).__module__, type(owner).__name__) not in _STATIC_ADAPTERS:
            return False
        if owner.training:
            return False
    return True


class PreviewKVForward:
    """One condition branch's official extract/decode cache; never attached to DiT.

    The first call uses the training family's existing layout validator. Its
    callable captures those validated static inputs without changing the
    training path. Decode has a different output contract: target tokens only.
    Compiled block forwards remain installed and receive the official cache.
    """

    def __init__(self, dit):
        self.dit = dit
        self.cache = QwenImage21KVCache(len(dit.transformer_blocks))
        self.inputs = None
        self.reference_tokens = None
        self.latent_shape = None
        self.condition_identity = None

    def __call__(self, latents, hiddens, mask, sigmas, *, reference_latents=None, image_slot_mask=None):
        if torch.is_grad_enabled():
            raise RuntimeError("Qwen preview KV cache is inference-only; disable gradients")
        if self.cache is None:
            raise RuntimeError("Qwen preview KV cache has been disposed")
        if self.inputs is None:
            return self._extract(latents, hiddens, mask, sigmas, reference_latents, image_slot_mask)
        if tuple(latents.shape) != self.latent_shape:
            raise ValueError("Qwen preview cached target latent shape changed")
        if _condition_identity(hiddens, mask, reference_latents, image_slot_mask) != self.condition_identity:
            raise ValueError("Qwen preview KV conditions changed; create a fresh cache")
        tokens = latents.flatten(2).transpose(1, 2)
        if self.reference_tokens is not None:
            tokens = torch.cat([self.reference_tokens, tokens], dim=1)
        output = self.dit(
            **self.inputs, hidden_states=tokens, timestep=sigmas.to(dtype=tokens.dtype),
            kv_cache=self.cache, kv_cache_mode="cached",
        )[0]
        batch, channels, height, width = self.latent_shape
        if output.shape != (batch, height * width, channels):
            raise ValueError(f"Qwen Image 2.1 cached DiT output shape mismatch: {tuple(output.shape)}")
        return output.transpose(1, 2).reshape(self.latent_shape)

    def _extract(self, latents, hiddens, mask, sigmas, references, slots):
        def extract(**kwargs):
            self.inputs = {key: value for key, value in kwargs.items()
                           if key not in {"hidden_states", "timestep"}}
            target_count = latents.shape[-2] * latents.shape[-1]
            packed = kwargs["hidden_states"]
            if packed.shape[1] > target_count:
                self.reference_tokens = packed[:, :-target_count].clone()
            return self.dit(**kwargs, kv_cache=self.cache, kv_cache_mode="extract")

        result = forward_for_loss(
            extract, latents, hiddens, mask, sigmas,
            reference_latents=references, image_slot_mask=slots,
        )
        self.latent_shape = tuple(latents.shape)
        self.condition_identity = _condition_identity(hiddens, mask, references, slots)
        return result

    def close(self):
        """Release tensors even if a captured traceback keeps this helper alive."""
        if self.cache is not None:
            for layer in self.cache.layer_caches:
                layer.k = layer.v = None
        self.cache = self.inputs = self.reference_tokens = self.latent_shape = None
        self.condition_identity = None
