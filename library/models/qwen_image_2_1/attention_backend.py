"""Qwen Image 2.1 attention backend selection."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import torch


QWEN_IMAGE_21_ATTENTION_MODES = frozenset({"torch", "flash"})


@dataclass(frozen=True)
class _DiffusersAttentionApi:
    backend_names: type[Enum]
    processor_type: type[object]
    prepare_qkv: Callable
    dispatch_attention: Callable


@dataclass
class _FlashMaskMetadata:
    mask: torch.Tensor | None
    mask_version: int | None
    lengths: torch.Tensor
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    valid_indices: torch.Tensor
    max_key_length: int
    all_valid: bool


_FLASH_MASK_METADATA_CACHE: OrderedDict[tuple[object, ...], _FlashMaskMetadata] = OrderedDict()
_FLASH_MASK_METADATA_CACHE_MAXSIZE = 32


def _cache_flash_mask_metadata(
    valid: torch.Tensor | None,
    *,
    batch: int,
    query_length: int,
    key_length: int,
    device: torch.device,
) -> _FlashMaskMetadata:
    """Reuse mask-derived varlen metadata across layers and checkpoint recomputes."""
    if valid is None:
        cache_key = ("all", device, batch, query_length, key_length)
        cached = _FLASH_MASK_METADATA_CACHE.get(cache_key)
        if cached is not None:
            _FLASH_MASK_METADATA_CACHE.move_to_end(cache_key)
            return cached
        lengths = torch.full((batch,), key_length, dtype=torch.int32, device=device)
        metadata = _FlashMaskMetadata(
            mask=None,
            mask_version=None,
            lengths=lengths,
            cu_q=torch.arange(batch + 1, device=device, dtype=torch.int32) * query_length,
            cu_k=torch.arange(batch + 1, device=device, dtype=torch.int32) * key_length,
            valid_indices=torch.arange(batch * key_length, device=device, dtype=torch.long),
            max_key_length=key_length,
            all_valid=True,
        )
    else:
        if valid.shape != (batch, key_length):
            raise ValueError(
                "Qwen Image 2.1 FlashAttention metadata expects a normalized key-valid mask "
                f"of shape {(batch, key_length)}, got {tuple(valid.shape)}"
            )
        mask_version = int(getattr(valid, "_version", 0))
        cache_key = ("mask", id(valid), batch, query_length, key_length)
        cached = _FLASH_MASK_METADATA_CACHE.get(cache_key)
        if cached is not None and cached.mask is valid and cached.mask_version == mask_version:
            _FLASH_MASK_METADATA_CACHE.move_to_end(cache_key)
            return cached

        lengths = valid.sum(dim=1, dtype=torch.int32)
        min_length = int(lengths.min().item())
        if min_length == 0:
            raise ValueError("Qwen Image 2.1 FlashAttention does not accept empty key sequences")
        cu_k = torch.zeros(batch + 1, device=device, dtype=torch.int32)
        cu_k[1:] = lengths.cumsum(dim=0)
        metadata = _FlashMaskMetadata(
            mask=valid,
            mask_version=mask_version,
            lengths=lengths,
            cu_q=torch.arange(batch + 1, device=device, dtype=torch.int32) * query_length,
            cu_k=cu_k,
            valid_indices=valid.reshape(-1).nonzero(as_tuple=True)[0],
            max_key_length=int(lengths.max().item()),
            all_valid=bool(int(lengths.sum().item()) == batch * key_length),
        )

    _FLASH_MASK_METADATA_CACHE[cache_key] = metadata
    _FLASH_MASK_METADATA_CACHE.move_to_end(cache_key)
    while len(_FLASH_MASK_METADATA_CACHE) > _FLASH_MASK_METADATA_CACHE_MAXSIZE:
        _FLASH_MASK_METADATA_CACHE.popitem(last=False)
    return metadata


def clear_qwen_image_2_1_attention_metadata_cache() -> None:
    """Clear cached mask metadata after a caller discards a shape family."""
    _FLASH_MASK_METADATA_CACHE.clear()


def normalize_qwen_image_2_1_attention_mode(value: object) -> str:
    mode = str(value or "torch").strip().lower().replace("-", "_")
    if mode in {"sdpa", "native"}:
        return "torch"
    if mode not in QWEN_IMAGE_21_ATTENTION_MODES:
        raise ValueError(
            "Qwen Image 2.1 attn_mode supports only 'torch' (native SDPA) and "
            f"'flash' (FlashAttention varlen); got {value!r}"
        )
    return mode


def _load_diffusers_attention_api() -> _DiffusersAttentionApi:
    try:
        from diffusers.models import AttentionBackendName
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from diffusers.models.transformers.transformer_qwenimage21 import (
            QwenImage21AttnProcessor,
            _qwenimage21_prepare_qkv,
        )
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Qwen Image 2.1 FlashAttention requires the Diffusers Qwen 2.1 "
            "attention processor and dispatcher"
        ) from exc
    return _DiffusersAttentionApi(
        backend_names=AttentionBackendName,
        processor_type=QwenImage21AttnProcessor,
        prepare_qkv=_qwenimage21_prepare_qkv,
        dispatch_attention=dispatch_attention_fn,
    )


class QwenImage21FlashAttnProcessor:
    """Use Flash varlen for image queries while preserving causal text attention."""

    _parallel_config = None

    def __init__(self, api: _DiffusersAttentionApi) -> None:
        self._api = api
        self._native_backend = api.backend_names.NATIVE
        self._flash_backend = api.backend_names.FLASH_VARLEN
        self._attention_backend = self._flash_backend

    def _attend(self, query, key, value, attn_mask, backend):
        if backend == self._flash_backend:
            return _flash_varlen_attention(query, key, value, mask=attn_mask)
        return self._api.dispatch_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            backend=self._native_backend,
            parallel_config=self._parallel_config,
        )

    def _attend_prefix_segment(self, query, key, value, segment, key_valid):
        start, end, is_text = segment
        segment_mask = None
        if is_text:
            segment_len = end - start
            segment_mask = torch.cat(
                [
                    torch.ones(segment_len, start, dtype=torch.bool, device=query.device),
                    torch.tril(
                        torch.ones(segment_len, segment_len, dtype=torch.bool, device=query.device)
                    ),
                ],
                dim=1,
            )[None, None]
        if key_valid is not None:
            valid_mask = key_valid[:, :end]
            segment_mask = (
                valid_mask
                if segment_mask is None
                else (segment_mask & valid_mask[:, None, None, :])
            )

        backend = self._native_backend if is_text else self._flash_backend
        return self._attend(query[:, start:end], key[:, :end], value[:, :end], segment_mask, backend)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask=None,
        rotary_emb=None,
        layer_cache=None,
        kv_cache_mode=None,
        cache_write_slice=None,
        segments=None,
        key_valid=None,
    ) -> torch.Tensor:
        query, key, value, seq_len_q = self._api.prepare_qkv(
            attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice
        )

        if segments is None:
            backend = (
                self._flash_backend
                if _is_key_padding_mask(attention_mask, batch=query.shape[0], key_length=key.shape[1])
                else self._native_backend
            )
            hidden_states = self._attend(query, key, value, attention_mask, backend)
        else:
            prefix_len = segments[-1][1] if segments else 0
            outputs = [
                self._attend_prefix_segment(query, key, value, segment, key_valid)
                for segment in segments
            ]
            target_mask = key_valid
            outputs.append(
                self._attend(query[:, prefix_len:], key, value, target_mask, self._flash_backend)
            )
            hidden_states = torch.cat(outputs, dim=1)

        hidden_states = hidden_states[:, :seq_len_q]
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


def _is_key_padding_mask(mask: object, *, batch: int, key_length: int) -> bool:
    if mask is None:
        return True
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        return False
    if mask.ndim == 1:
        return batch == 1 and mask.shape[0] == key_length
    if mask.ndim == 2:
        return tuple(mask.shape) == (batch, key_length)
    return (
        mask.ndim == 4
        and mask.shape[0] == batch
        and mask.shape[1] == 1
        and mask.shape[2] == 1
        and mask.shape[3] == key_length
    )


def _normalize_key_valid_mask(mask, *, batch: int, key_length: int, device) -> torch.Tensor:
    if mask is None:
        return torch.ones((batch, key_length), dtype=torch.bool, device=device)
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise ValueError("Qwen Image 2.1 FlashAttention requires a boolean key-valid mask")
    if mask.ndim == 1 and batch == 1 and mask.shape[0] == key_length:
        return mask[None, :]
    if mask.ndim == 2 and tuple(mask.shape) == (batch, key_length):
        return mask
    if (
        mask.ndim == 4
        and mask.shape[0] == batch
        and mask.shape[1] == 1
        and mask.shape[2] == 1
        and mask.shape[3] == key_length
    ):
        return mask[:, 0, 0]
    raise ValueError(
        "Qwen Image 2.1 FlashAttention supports only a key-padding mask shaped "
        f"(B,K) or (B,1,1,K); got {tuple(mask.shape)}"
    )


@torch.compiler.disable(recursive=True)
def _flash_varlen_attention(query, key, value, *, mask):
    """Pack the actual valid K/V positions outside compiled checkpoint blocks."""
    from networks import attention_dispatch

    if not attention_dispatch.flash_attn_available_for_dtype(query.dtype):
        raise RuntimeError(
            f"Qwen Image 2.1 FlashAttention does not support Q/K/V dtype {query.dtype}"
        )
    if query.device.type != "cuda" or key.device != query.device or value.device != query.device:
        raise RuntimeError("Qwen Image 2.1 FlashAttention requires Q/K/V on the same CUDA device")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("Qwen Image 2.1 FlashAttention requires matching Q/K/V dtypes")
    if query.shape[0] != key.shape[0] or key.shape != value.shape:
        raise ValueError("Qwen Image 2.1 FlashAttention requires matching K/V and Q/K batches")
    if query.shape[2:] != key.shape[2:]:
        raise ValueError("Qwen Image 2.1 FlashAttention requires matching head count and dimension")

    batch, query_length, heads, head_dim = query.shape
    key_length = key.shape[1]
    valid = None if mask is None else _normalize_key_valid_mask(
        mask, batch=batch, key_length=key_length, device=query.device
    )
    metadata = _cache_flash_mask_metadata(
        valid,
        batch=batch,
        query_length=query_length,
        key_length=key_length,
        device=query.device,
    )
    q_packed = query.reshape(batch * query_length, heads, head_dim).contiguous()
    if metadata.all_valid:
        k_packed = key.reshape(batch * key_length, heads, head_dim).contiguous()
        v_packed = value.reshape(batch * key_length, heads, head_dim).contiguous()
    else:
        flat_key = key.reshape(batch * key_length, heads, head_dim)
        flat_value = value.reshape(batch * key_length, heads, head_dim)
        k_packed = flat_key.index_select(0, metadata.valid_indices)
        v_packed = flat_value.index_select(0, metadata.valid_indices)
    output = attention_dispatch.flash_attn_varlen_func(
        q_packed,
        k_packed,
        v_packed,
        metadata.cu_q,
        metadata.cu_k,
        query_length,
        metadata.max_key_length,
        dropout_p=0.0,
    )
    return output.reshape(batch, query_length, heads, head_dim)


def prepare_qwen_image_2_1_attention(
    model: object,
    mode: object,
    *,
    dtype: torch.dtype | None,
) -> str:
    """Configure packed FlashAttention without changing Diffusers global state."""

    normalized = normalize_qwen_image_2_1_attention_mode(mode)
    if normalized == "torch":
        return normalized
    if dtype != torch.bfloat16:
        raise RuntimeError(
            f"Qwen Image 2.1 attn_mode='flash' is validated only for bf16 compute; got {dtype}"
        )

    try:
        from networks import attention_dispatch

        if not attention_dispatch.flash_attn_available_for_dtype(dtype):
            raise RuntimeError("installed FlashAttention provider does not support BF16")
    except (ImportError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "Qwen Image 2.1 attn_mode='flash' requires a compatible "
            "FlashAttention 2 varlen provider with BF16 support"
        ) from exc

    api = _load_diffusers_attention_api()

    modules = getattr(model, "modules", None)
    if not callable(modules):
        raise TypeError("Qwen Image 2.1 transformer does not expose modules()")
    configured = 0
    for module in modules():
        processor = getattr(module, "processor", None)
        if not isinstance(processor, api.processor_type):
            continue
        setter = getattr(module, "set_processor", None)
        if not callable(setter):
            raise TypeError("Qwen Image 2.1 attention module cannot set its processor")
        setter(QwenImage21FlashAttnProcessor(api))
        configured += 1
    if configured == 0:
        raise RuntimeError(
            "Qwen Image 2.1 transformer has no supported attention processors; "
            "cannot configure FlashAttention"
        )
    return normalized


__all__ = [
    "QWEN_IMAGE_21_ATTENTION_MODES",
    "QwenImage21FlashAttnProcessor",
    "clear_qwen_image_2_1_attention_metadata_cache",
    "normalize_qwen_image_2_1_attention_mode",
    "prepare_qwen_image_2_1_attention",
]
