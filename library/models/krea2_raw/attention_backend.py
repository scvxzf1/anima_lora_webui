"""Krea-2 attention backends shared by training and inference."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel


KREA2_ATTENTION_MODES = frozenset({"torch", "flash"})


def normalize_krea2_attention_mode(value: object) -> str:
    mode = str(value or "torch").strip().lower().replace("-", "_")
    if mode in {"sdpa", "cudnn", "cudnn_sdpa"}:
        return "torch"
    if mode not in KREA2_ATTENTION_MODES:
        raise ValueError(
            "Krea-2 attn_mode supports only 'torch' (cuDNN SDPA) and "
            f"'flash' (packed FlashAttention varlen); got {value!r}"
        )
    return mode


def validate_krea2_attention_mode(
    mode: object,
    *,
    dtype: torch.dtype | None,
    compile_enabled: bool = False,
) -> str:
    """Validate a Krea-2 attention mode before loading the large DiT."""

    normalized = normalize_krea2_attention_mode(mode)
    if normalized == "flash":
        from networks import attention_dispatch

        if dtype not in {torch.float16, torch.bfloat16}:
            raise RuntimeError(
                "Krea-2 attn_mode='flash' requires fp16 or bf16 compute; "
                f"got {dtype}"
            )
        if not attention_dispatch.flash_attn_available_for_dtype(dtype):
            provider = (
                "flash-attention-v100 (FP16 only)"
                if attention_dispatch.flash_attn_v100_provider
                else "FlashAttention 2"
            )
            raise RuntimeError(
                f"Krea-2 attn_mode='flash' requires {provider} with {dtype} support"
            )
        if compile_enabled:
            # Boolean packing produces a symbolic packed-token length. This is
            # enabled only for a run that explicitly selected the Flash backend.
            torch._dynamo.config.capture_dynamic_output_shape_ops = True

    return normalized


def prepare_krea2_attention(
    model: object,
    mode: object,
    *,
    dtype: torch.dtype | None,
    compile_enabled: bool = False,
) -> str:
    """Validate and install a Krea-2 attention mode before optional compile."""

    normalized = validate_krea2_attention_mode(
        mode,
        dtype=dtype,
        compile_enabled=compile_enabled,
    )

    setter = getattr(model, "set_attention_mode", None)
    if not callable(setter):
        raise TypeError("Krea-2 model does not expose set_attention_mode()")
    setter(normalized)
    return normalized


def run_krea2_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    mask: Tensor | None = None,
    scale: float | None = None,
    gqa: bool = False,
    mode: str = "torch",
) -> Tensor:
    """Run Krea-2 self-attention with the model's padding-mask contract.

    This is an internal DiT dispatch, not a general masked-attention API.
    ``mask`` must come from ``dit._mask``: a boolean outer product of valid
    tokens with at least one valid token per batch row. Causal or arbitrary
    four-dimensional masks are intentionally unsupported by the varlen path.
    """
    if mode == "flash":
        return _flash_varlen_attention(q, k, v, mask=mask, scale=scale)
    if mode != "torch":
        raise RuntimeError(f"unconfigured Krea-2 attention mode: {mode!r}")
    return _cudnn_attention(q, k, v, mask=mask, scale=scale, gqa=gqa)


def _cudnn_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    mask: Tensor | None,
    scale: float | None,
    gqa: bool,
) -> Tensor:
    try:
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, scale=scale, enable_gqa=gqa
            )
    except (RuntimeError, NotImplementedError):
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=scale, enable_gqa=gqa
        )
    return rearrange(output, "B H L D -> B L (H D)")


@torch.compiler.disable(recursive=True)
def _flash_varlen_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    mask: Tensor | None,
    scale: float | None,
) -> Tensor:
    """Run dynamic token packing outside compiled checkpointed blocks.

    Boolean packing and the padded-output scatter have data-dependent shapes.
    AOTAutograd can compile the forward, but checkpoint recomputation lowers
    ``IndexPutBackward`` through ``aten.nonzero`` and fails in Inductor.  The
    FlashAttention kernel still runs normally; only its small packing wrapper
    remains eager while the surrounding DiT block stays compiled.
    """
    from networks import attention_dispatch

    _validate_flash_qkv(q, k, v)
    if not attention_dispatch.flash_attn_available_for_dtype(q.dtype):
        raise RuntimeError(
            f"Krea-2 FlashAttention does not support Q/K/V dtype {q.dtype}"
        )
    flash_varlen: Callable = attention_dispatch.flash_attn_varlen_func

    batch, _heads, sequence, _dim = q.shape
    if mask is None:
        valid = torch.ones(batch, sequence, device=q.device, dtype=torch.bool)
    else:
        expected_mask_shape = (batch, 1, sequence, sequence)
        if tuple(mask.shape) != expected_mask_shape or mask.dtype != torch.bool:
            raise ValueError(
                "Krea-2 FlashAttention expects a boolean (B,1,L,L) "
                f"outer-product mask; got shape={tuple(mask.shape)}, dtype={mask.dtype}"
            )
        valid = mask[:, 0].diagonal(dim1=-2, dim2=-1)

    lengths = valid.sum(dim=1, dtype=torch.int32)
    cu_seqlens = torch.zeros(batch + 1, device=q.device, dtype=torch.int32)
    cu_seqlens[1:] = lengths.cumsum(dim=0)
    q_packed = rearrange(q, "b h l d -> b l h d")[valid].contiguous()
    k_packed = rearrange(k, "b h l d -> b l h d")[valid].contiguous()
    v_packed = rearrange(v, "b h l d -> b l h d")[valid].contiguous()
    output_packed = flash_varlen(
        q_packed,
        k_packed,
        v_packed,
        cu_seqlens,
        cu_seqlens,
        sequence,
        sequence,
        softmax_scale=scale,
    )
    output = torch.zeros(
        batch,
        sequence,
        q.shape[1],
        q.shape[-1],
        device=q.device,
        dtype=q.dtype,
    )
    output[valid] = output_packed
    return rearrange(output, "b l h d -> b l (h d)")


def _validate_flash_qkv(q: Tensor, k: Tensor, v: Tensor) -> None:
    tensors = {"q": q, "k": k, "v": v}
    for name, tensor in tensors.items():
        if tensor.ndim != 4:
            raise ValueError(
                f"Krea-2 FlashAttention expects {name} in (B,H,L,D); "
                f"got shape={tuple(tensor.shape)}"
            )
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(
            "Krea-2 FlashAttention requires matching Q/K/V dtypes; "
            f"got q={q.dtype}, k={k.dtype}, v={v.dtype}"
        )
    if q.device != k.device or q.device != v.device:
        raise ValueError("Krea-2 FlashAttention requires Q/K/V on the same device")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("Krea-2 FlashAttention requires matching Q/K/V batch sizes")
    if q.shape[2:] != k.shape[2:] or q.shape[2:] != v.shape[2:]:
        raise ValueError(
            "Krea-2 FlashAttention requires matching Q/K/V sequence and head dimensions"
        )
    if k.shape[1] != v.shape[1] or k.shape[1] == 0 or q.shape[1] % k.shape[1] != 0:
        raise ValueError(
            "Krea-2 FlashAttention GQA requires equal K/V heads and "
            "Q heads divisible by K/V heads"
        )


__all__ = [
    "KREA2_ATTENTION_MODES",
    "normalize_krea2_attention_mode",
    "prepare_krea2_attention",
    "validate_krea2_attention_mode",
]
