"""Lazy, frozen Qwen3.5 runtime with explicit NF4/BF16 placement."""

from __future__ import annotations

import gc
from types import SimpleNamespace

import torch

from .cache_policy import resolve_text_encoder_path, validate_quantization


class MossTextEncoderRuntime(torch.nn.Module):
    """Keep a tiny parameter for generic cache callers; load only on forward.

    ``to('cpu')`` releases the loaded encoder, including packed NF4 weights.
    Subsequent forwards reload from the same local checkpoint. Dtype changes
    never reach quantized parameters.
    """

    def __init__(self, path: str, *, dtype: torch.dtype, device, quantization: str):
        super().__init__()
        if dtype != torch.bfloat16:
            raise ValueError("MOSS text encoding requires BF16 compute, including NF4")
        self.path = resolve_text_encoder_path(path)
        self.quantization = validate_quantization(quantization)
        self.register_parameter("_anchor", torch.nn.Parameter(torch.empty(0, dtype=dtype), requires_grad=False))
        self._encoder = None
        self.config = SimpleNamespace(use_cache=False, hidden_size=4096)
        self.to(device)
        self.eval()

    @property
    def device(self):
        return self._anchor.device

    @property
    def dtype(self):
        return self._anchor.dtype

    def train(self, mode: bool = True):
        return super().train(False)

    def requires_grad_(self, requires_grad: bool = True):
        if requires_grad:
            raise ValueError("MOSS text encoder is frozen; text-encoder training is unsupported")
        return super().requires_grad_(False)

    def release(self) -> None:
        self._encoder = None
        gc.collect()

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, _memory_format = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is not None and dtype != torch.bfloat16:
            raise ValueError("MOSS text encoder supports only BF16 compute; packed NF4 weights cannot be cast")
        device = self.device if device is None else torch.device(device)
        if device.type not in {"cpu", "cuda"}:
            raise ValueError(f"MOSS text encoder device must be cpu or cuda, got {device}")
        if device != self.device or device.type == "cpu":
            self.release()
        self._anchor = torch.nn.Parameter(
            self._anchor.to(device=device, dtype=torch.bfloat16, non_blocking=non_blocking),
            requires_grad=False,
        )
        return self

    def cpu(self):
        return self.to("cpu")

    def cuda(self, device=None):
        return self.to(torch.device("cuda", device) if device is not None else "cuda")

    def _load(self):
        if self._encoder is not None:
            return self._encoder
        if self.quantization == "nf4" and self.device.type != "cuda":
            raise RuntimeError("MOSS NF4 text encoding requires CUDA placement; use quantization='bf16' explicitly for CPU")
        try:
            from transformers import Qwen3_5Model
        except ImportError as exc:
            raise RuntimeError("MOSS requires Transformers with Qwen3_5Model support (model config uses 5.12.1)") from exc
        kwargs = {"local_files_only": True, "dtype": torch.bfloat16}
        if self.quantization == "nf4":
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("MOSS NF4 requires a CUDA device supporting BF16 compute")
            try:
                import bitsandbytes  # noqa: F401
                from transformers import BitsAndBytesConfig
                import accelerate  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("MOSS NF4 requires bitsandbytes and accelerate; no precision fallback is applied") from exc
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = {"": str(self.device)}
        model = Qwen3_5Model.from_pretrained(self.path, **kwargs)
        if self.quantization == "bf16":
            model.to(device=self.device, dtype=torch.bfloat16)
        model.config.use_cache = False
        self._encoder = model.eval().requires_grad_(False)
        return self._encoder

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self._load()(*args, **kwargs)


def load_moss_text_encoder(path, *, dtype, device="cpu", quantization="nf4"):
    """Return a frozen lazy module; never download or load another checkpoint."""
    return MossTextEncoderRuntime(path, dtype=dtype, device=device, quantization=quantization)
