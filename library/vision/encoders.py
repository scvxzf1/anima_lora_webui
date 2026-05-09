"""Vision-encoder registry (originally for img2emb; reused live by IP-Adapter).

PE-Core-L14-336 (Meta Perception Encoder) is the sole supported encoder,
vendored at ``library/models/pe.py`` so we don't have to clone
perception_models or install xformers.

``encode(pixel_values)`` returns ``(last_hidden_state[B, T, D],
pooled[B, D_pool])``. ``T`` includes a CLS token at position 0 (PE-Core-L14-336
has ``use_cls_token=True``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from library.vision.buckets import BucketSpec, get_bucket_spec

logger = logging.getLogger(__name__)

# Repo root, used by the ``_default_*_model_id`` helpers to point at vendored
# checkpoints under ``models/``. encoders.py lives at
# ``library/vision/encoders.py`` so two ``parents`` jumps land on the repo
# root regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- shared output shim


class _EncoderOutput:
    """Minimal HF ``BaseModelOutput``-shaped container."""

    __slots__ = ("last_hidden_state", "pooler_output")

    def __init__(self, last_hidden_state: torch.Tensor, pooler_output: torch.Tensor):
        self.last_hidden_state = last_hidden_state
        self.pooler_output = pooler_output


# --------------------------------------------------------------------------- PE-Core-L14-336


def _default_pe_model_id() -> str:
    return str(REPO_ROOT / "models" / "pe" / "PE-Core-L14-336.pt")


class _PEProcessor:
    """PE's reference preprocessing: squash-resize + ToTensor + ``[0.5, 0.5,
    0.5]`` mean/std (i.e. map [0,1] to [-1, 1])."""

    _MEAN = (0.5, 0.5, 0.5)
    _STD = (0.5, 0.5, 0.5)

    def __init__(self, image_size):
        from torchvision import transforms

        size_hw = (image_size, image_size) if isinstance(image_size, int) else (
            int(image_size[0]), int(image_size[1])
        )
        self.image_size = size_hw
        self.transform = transforms.Compose(
            [
                transforms.Resize(size_hw, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(self._MEAN, self._STD),
            ]
        )

    def __call__(self, images, return_tensors: str = "pt"):
        assert return_tensors == "pt"
        if not isinstance(images, (list, tuple)):
            images = [images]
        return {
            "pixel_values": torch.stack([self.transform(img) for img in images], dim=0)
        }


class _PEEncoder:
    """Wraps the vendored PEVisionTransformer to produce
    ``(last_hidden_state, pooler_output)``."""

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, pixel_values: torch.Tensor) -> _EncoderOutput:
        feats, pooled = self.inner.encode(pixel_values)
        return _EncoderOutput(last_hidden_state=feats, pooler_output=pooled)


def _load_pe_encoder(device: torch.device, model_id: str) -> _PEEncoder:
    """Build the vendored PE vision tower and load Meta's official ``.pt``
    checkpoint (CLIP-format). ``model_id`` is a local file path to the ``.pt``,
    fetched via ``make download-pe``."""
    from library.models.pe import build_pe_vision

    ckpt_path = Path(model_id)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"PE checkpoint not found at {ckpt_path}. "
            f"Run `make download-pe` to fetch facebook/PE-Core-L14-336."
        )
    logger.info(f"Loading PE-Core-L14-336 from {ckpt_path}")
    model = build_pe_vision("PE-Core-L14-336")
    model.load_pe_checkpoint(str(ckpt_path), verbose=True)
    model = model.to(dtype=torch.bfloat16, device=device).eval()
    model.requires_grad_(False)
    return _PEEncoder(model)


# --------------------------------------------------------------------------- registry


@dataclass(frozen=True)
class EncoderInfo:
    name: str
    bucket_spec: BucketSpec
    d_enc: int
    d_pool: int
    default_model_id: Callable[[], str]
    processor_factory: Callable[..., object]  # (image_size) -> processor
    loader: Callable[[torch.device, str], object]  # (device, model_id) -> encoder

    def t_max_tokens(self) -> int:
        return self.bucket_spec.t_max_tokens


_REGISTRY: dict[str, EncoderInfo] = {
    "pe": EncoderInfo(
        name="pe",
        bucket_spec=get_bucket_spec("pe"),
        d_enc=1024,
        d_pool=1024,
        default_model_id=_default_pe_model_id,
        processor_factory=_PEProcessor,
        loader=_load_pe_encoder,
    ),
}


def get_encoder_info(name: str) -> EncoderInfo:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown encoder {name!r}; available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def available_encoders() -> list[str]:
    return sorted(_REGISTRY)
