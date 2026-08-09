from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
import torch
from PIL import Image

from library.inference import output


def _args(**overrides) -> Namespace:
    base = dict(
        seed=123456,
        sampler="euler",
        infer_steps=28,
        guidance_scale=4.0,
        flow_shift=1.0,
        prompt="1girl, masterpiece",
        negative_prompt="low quality",
        image_size=(1024, 1024),
        save_path="",
        no_metadata=False,
    )
    base.update(overrides)
    return Namespace(**base)


def _sample_tensor(w: int = 8, h: int = 8):
    import torch

    return torch.zeros(3, h, w)


def test_save_images_embeds_png_metadata(tmp_path) -> None:
    args = _args(save_path=str(tmp_path))
    saved = output.save_images(_sample_tensor(), args)
    path = Path(f"{saved}.png")

    with Image.open(path) as img:
        info = dict(getattr(img, "text", None) or img.info)

    assert int(info["seed"]) == 123456
    assert info["sampler"] == "euler"
    assert int(info["infer_steps"]) == 28
    assert float(info["guidance_scale"]) == 4.0
    assert float(info["flow_shift"]) == 1.0
    assert info["prompt"] == "1girl, masterpiece"
    assert info["negative_prompt"] == "low quality"
    assert int(info["width"]) == 1024
    assert int(info["height"]) == 1024
    assert info["timestamp"]


def test_save_images_omits_metadata_when_no_metadata(tmp_path) -> None:
    args = _args(save_path=str(tmp_path), no_metadata=True)
    saved = output.save_images(_sample_tensor(), args)
    path = Path(f"{saved}.png")

    with Image.open(path) as img:
        info = dict(getattr(img, "text", None) or img.info)

    assert "seed" not in info
    assert "sampler" not in info
    assert "prompt" not in info


def test_save_images_filename_keeps_legacy_shape(tmp_path) -> None:
    args = _args(save_path=str(tmp_path), seed=42)
    saved = output.save_images(_sample_tensor(), args)
    name = Path(f"{saved}.png").name

    assert name.startswith("20") and name.endswith("_42.png")


def test_decode_latent_rejects_batch_gt_one():
    class FakeVae:
        dtype = torch.float32

        def to(self, *_args, **_kwargs):
            return self

        def decode_to_pixels(self, latent):
            batch, channels, _frames, height, width = latent.shape
            return torch.zeros(batch, channels, height, width)

    with pytest.raises(ValueError, match="batch size 1"):
        output.decode_latent(
            FakeVae(),
            torch.zeros(2, 16, 1, 4, 4),
            torch.device("cpu"),
        )
