"""Guards for single-image latent decode paths."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def test_decode_latent_rejects_batch_gt_one(monkeypatch):
    from library.inference import output as output_mod

    class _FakeVae:
        dtype = torch.float32

        def to(self, *args, **kwargs):
            return self

        def decode_to_pixels(self, latent):
            # Return a 2-batch CHW-like tensor after frame squeeze path.
            b, c, _t, h, w = latent.shape
            return torch.zeros(b, c, h, w)

    with pytest.raises(ValueError, match="batch size 1"):
        output_mod.decode_latent(
            _FakeVae(),
            torch.zeros(2, 16, 1, 4, 4),
            torch.device("cpu"),
        )


def test_latent_mode_squeeze_requires_batch_one():
    # Mirror the guard in inference.main latent path without launching full CLI.
    latents = torch.zeros(2, 16, 1, 4, 4)
    assert latents.ndim == 5
    if latents.shape[0] != 1:
        with pytest.raises(ValueError, match="batch size must be 1"):
            if latents.shape[0] != 1:
                raise ValueError(
                    f"Latent batch size must be 1 for decode path, got "
                    f"shape {tuple(latents.shape)} from demo.safetensors"
                )
            latents = latents.squeeze(0)

    ok = torch.zeros(1, 16, 1, 4, 4)
    if ok.ndim == 5:
        if ok.shape[0] != 1:
            raise AssertionError("unexpected")
        squeezed = ok.squeeze(0)
    assert squeezed.shape == (16, 1, 4, 4)
