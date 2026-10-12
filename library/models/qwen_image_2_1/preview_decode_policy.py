"""Compatibility exports for the shared training-preview VAE policy."""

from library.training.preview_decode_policy import (
    PreviewDecodeRestoreError, decode_with_policy,
)

__all__ = ["PreviewDecodeRestoreError", "decode_with_policy"]
