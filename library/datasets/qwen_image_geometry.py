"""Qwen Image 2.1 pixel alignment shared by resize, training and audit.

The VAE stride is 16 and DiT additionally groups latent tokens in 2x2
patches. Other model families retain the canonical 16-pixel bucket table.
"""

from __future__ import annotations


def align_qwen_resolution(size: tuple[int, int]) -> tuple[int, int]:
    if min(size) < 32:
        raise ValueError("Qwen Image 2.1 image dimensions must be at least 32 pixels")
    return tuple(int(value) // 32 * 32 for value in size)


def align_qwen_bucket_manager(manager) -> None:
    manager.set_predefined_resos(sorted({
        align_qwen_resolution(size) for size in manager.predefined_resos
    }))
