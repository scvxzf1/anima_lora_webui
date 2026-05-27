"""Compatibility imports for the historical ``preprocess`` package.

The executable preprocess scripts now live under ``scripts.preprocess``.
Keep these module attributes so older tests and local tooling that import
``preprocess.resize_images`` continue to resolve without duplicating code.
"""

from __future__ import annotations

from importlib import import_module

cache_latents = import_module("scripts.preprocess.cache_latents")
cache_pe_encoder = import_module("scripts.preprocess.cache_pe_encoder")
cache_pooled_text = import_module("scripts.preprocess.cache_pooled_text")
cache_text_embeddings = import_module("scripts.preprocess.cache_text_embeddings")
generate_masks = import_module("scripts.preprocess.generate_masks")
generate_masks_mit = import_module("scripts.preprocess.generate_masks_mit")
merge_masks = import_module("scripts.preprocess.merge_masks")
resize_images = import_module("scripts.preprocess.resize_images")

__all__ = [
    "cache_latents",
    "cache_pe_encoder",
    "cache_pooled_text",
    "cache_text_embeddings",
    "generate_masks",
    "generate_masks_mit",
    "merge_masks",
    "resize_images",
]
