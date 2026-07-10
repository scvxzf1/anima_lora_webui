"""Shared helpers for no-extra-dataset prior preservation."""

from __future__ import annotations

import argparse


def build_diff_output_prior_caption(
    caption: str,
    *,
    trigger: str | None,
    class_prompt: str | None,
) -> str:
    """Build the DOP/class-prompt prior caption for one training caption.

    First version mirrors ai-toolkit's simple trigger replacement: if a trigger
    is configured, replace every literal occurrence with the class prompt. When
    no trigger is configured, fall back to the class prompt itself.
    """
    class_text = (class_prompt or "").strip()
    trigger_text = (trigger or "").strip()
    if not trigger_text:
        return class_text
    return str(caption or "").replace(trigger_text, class_text).strip()


def prior_preservation_enabled(args: argparse.Namespace) -> bool:
    return float(getattr(args, "prior_preservation_weight", 0.0) or 0.0) > 0.0


def blank_prompt_preservation_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "blank_prompt_preservation", False)) and prior_preservation_enabled(
        args
    )


def diff_output_preservation_enabled(args: argparse.Namespace) -> bool:
    return prior_preservation_enabled(args) and bool(
        str(getattr(args, "diff_output_preservation_class", "") or "").strip()
    )


def inverted_mask_prior_enabled(args: argparse.Namespace, batch: dict) -> bool:
    return (
        float(getattr(args, "inverted_mask_prior_weight", 0.0) or 0.0) > 0.0
        and batch.get("alpha_masks") is not None
    )
