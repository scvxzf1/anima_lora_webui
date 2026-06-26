"""Shared helpers for no-extra-dataset prior preservation."""

from __future__ import annotations


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
