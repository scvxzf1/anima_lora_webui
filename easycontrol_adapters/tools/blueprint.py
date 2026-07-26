"""Shared EasyControl descriptor-blueprint helpers.

Used by the subject / subject_edit miners (and originally by near_twins) to
rewrite only the generated ``[[datasets]]`` tail of a descriptor TOML while
preserving the user-owned head tables. Kept free of near_twins matching
logic so the subject data-side port does not need the full near_twins package.
"""

from __future__ import annotations

import re

_BLUEPRINT_SENTINEL = (
    "# === generated dataset blueprint (rewritten by the miner; do not edit below) ==="
)

# Blueprint section headers ([general] / [[datasets]] / [[datasets.subsets]]).
# These never appear in the head (the head only carries [staging]/[preprocess]/
# [training]), so the first one marks where the generated blueprint begins — the
# fallback boundary when the sentinel comment has been hand-edited away.
_BLUEPRINT_HEADER_RE = re.compile(r"^\s*\[\[?(?:general|datasets)\b")


def _strip_blueprint(existing: str) -> str:
    """Return the user-owned head of ``existing``, dropping any prior blueprint.

    Preferred boundary is the sentinel comment. If it's missing (the user edited
    the head and lost the sentinel line), fall back to the first blueprint
    section header and rewind past the contiguous comment/blank block that
    introduces it — so the blueprint's own header comments don't accumulate
    across runs. Without this fallback a missing sentinel makes ``split`` keep
    the whole file (blueprint included) as head, and each run appends another
    blueprint copy.
    """
    if _BLUEPRINT_SENTINEL in existing:
        return existing.split(_BLUEPRINT_SENTINEL, 1)[0].rstrip()
    lines = existing.splitlines()
    for i, line in enumerate(lines):
        if _BLUEPRINT_HEADER_RE.match(line):
            j = i
            while j > 0 and (
                lines[j - 1].lstrip().startswith("#") or not lines[j - 1].strip()
            ):
                j -= 1
            return "\n".join(lines[:j]).rstrip()
    return existing.rstrip()
