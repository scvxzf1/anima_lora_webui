"""Cross-domain delete/path boundary smoke locks."""

from __future__ import annotations

from pathlib import Path

import pytest

from web.services import path_safety


def test_output_root_and_repo_root_boundaries_do_not_overlap_secret(tmp_path: Path):
    repo = tmp_path / "repo"
    output = tmp_path / "output"
    secret = tmp_path / "secret" / "a.safetensors"
    repo.mkdir(); output.mkdir(); secret.parent.mkdir(); secret.write_bytes(b"x")
    allow = path_safety.allowed_weight_dirs(root=repo, output_root=output)
    assert path_safety.is_under_allowed_dirs(secret, allow) is False
    inside = output / "run" / "a.safetensors"
    inside.parent.mkdir(parents=True)
    inside.write_bytes(b"x")
    assert path_safety.is_under_allowed_dirs(inside, allow) is True
