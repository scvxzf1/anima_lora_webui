"""preview / analysis weight resolve share path_safety policy."""

from __future__ import annotations

from pathlib import Path

import pytest

from web.services import path_safety
from web.services.preview import common as preview_common
from web.services.weight_analysis import paths as analysis_paths


def test_preview_and_analysis_reject_outside_like_resolve_allowed(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    output = tmp_path / "output"
    secret = tmp_path / "secret" / "a.safetensors"
    repo.mkdir()
    output.mkdir()
    secret.parent.mkdir()
    secret.write_bytes(b"x")

    monkeypatch.setattr(preview_common, "_root", lambda: repo)
    monkeypatch.setattr(analysis_paths, "_root", lambda: repo)
    monkeypatch.setattr(
        preview_common,
        "_allowed_weight_dirs",
        lambda task=None: path_safety.allowed_weight_dirs(root=repo, output_root=output),
    )
    monkeypatch.setattr(
        analysis_paths,
        "_allowed_weight_dirs",
        lambda task=None: path_safety.allowed_weight_dirs(root=repo, output_root=output),
    )

    with pytest.raises(ValueError):
        preview_common._resolve_weight_file(str(secret))
    with pytest.raises(ValueError):
        analysis_paths.resolve_analysis_weight(str(secret))
    with pytest.raises(ValueError):
        preview_common._resolve_weight_file("../secret/a.safetensors")
    with pytest.raises(ValueError):
        analysis_paths.resolve_analysis_weight("../secret/a.safetensors")


def test_preview_and_analysis_accept_under_output_root(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    output = tmp_path / "output"
    weight = output / "run" / "a.safetensors"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"x")
    repo.mkdir(exist_ok=True)

    monkeypatch.setattr(preview_common, "_root", lambda: repo)
    monkeypatch.setattr(analysis_paths, "_root", lambda: repo)
    allow = path_safety.allowed_weight_dirs(root=repo, output_root=output)
    monkeypatch.setattr(preview_common, "_allowed_weight_dirs", lambda task=None: allow)
    monkeypatch.setattr(analysis_paths, "_allowed_weight_dirs", lambda task=None: allow)

    assert preview_common._resolve_weight_file(str(weight)) == weight.resolve()
    assert analysis_paths.resolve_analysis_weight(str(weight)) == weight.resolve()
