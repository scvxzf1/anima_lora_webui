"""Unified path allowlist policy tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from web.services import path_safety
from web.services import continue_lora_service
from web.services import image_test_service
from web.services.preview import common as preview_common
from web.services.weight_analysis import paths as analysis_paths


def test_resolve_allowed_file_rejects_parent_escape(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    allow = [root]
    with pytest.raises(ValueError, match=r"\.\."):
        path_safety.resolve_allowed_file("../x.safetensors", root=root, allowed_dirs=allow)


def test_resolve_allowed_file_rejects_outside_absolute(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside" / "a.safetensors"
    outside.parent.mkdir()
    outside.write_bytes(b"x")
    with pytest.raises(ValueError, match="允许范围"):
        path_safety.resolve_allowed_file(str(outside), root=root, allowed_dirs=[root])


def test_resolve_allowed_file_accepts_under_allowlist(tmp_path: Path):
    root = tmp_path / "repo"
    weight = root / "models" / "a.safetensors"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"x")
    resolved = path_safety.resolve_allowed_file(
        "models/a.safetensors",
        root=root,
        allowed_dirs=[root],
    )
    assert resolved == weight.resolve()


def test_continue_lora_rejects_outside_absolute(tmp_path: Path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "secret" / "a.safetensors"
    outside.parent.mkdir()
    outside.write_bytes(b"x")
    # avoid real safetensors parse if somehow accepted
    monkeypatch.setattr(continue_lora_service, "_read_safetensors_header", lambda p: ({}, []))
    with pytest.raises(ValueError, match=r"允许范围|\.\.|路径"):
        continue_lora_service.inspect_continue_lora_weight(
            str(outside),
            root=root,
            output_root=root / "output",
        )


def test_preview_and_analysis_reject_outside_like_resolve_allowed(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    output = tmp_path / "output"
    secret = tmp_path / "secret" / "a.safetensors"
    repo.mkdir()
    output.mkdir()
    secret.parent.mkdir()
    secret.write_bytes(b"x")

    monkeypatch.setattr(preview_common, "_root", lambda: repo)
    monkeypatch.setattr(analysis_paths, "_root", lambda: repo)
    allowed = path_safety.allowed_weight_dirs(root=repo, output_root=output)
    monkeypatch.setattr(
        preview_common, "_allowed_weight_dirs", lambda task=None: allowed
    )
    monkeypatch.setattr(
        analysis_paths, "_allowed_weight_dirs", lambda task=None: allowed
    )

    for candidate in (str(secret), "../secret/a.safetensors"):
        with pytest.raises(ValueError):
            preview_common._resolve_weight_file(candidate)
        with pytest.raises(ValueError):
            analysis_paths.resolve_analysis_weight(candidate)


def test_preview_and_analysis_accept_under_output_root(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    output = tmp_path / "output"
    weight = output / "run" / "a.safetensors"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"x")
    repo.mkdir(exist_ok=True)

    monkeypatch.setattr(preview_common, "_root", lambda: repo)
    monkeypatch.setattr(analysis_paths, "_root", lambda: repo)
    allowed = path_safety.allowed_weight_dirs(root=repo, output_root=output)
    monkeypatch.setattr(
        preview_common, "_allowed_weight_dirs", lambda task=None: allowed
    )
    monkeypatch.setattr(
        analysis_paths, "_allowed_weight_dirs", lambda task=None: allowed
    )

    assert preview_common._resolve_weight_file(str(weight)) == weight.resolve()
    assert analysis_paths.resolve_analysis_weight(str(weight)) == weight.resolve()


def test_image_test_allowlist_still_includes_project_root(tmp_path, monkeypatch):
    monkeypatch.setattr(image_test_service, "ROOT", tmp_path)
    preferred = [tmp_path / "output" / "runs"]
    search = [tmp_path / "models"]
    for directory in preferred + search:
        directory.mkdir(parents=True, exist_ok=True)

    allowlist = image_test_service._image_test_weight_allowlist(
        preferred_dirs=preferred,
        search_dirs=search,
    )

    roots = {str(path.resolve()) for path in allowlist}
    assert str(tmp_path.resolve()) in roots


def test_preview_relative_project_path_still_resolves(tmp_path, monkeypatch):
    monkeypatch.setattr(preview_common, "ROOT", tmp_path)
    monkeypatch.setattr(preview_common, "_root", lambda: tmp_path)
    relative = "output/tests/sample.png"
    target = tmp_path / "output" / "tests"
    target.mkdir(parents=True, exist_ok=True)
    (target / "sample.png").write_bytes(b"x")

    resolved = preview_common._resolve_preview_file(relative)

    assert resolved == (tmp_path / relative).resolve()
    assert resolved.is_file()
