"""Unified path allowlist policy tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from web.services import path_safety
from web.services import continue_lora_service


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
