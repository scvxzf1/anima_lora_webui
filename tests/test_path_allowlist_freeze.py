"""S-R3/S-R4 freeze: keep project-root relative path convenience (Round C decision A)."""

from __future__ import annotations

from pathlib import Path

from web.services import image_test_service
from web.services.preview import common as preview_common


def test_image_test_allowlist_still_includes_project_root(tmp_path, monkeypatch):
    monkeypatch.setattr(image_test_service, "ROOT", tmp_path)
    preferred = [tmp_path / "output" / "runs"]
    search = [tmp_path / "models"]
    for d in preferred + search:
        d.mkdir(parents=True, exist_ok=True)
    allow = image_test_service._image_test_weight_allowlist(
        preferred_dirs=preferred,
        search_dirs=search,
    )
    roots = {str(p.resolve()) for p in allow}
    assert str(tmp_path.resolve()) in roots


def test_preview_relative_project_path_still_resolves(tmp_path, monkeypatch):
    monkeypatch.setattr(preview_common, "ROOT", tmp_path)
    # facade get may still point elsewhere; patch _root used by helpers
    monkeypatch.setattr(preview_common, "_root", lambda: tmp_path)
    rel = "output/tests/sample.png"
    target = tmp_path / "output" / "tests"
    target.mkdir(parents=True, exist_ok=True)
    (target / "sample.png").write_bytes(b"x")
    resolved = preview_common._resolve_preview_file(rel)
    assert resolved == (tmp_path / rel).resolve()
    assert resolved.is_file()
