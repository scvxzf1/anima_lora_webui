from __future__ import annotations

from pathlib import Path

from scripts import sync_vendor


def test_compare_vendor_tree_reports_missing_extra_and_changed(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    (expected / "pkg").mkdir(parents=True)
    (actual / "pkg").mkdir(parents=True)
    (expected / "pkg" / "same.py").write_text("same\n", encoding="utf-8")
    (actual / "pkg" / "same.py").write_text("same\n", encoding="utf-8")
    (expected / "pkg" / "changed.py").write_text("new\n", encoding="utf-8")
    (actual / "pkg" / "changed.py").write_text("old\n", encoding="utf-8")
    (expected / "pkg" / "missing.py").write_text("expected\n", encoding="utf-8")
    (actual / "pkg" / "extra.py").write_text("extra\n", encoding="utf-8")

    issues = sync_vendor._compare_vendor_tree(expected, actual, "demo")

    assert "demo: changed pkg/changed.py" in issues
    assert "demo: missing pkg/missing.py" in issues
    assert "demo: extra pkg/extra.py" in issues
    assert all("same.py" not in issue for issue in issues)


def test_vendor_check_builds_expected_tree_outside_repo(tmp_path: Path, monkeypatch) -> None:
    actual_root = tmp_path / "actual"
    roots = {
        "TAGGER_VENDOR": actual_root / "tagger" / "_vendor",
        "DIRECTEDIT_VENDOR": actual_root / "directedit" / "_vendor",
        "HYDRALORA_VENDOR": actual_root / "hydralora" / "_vendor",
        "TRAINER_VENDOR": actual_root / "trainer" / "_vendor",
    }
    for name, root in roots.items():
        monkeypatch.setattr(sync_vendor, name, root)
        root.mkdir(parents=True)
        (root / "marker.py").write_text("fresh\n", encoding="utf-8")

    def fake_build_all_vendor_trees() -> None:
        for root in (
            sync_vendor.TAGGER_VENDOR,
            sync_vendor.DIRECTEDIT_VENDOR,
            sync_vendor.HYDRALORA_VENDOR,
            sync_vendor.TRAINER_VENDOR,
        ):
            root.mkdir(parents=True)
            (root / "marker.py").write_text("fresh\n", encoding="utf-8")

    monkeypatch.setattr(sync_vendor, "_build_all_vendor_trees", fake_build_all_vendor_trees)

    assert sync_vendor.check_vendor_trees() == 0
    for root in roots.values():
        assert (root / "marker.py").read_text(encoding="utf-8") == "fresh\n"
