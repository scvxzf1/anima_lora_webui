from __future__ import annotations

from tests.frontend_test_support import STATIC_DIR


def _read(relative: str) -> str:
    return (STATIC_DIR / relative).read_text(encoding="utf-8")


def test_native_dialogs_use_containing_width_and_viewport_fallbacks() -> None:
    for relative in (
        "css/dragon/04b-dragon-model-quick-picker.css",
        "css/dragon/06b-dragon-history-sample-dialog.css",
    ):
        css = _read(relative)
        assert "calc(100vw -" not in css
        assert "calc(100% - var(--dragon-sp-8))" in css
        assert "calc(100vh - var(--dragon-sp-9))" in css
        assert "calc(100dvh - var(--dragon-sp-9))" in css
        assert "calc(100% - var(--dragon-sp-5))" in css


def test_shared_dialogs_bound_minimum_height_to_available_viewport() -> None:
    shared = _read("css/dragon/06a-dragon-shared-dialogs.css")
    pages = _read("css/dragon/06-dragon-pages.css")

    assert "min-height: 420px;" not in shared
    assert shared.count("calc(88vh - var(--dragon-shared-dialog-header-reserve))") == 2
    assert shared.count("calc(88dvh - var(--dragon-shared-dialog-header-reserve))") == 2
    assert "--dragon-shared-dialog-header-reserve: 92px;" in pages
    assert "width: min(1120px, calc(100% - var(--dragon-sp-6)));" in pages
    assert "max-height: min(88vh, 920px);" in pages
    assert "max-height: min(88dvh, 920px);" in pages
    assert "max-height: calc(88vh - var(--dragon-shared-dialog-header-reserve));" in pages
    assert "max-height: calc(88dvh - var(--dragon-shared-dialog-header-reserve));" in pages
