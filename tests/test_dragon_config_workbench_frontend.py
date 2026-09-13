from pathlib import Path


STATIC = Path(__file__).resolve().parents[1] / "web" / "static"


def _read(relative: str) -> str:
    return (STATIC / relative).read_text(encoding="utf-8")


def test_training_config_uses_optional_preset_library_in_both_views() -> None:
    page = _read("js/dragon-ui/pages/config-page.js")
    view = _read("js/dragon-ui/pages/config-all-view.js")
    library = _read("js/dragon-ui/pages/training-preset-library.js")

    assert "export function bindConfigPresetLibrary" in view
    assert "allConfigCleanup = bindConfigPresetLibrary(root" in page
    assert page.count("data-config-preset-toggle") >= 1
    assert "data-config-preset-toggle" in library
    assert "persistPresetLibraryCollapsed(collapsed)" in page


def test_training_config_loads_compact_responsive_workbench_styles() -> None:
    route_styles = _read("js/dragon-ui/route-styles.js")
    css = _read("css/dragon/04f-dragon-config-workbench.css")

    assert "04f-dragon-config-workbench.css" in route_styles
    assert "grid-template-columns: minmax(150px, 180px) minmax(0, 1fr)" in css
    assert ".dragon-config-index-links" in css
    assert "position: absolute" in css
    assert "data-preset-collapsed" in _read("css/dragon/04c-dragon-config-all.css")
