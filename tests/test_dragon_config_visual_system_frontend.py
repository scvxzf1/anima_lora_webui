from pathlib import Path
import re


STATIC = Path(__file__).resolve().parents[1] / "web" / "static"


def _read(relative: str) -> str:
    return (STATIC / relative).read_text(encoding="utf-8")


def test_config_route_loads_visual_system_after_layout_layers() -> None:
    route_styles = _read("js/dragon-ui/route-styles.js")
    fields = route_styles.index("04g-dragon-config-fields.css")
    visual = route_styles.index("04h-dragon-config-visual-system.css")

    assert fields < visual


def test_visual_system_owns_the_same_tokens_for_both_themes() -> None:
    css = _read("css/dragon/04h-dragon-config-visual-system.css")
    light, dark = css.split('html[data-ui-mode="dragon"][data-theme="dark"]', 1)
    colors = r"(--cfg-[\w-]+): #[0-9a-f]+;"
    assert set(re.findall(colors, light)) == set(re.findall(colors, dark))
    assert "--cfg-surface: #ffffff" in light
    assert "--cfg-surface: #202126" in dark
    assert "--cfg-accent-ink:" in light and "--cfg-accent-ink:" in dark
    assert "--config-surface: var(--cfg-surface)" in css
    for module in ("structure", "properties", "dialogs"):
        assert f"config-workbench/{module}.css?v=" in css
        assert len(_read(f"css/dragon/config-workbench/{module}.css").splitlines()) < 300


def test_visual_system_keeps_fields_flat_and_state_driven() -> None:
    css = _read("css/dragon/config-workbench/properties.css")

    assert ".dragon-config-block[data-config-availability=\"unavailable\"]" in css
    assert ':is(.dragon-field, .dragon-config-dataset-card)[data-dirty="true"]' in css
    assert "box-shadow: inset 2px 0 var(--cfg-warning)" in css
    assert '.dragon-field[data-preflight-error="true"]' in css
    assert 'html[data-dragon-config-help="contextual"]' in css
    assert ".dragon-field:is(:hover, :focus-within)" in css
    assert "isolation: isolate" in css


def test_visual_system_forces_mobile_single_column_and_full_width_drawer() -> None:
    structure = _read("css/dragon/config-workbench/structure.css")
    fields = _read("css/dragon/config-workbench/properties.css")
    mobile = structure.split("@media (max-width: 734px)", 1)[1]

    assert "grid-template-columns: minmax(0, 1fr)" in mobile
    assert ".dragon-config-tag-filters" in mobile
    assert "overflow-x: auto" in mobile
    assert "width: 100%" in mobile
    assert "max-width: none" in mobile
    drawer = structure.split("@media (max-width: 1000px)", 1)[1]
    assert "[hidden] { display: none !important; }" in structure
    assert "&:has(.dragon-config-all-workspace) .dragon-training-preset-library" in drawer
    assert "position: fixed" in drawer
    assert "grid-template-columns: minmax(0, 1fr)" in fields.split("@media (max-width: 734px)", 1)[1]


def test_config_dialogs_share_theme_and_keep_search_labels_accessible() -> None:
    css = _read("css/dragon/config-workbench/dialogs.css")
    assert ".dragon-model-quick-dialog" in css
    assert ".dragon-dialog-host)::backdrop" in css
    assert "backdrop-filter: none" in css
    assert "clip-path: inset(50%)" in css
    assert ".dragon-btn span { color: inherit" in css
    assert "color: var(--cfg-accent-ink)" in css


def test_config_icon_controls_have_names_and_tooltips() -> None:
    help_source = _read("js/dragon-ui/pages/config-field-help.js")
    library = _read("js/dragon-ui/pages/training-preset-library.js")
    assert "renderIcon('circleHelp')" in help_source
    assert 'aria-haspopup="dialog"' in help_source
    for action in ("new-group", "import", "save-as", "export", "refresh"):
        button = re.search(rf'<button[^>]+data-training-preset-action="{action}"[^>]+>', library)
        assert button, action
        assert "title=" in button.group() and "aria-label=" in button.group()
