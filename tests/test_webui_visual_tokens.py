from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKENS = (ROOT / "web/static/css/00-tokens.css").read_text(encoding="utf-8")
STYLE = (ROOT / "web/static/style.css").read_text(encoding="utf-8")


def test_visual_tokens_define_instrument_panel_surfaces() -> None:
    required = [
        "--bg",
        "--bg-card",
        "--bg-input",
        "--text",
        "--text-dim",
        "--accent",
        "--border",
        "--radius",
        "--surface-raised",
        "--status-idle",
        "--status-running",
        "--status-error",
        "--status-warning",
        "--font-size-field",
        "--font-size-field-label",
        "--control-height",
        "--header-height",
        "--space-1",
        "--space-2",
        "--space-3",
    ]
    missing = [name for name in required if name not in TOKENS]
    assert not missing, f"missing visual tokens: {missing}"


def test_light_and_dark_theme_blocks_exist() -> None:
    assert ":root" in TOKENS
    assert ':root[data-theme="light"]' in TOKENS or "[data-theme=\"light\"]" in TOKENS
    # dark may be default :root; ensure light override exists and is not empty
    assert "--bg:" in TOKENS
    assert TOKENS.count("--bg:") >= 2


def test_style_entry_keeps_token_first() -> None:
    first_import = next(
        line.strip()
        for line in STYLE.splitlines()
        if line.strip().startswith("@import url(\"./css/")
    )
    assert "00-tokens.css" in first_import


def test_design_system_extended_tokens_are_defined() -> None:
    required = [
        "--font-size-title",
        "--font-size-section",
        "--font-size-mono",
        "--control-height-sm",
        "--control-height-md",
        "--control-height-lg",
        "--space-5",
        "--surface-page",
        "--surface-panel",
        "--surface-input",
        "--surface-sticky",
        "--status-success",
        "--panel-shadow-soft",
    ]
    root = Path(__file__).resolve().parents[1]
    blob = (root / "web/static/css/00-tokens.css").read_text(encoding="utf-8")
    extend = root / "web/static/css/ds/00-tokens-extend.css"
    if extend.exists():
        blob += "\n" + extend.read_text(encoding="utf-8")
    missing = [name for name in required if name not in blob]
    assert not missing, f"missing design-system tokens: {missing}"
