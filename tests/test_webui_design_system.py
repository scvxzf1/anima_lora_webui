from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STYLE = (ROOT / "web/static/style.css").read_text(encoding="utf-8")
DS_DIR = ROOT / "web/static/css/ds"


def test_style_entry_imports_design_system_layer() -> None:
    for rel in [
        "./css/ds/00-tokens-extend.css",
        "./css/ds/10-primitives.css",
        "./css/ds/20-patterns.css",
    ]:
        assert rel in STYLE, f"style.css missing import {rel}"


def test_design_system_files_exist() -> None:
    for name in ["00-tokens-extend.css", "10-primitives.css", "20-patterns.css"]:
        assert (DS_DIR / name).is_file(), f"missing {name}"


def test_primitives_define_core_classes() -> None:
    prim = (DS_DIR / "10-primitives.css").read_text(encoding="utf-8")
    for selector in [
        ".ui-btn",
        ".ui-btn--primary",
        ".ui-btn--highlight",
        ".ui-btn--danger",
        ".ui-field",
        ".ui-field__label",
        ".ui-field__control",
        ".ui-segmented",
        ".ui-segmented__btn",
        ".ui-card",
        ".ui-toolbar",
        ".ui-sidebar",
        ".ui-stat",
        ".ui-stat__value",
        ".ui-sticky",
    ]:
        assert selector in prim, f"missing primitive {selector}"


def test_patterns_define_console_boards() -> None:
    patterns = (DS_DIR / "20-patterns.css").read_text(encoding="utf-8")
    for selector in [
        ".page-shell",
        ".workbench",
        ".monitor-board",
        ".history-board",
    ]:
        assert selector in patterns, f"missing pattern {selector}"
