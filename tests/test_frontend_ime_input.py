from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_config_search_defers_form_rerender_during_ime_composition() -> None:
    source = _source("web/static/js/features/config-form/group-entry.js")

    assert "if (event.isComposing) return;" in source
    assert "search.addEventListener('compositionstart'" in source
    assert "applyConfigSearch.cancel();" in source
    assert "search.addEventListener('compositionend'" in source
    assert "if (event.isComposing || event.keyCode === 229) return;" in source


def test_dataset_picker_search_preserves_input_node_while_filtering() -> None:
    source = _source("web/static/js/features/config-form/dataset-picker-dialog.js")
    input_handler = source.split("search.addEventListener('input', () => {", 1)[1].split(
        "});", 1
    )[0]

    assert "renderConfigDatasetPickerResults(body);" in input_handler
    assert "renderConfigDatasetPickerDialog();" not in input_handler
    assert "currentWorkspace.replaceWith(workspace);" in source
