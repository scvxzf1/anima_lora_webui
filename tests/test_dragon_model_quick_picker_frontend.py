from __future__ import annotations

import json
import subprocess

from tests.frontend_test_support import STATIC_DIR


def _read(relative: str) -> str:
    return (STATIC_DIR / relative).read_text(encoding="utf-8")


def test_quick_picker_is_scoped_to_base_models_and_applies_all_paths() -> None:
    page = _read("js/dragon-ui/pages/config-page.js")
    picker = _read("js/dragon-ui/pages/model-quick-picker.js")

    assert "sub.id === 'base-models' ? renderModelQuickPickerTrigger() : ''" in page
    assert "sub.id === 'base-models' ? renderModelQuickPickerDialog() : ''" in page
    assert "bindModelQuickPicker(wrapper" in page
    assert "MODEL_QUICK_PATH_KEYS.forEach" in page
    assert "input.dispatchEvent(new Event('input', { bubbles: true }))" in page
    assert "需点击保存才会生效" in page
    assert "MODEL_QUICK_PATH_KEYS = Object.freeze(MODEL_PATH_FIELDS.map" in picker
    assert "api('/api/settings/model-configs')" in picker
    assert "dialog.showModal()" in picker
    assert "if (search) search.value = ''" in picker
    assert "data-model-quick-preview" in picker


def test_quick_picker_preserves_model_library_order() -> None:
    module_uri = (STATIC_DIR / "js/dragon-ui/pages/model-quick-picker.js").as_uri()
    script = f"""
globalThis.window = {{ fetch() {{}} }};
const mod = await import({json.dumps(module_uri)});
const groups = mod.orderedModelGroups({{
  items: [
    {{id: 'a', name: 'A', pretrained_model_name_or_path: 'a', qwen3: 'qa', vae: 'va'}},
    {{id: 'b', name: 'B', pretrained_model_name_or_path: 'b', qwen3: 'qb', vae: 'vb'}},
    {{id: 'c', name: 'C', pretrained_model_name_or_path: 'c', qwen3: 'qc', vae: 'vc'}},
  ],
  groups: [
    {{id: 'second', label: '第二组', item_ids: ['c']}},
    {{id: 'first', label: '第一组', item_ids: ['b', 'a']}},
  ],
}});
console.log(JSON.stringify(groups.map((group) => [group.id, group.items.map((item) => item.id)])));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == [["second", ["c"]], ["first", ["b", "a"]]]


def test_quick_picker_has_centered_responsive_path_preview() -> None:
    css = _read("css/dragon/04b-dragon-model-quick-picker.css")
    stylesheet = _read("css/dragon-style.css")

    assert "04b-dragon-model-quick-picker.css?v=dragon-ui-20260819v1" in stylesheet
    assert "body[data-dragon-ui] #dragon-root .dragon-config-detail-header" in css
    assert "body[data-dragon-ui] .dragon-model-quick-dialog" in css
    assert "position: fixed;" in css
    assert "margin: auto;" in css
    assert ".dragon-model-quick-dialog::backdrop" in css
    assert ".dragon-model-quick-groups" in css
    assert ".dragon-model-quick-paths code" in css
    assert "overflow-wrap: anywhere;" in css
    assert "@media (max-width: 734px)" in css
    assert "@media (max-width: 430px)" in css
