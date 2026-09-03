from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from tests.frontend_test_support import REPO_ROOT, STATIC_DIR


def _read(relative: str) -> str:
    return (STATIC_DIR / relative).read_text(encoding="utf-8")


def test_dragon_model_config_is_primary_route_after_training_history() -> None:
    nav = _read("js/dragon-ui/nav.js")
    catalog = nav[nav.index("const PRIMARY_NAV_ITEMS = ["):nav.index("];", nav.index("const PRIMARY_NAV_ITEMS = ["))]

    training_history = "{ id: 'history', label: '训练历史', hash: '#history' }"
    model = "{ id: 'model-config', label: '模型配置', hash: '#model-config' }"
    assert catalog.index(training_history) < catalog.index(model)
    assert "hash.startsWith('#model-config')" in nav
    assert "PRIMARY_NAV_ITEMS.map(renderPrimaryNavItem)" in nav
    assert "PRIMARY_NAV_ITEMS.map((item) =>" in nav


def test_dragon_model_config_grouped_library_contract() -> None:
    page = _read("js/dragon-ui/pages/model-config.js")
    library = _read("js/dragon-ui/pages/model-config-library.js")
    state = _read("js/dragon-ui/pages/model-config-state.js")
    css = _read("css/dragon/06-dragon-pages.css")

    assert "normalizeModelGroups" in page
    assert 'data-model-action="add-group"' in page
    assert "groups: state.groups.map(cleanModelGroup)" in page
    assert "validateModelGroups(state.groups, state.items)" in page
    assert "disposeModelConfigDrag(state.root, state)" in page
    assert "serializeModelState(state.items, state.defaultId, state.groups)" in page
    for contract in (
        'data-model-group-row="${escapeAttribute(group.id)}"',
        'data-model-group-action="rename"',
        'data-model-group-action="delete"',
        'data-model-dropzone="${escapeAttribute(group.id)}"',
        'data-model-row="${escapeAttribute(item.id)}"',
        "scheduleOrderedRowDropTarget",
        "onPlaceItem(itemId, groupId, anchorId, position)",
        "placeDraggedGroup(root, state, group, event.clientY, onPlaceGroup)",
        "onPlaceGroup(sourceId, insertionIndex)",
        "cancelOnEscape",
    ):
        assert contract in library
    for helper in (
        "normalizeModelGroups",
        "moveModelGroup",
        "moveModelItemInGroups",
        "placeModelGroup",
        "placeModelItem",
        "removeModelGroup",
    ):
        assert f"export function {helper}" in state
    for selector in (
        ".dragon-model-config-group",
        ".dragon-model-config-group-drag-handle",
        ".dragon-model-config-item-drop-before::before",
        ".dragon-model-config-group-drop-after::after",
        ".dragon-model-config-dropzone[data-over=\"true\"]::before",
    ):
        assert selector in css
    assert "body[data-dragon-ui] #dragon-root .dragon-model-config-group-header" in css


def test_dragon_model_config_actions_stay_inside_editor() -> None:
    page = _read("js/dragon-ui/pages/model-config.js")
    css = _read("css/dragon/06-dragon-pages.css")

    editor_start = page.index('<form class="dragon-model-config-editor"')
    editor_body = page.index('data-model-editor-body', editor_start)
    savebar = page.index('class="dragon-savebar dragon-model-config-savebar"')
    editor_end = page.index("</form>", editor_start)
    assert editor_start < editor_body < savebar < editor_end
    assert "const editorBody = root.querySelector('[data-model-editor-body]')" in page
    assert "editorBody.innerHTML = renderEditor(selectedItem(state), state)" in page
    assert "editor.innerHTML = renderEditor(selectedItem(state), state)" not in page
    assert ".dragon-model-config-editor .dragon-model-config-savebar" in css
    assert "position: static;" in css[css.index(".dragon-model-config-editor .dragon-model-config-savebar"):]
    assert "body[data-dragon-ui] #dragon-root .dragon-model-config-page > .dragon-tool-hero" in css
    assert "body[data-dragon-ui] #dragon-root .dragon-model-editor-header" in css


def test_dragon_model_config_group_state_helpers_with_node() -> None:
    if not shutil.which("node"):
        pytest.skip("node is optional for model configuration helper checks")
    script = r"""
const mod = await import('./web/static/js/dragon-ui/pages/model-config-state.js');
const items = [
  {id: 'a', name: 'A', model_family: 'anima', pretrained_model_name_or_path: 'a', qwen3: 'q', vae: 'v'},
  {id: 'b', name: 'B', model_family: 'anima', pretrained_model_name_or_path: 'b', qwen3: 'q', vae: 'v'},
  {id: 'c', name: 'C', model_family: 'krea2_raw', pretrained_model_name_or_path: 'c', qwen3: 'q', vae: 'v'},
];
const groups = mod.normalizeModelGroups([
  {id: 'main', label: '主力', item_ids: ['a', 'b']},
  {id: 'lab', label: '实验', item_ids: ['c']},
], items);
const crossGroup = mod.placeModelItem(groups, 'b', 'lab', 'c', 'before');
const itemOrder = mod.moveModelItemInGroups(crossGroup, 'b', 1);
const groupOrder = mod.placeModelGroup(itemOrder, 'lab', 0);
const removed = mod.removeModelGroup(groupOrder, 'main');
console.log(JSON.stringify({
  crossGroup: crossGroup.map((group) => [group.id, group.item_ids]),
  itemOrder: itemOrder.map((group) => [group.id, group.item_ids]),
  groupOrder: groupOrder.map((group) => group.id),
  removed: removed.map((group) => [group.id, group.item_ids]),
  valid: mod.validateModelGroups(groups, items),
  invalid: mod.validateModelGroups([{id: 'x', label: 'X', item_ids: ['a']}], items)?.message,
}));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "crossGroup": [["main", ["a"]], ["lab", ["b", "c"]]],
        "itemOrder": [["main", ["a"]], ["lab", ["c", "b"]]],
        "groupOrder": ["lab", "main"],
        "removed": [["lab", ["c", "b", "a"]]],
        "valid": None,
        "invalid": "每个模型配置都必须属于一个分组",
    }
