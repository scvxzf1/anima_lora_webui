from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from tests.frontend_test_support import REPO_ROOT


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
