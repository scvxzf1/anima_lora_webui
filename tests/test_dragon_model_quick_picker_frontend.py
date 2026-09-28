from __future__ import annotations

import json
import subprocess

from tests.frontend_test_support import STATIC_DIR


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
