from __future__ import annotations

import json
import subprocess

import pytest

from tests.frontend_test_support import STATIC_DIR

pytestmark = pytest.mark.integration


def test_config_dirty_updates_are_incremental_after_binding() -> None:
    module_uri = (STATIC_DIR / "js/dragon-ui/pages/config-dirty-state.js").resolve().as_uri()
    script = f"""
let queryCount = 0;
const dirtyWrites = Array(145).fill(0);
const resets = Array.from({{ length: 145 }}, () => ({{ hidden: true }}));
const fields = Array.from({{ length: 145 }}, (_, index) => {{
  const values = {{ configFieldKey: `key-${{index}}` }};
  return {{
    dataset: new Proxy(values, {{
      set(target, key, value) {{
        if (key === 'dirty') dirtyWrites[index] += 1;
        target[key] = value;
        return true;
      }},
    }}),
    querySelector() {{ queryCount += 1; return resets[index]; }},
  }};
}});
const count = {{ textContent: '' }};
const label = {{ textContent: '' }};
const changedOnly = {{ disabled: true, dataset: {{}}, querySelector() {{ queryCount += 1; return label; }} }};
const root = {{
  querySelectorAll() {{ queryCount += 1; return fields; }},
  querySelector(selector) {{
    queryCount += 1;
    if (selector === '[data-config-dirty-count]') return count;
    if (selector === '[data-config-changed-only]') return changedOnly;
    return null;
  }},
}};

const mod = await import({json.dumps(module_uri + '?runtime-test')});
const bindings = mod.createConfigDirtyBindings(root);
const queriesAfterBind = queryCount;
const state = {{
  baselineValues: Object.fromEntries(fields.map((_, index) => [`key-${{index}}`, ''])),
  draftValues: Object.fromEntries(fields.map((_, index) => [`key-${{index}}`, ''])),
  dirtyKeys: new Set(),
  dirty: false,
  showChangedOnly: false,
}};

for (let index = 0; index < 100; index += 1) {{
  state.draftValues['key-42'] = `value-${{index}}`;
  mod.updateConfigDirtyKey(state, 'key-42', '');
  state.dirty = state.dirtyKeys.size > 0;
  mod.renderConfigDirtyState(bindings, state, 'key-42');
}}
state.draftValues['key-42'] = '';
mod.updateConfigDirtyKey(state, 'key-42', '');
state.dirty = state.dirtyKeys.size > 0;
mod.renderConfigDirtyState(bindings, state, 'key-42');

console.log(JSON.stringify({{
  queriesAfterBind,
  queriesAfterUpdates: queryCount,
  targetWrites: dirtyWrites[42],
  otherWrites: dirtyWrites.reduce((sum, value, index) => sum + (index === 42 ? 0 : value), 0),
  dirtySize: state.dirtyKeys.size,
  countText: count.textContent,
  resetHidden: resets[42].hidden,
}}));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["queriesAfterBind"] >= 148
    assert payload["queriesAfterUpdates"] == payload["queriesAfterBind"]
    assert payload["targetWrites"] == 2
    assert payload["otherWrites"] == 0
    assert payload["dirtySize"] == 0
    assert payload["countText"] == "未修改"
    assert payload["resetHidden"] is True
