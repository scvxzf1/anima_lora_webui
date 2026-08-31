from __future__ import annotations

import json
import subprocess

import pytest

from tests.frontend_test_support import STATIC_DIR

pytestmark = pytest.mark.integration


def test_live_dom_bindings_query_once_and_skip_identical_writes() -> None:
    module_uri = (STATIC_DIR / "js/dragon-ui/pages/live-training-dom.js").resolve().as_uri()
    script = f"""
let queryCount = 0;
let textMutations = 0;
let widthMutations = 0;
let datasetMutations = 0;
let propertyMutations = 0;
let attributeMutations = 0;
const nodes = new Map();

function nodeFor(selector) {{
  if (nodes.has(selector)) return nodes.get(selector);
  let text = '';
  let width = '';
  let hidden = false;
  const datasetValues = {{}};
  const attributes = new Map();
  const node = {{
    dataset: new Proxy(datasetValues, {{
      set(target, key, value) {{ target[key] = value; datasetMutations += 1; return true; }},
    }}),
    get textContent() {{ return text; }},
    set textContent(value) {{ text = value; textMutations += 1; }},
    get hidden() {{ return hidden; }},
    set hidden(value) {{ hidden = value; propertyMutations += 1; }},
    getAttribute(key) {{ return attributes.get(key) ?? null; }},
    setAttribute(key, value) {{ attributes.set(key, value); attributeMutations += 1; }},
    style: {{
      get width() {{ return width; }},
      set width(value) {{ width = value; widthMutations += 1; }},
    }},
  }};
  nodes.set(selector, node);
  return node;
}}

const root = {{
  querySelector(selector) {{ queryCount += 1; return nodeFor(selector); }},
  querySelectorAll(selector) {{ queryCount += 1; return selector === '[data-live-section]' ? [nodeFor('section:0'), nodeFor('section:1')] : []; }},
}};

const mod = await import({json.dumps(module_uri + '?runtime-test')});
const dom = mod.createLiveDomBindings(root);
const queriesAfterBind = queryCount;
for (let index = 0; index < 100; index += 1) {{
  mod.setLiveText(dom, 'state', '训练中');
  mod.setLiveWidth(dom.progressFill, 42);
  mod.setLiveDataset(dom.stateBadge, 'state', 'running');
  mod.setLiveProperty(dom.sections[0], 'hidden', true);
  mod.setLiveAttribute(dom.progress, 'aria-valuenow', 42);
}}
console.log(JSON.stringify({{
  queriesAfterBind,
  queriesAfterUpdates: queryCount,
  textMutations,
  widthMutations,
  datasetMutations,
  propertyMutations,
  attributeMutations,
  sectionCount: dom.sections.length,
}}));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["queriesAfterBind"] > 20
    assert payload["queriesAfterUpdates"] == payload["queriesAfterBind"]
    assert payload["textMutations"] == 1
    assert payload["widthMutations"] == 1
    assert payload["datasetMutations"] == 1
    assert payload["propertyMutations"] == 1
    assert payload["attributeMutations"] == 1
    assert payload["sectionCount"] == 2
