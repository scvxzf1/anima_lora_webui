from __future__ import annotations

import json
import subprocess

import pytest

from tests.frontend_test_support import REPO_ROOT

pytestmark = pytest.mark.integration


def test_dataset_path_bindings_cancel_timers_and_listeners() -> None:
    script = r"""
const timers = new Map();
let nextTimer = 1;
globalThis.window = {
  setTimeout(callback, delay) { const id = nextTimer++; timers.set(id, { callback, delay }); return id; },
  clearTimeout(id) { timers.delete(id); },
};

function eventNode(extra = {}) {
  const listeners = new Map();
  return {
    ...extra,
    addEventListener(type, handler) { listeners.set(type, handler); },
    removeEventListener(type, handler) {
      if (listeners.get(type) === handler) listeners.delete(type);
    },
    listenerCount() { return listeners.size; },
  };
}
const inputs = [];
const controls = [];
const rows = Array.from({ length: 12 }, () => {
  const input = eventNode({ value: '/tmp/images', isConnected: true });
  const copy = eventNode({ disabled: false });
  const browse = eventNode();
  const statusText = { textContent: '' };
  const status = { dataset: {}, lastElementChild: statusText };
  inputs.push(input);
  controls.push(copy, browse);
  return {
    querySelector(selector) {
      if (selector === '[data-field="source_dir"]') return input;
      if (selector === '[data-dataset-copy]') return copy;
      if (selector === '[data-dataset-browse]') return browse;
      if (selector === '[data-dataset-path-status]') return status;
      return null;
    },
  };
});
const root = { querySelectorAll() { return rows; } };
const mod = await import('./web/static/js/dragon-ui/pages/dataset-editor-paths.js?runtime-test');
const cleanup = mod.bindDatasetPathTools(async () => ({}), root);
const before = {
  timers: timers.size,
  listeners: [...inputs, ...controls].reduce((sum, node) => sum + node.listenerCount(), 0),
};
cleanup();
const after = {
  timers: timers.size,
  listeners: [...inputs, ...controls].reduce((sum, node) => sum + node.listenerCount(), 0),
};
console.log(JSON.stringify({ before, after }));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    assert payload["before"] == {"timers": 12, "listeners": 36}
    assert payload["after"] == {"timers": 0, "listeners": 0}
