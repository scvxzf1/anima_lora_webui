from __future__ import annotations

import json
import subprocess

from tests.frontend_test_support import REPO_ROOT


def test_dataset_dirty_bindings_and_row_summaries_are_incremental() -> None:
    script = r"""
let rootQueries = 0;
let previewScans = 0;
const nodes = new Map();
function textNode() { return { textContent: '', dataset: {} }; }
nodes.set('[data-dataset-dirty]', textNode());
nodes.set('[data-dataset-dirty-text]', textNode());
nodes.set('[data-savebar-status]', textNode());
nodes.set('[data-dataset-sync-card]', {
  attrs: {},
  getAttribute(key) { return this.attrs[key] ?? null; },
  setAttribute(key, value) { this.attrs[key] = value; },
});
const previews = Array.from({ length: 20 }, () => ({ disabled: false }));
const root = {
  querySelector(selector) { rootQueries += 1; return nodes.get(selector) || null; },
  querySelectorAll(selector) {
    rootQueries += 1;
    if (selector === '[data-dataset-preview]') { previewScans += 1; return previews; }
    return [];
  },
};

let otherRowQueries = 0;
const summary = { textContent: '' };
const controls = new Map([
  ['is_reg', { value: 'true' }],
  ['num_repeats', { value: '4' }],
  ['resolution', { value: '768' }],
]);
const row = {
  querySelector(selector) {
    const match = selector.match(/data-field="([^"]+)"/);
    if (match) return controls.get(match[1]) || null;
    if (selector === '[data-row-summary]') return summary;
    return null;
  },
};
const changedControl = {
  dataset: { field: 'num_repeats' },
  closest() { return row; },
};
const unrelatedControl = {
  dataset: { field: 'source_dir' },
  closest() { otherRowQueries += 1; return row; },
};

const mod = await import('./web/static/js/dragon-ui/pages/dataset-editor-runtime.js?runtime-test');
const bindings = mod.createDatasetEditorBindings(root);
const queriesAfterBind = rootQueries;
const state = { dirty: false, readonly: false, selectedFile: 'a.toml', datasetConfig: 'a.toml' };
for (let index = 0; index < 100; index += 1) {
  if (!state.dirty) {
    state.dirty = true;
    mod.renderDatasetDirtyState(bindings, state);
    mod.disableDatasetPreviews(root);
  }
}
const updated = mod.updateDatasetRowSummaryForControl(changedControl);
const skipped = mod.updateDatasetRowSummaryForControl(unrelatedControl);

console.log(JSON.stringify({
  queriesAfterBind,
  rootQueries,
  previewScans,
  disabledPreviews: previews.filter((button) => button.disabled).length,
  dirtyText: nodes.get('[data-dataset-dirty-text]').textContent,
  summary: summary.textContent,
  updated,
  skipped,
  otherRowQueries,
}));
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
    assert payload == {
        "queriesAfterBind": 4,
        "rootQueries": 5,
        "previewScans": 1,
        "disabledPreviews": 20,
        "dirtyText": "有未保存更改",
        "summary": "正则数据 · 重复 4 · 768px",
        "updated": True,
        "skipped": False,
        "otherRowQueries": 0,
    }
