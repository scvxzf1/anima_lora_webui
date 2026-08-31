from __future__ import annotations

import json
import subprocess

from tests.frontend_test_support import REPO_ROOT


def test_history_search_index_reuses_normalized_task_text() -> None:
    script = r"""
let searchFieldReads = 0;
const tasks = Array.from({ length: 200 }, (_, index) => {
  const task = {
    id: `task-${index}`,
    name: `训练 ${index}`,
    group: index % 2 ? 'portrait' : 'landscape',
    history_source_config_file: index % 3 ? 'lora.toml' : 'krea.toml',
    job: 'training', state: 'done', archived: false, started_at: index,
  };
  Object.defineProperty(task, 'message', {
    get() { searchFieldReads += 1; return index % 10 === 0 ? 'needle' : 'ordinary'; },
  });
  return task;
});

const mod = await import('./web/static/js/dragon-ui/pages/history-model.js?runtime-test');
const index = mod.createHistorySearchIndex(tasks);
const readsToBuild = searchFieldReads;
searchFieldReads = 0;
let indexedIds = [];
for (let iteration = 0; iteration < 100; iteration += 1) {
  indexedIds = mod.filterHistoryTasks(tasks, { search: 'needle', archived: 'active' }, index).map((task) => task.id);
}
const indexedReads = searchFieldReads;
searchFieldReads = 0;
const plainIds = mod.filterHistoryTasks(tasks, { search: 'needle', archived: 'active' }).map((task) => task.id);
const plainReads = searchFieldReads;
const collectionIds = mod.filterHistoryTasks(tasks, { search: '组:portrait', archived: 'active' }, index).map((task) => task.id);
const configIds = mod.filterHistoryTasks(tasks, { search: '配置:krea', archived: 'active' }, index).map((task) => task.id);

console.log(JSON.stringify({
  readsToBuild,
  indexedReads,
  plainReads,
  sameResults: JSON.stringify(indexedIds) === JSON.stringify(plainIds),
  matched: indexedIds.length,
  collectionMatched: collectionIds.length,
  configMatched: configIds.length,
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
        "readsToBuild": 200,
        "indexedReads": 0,
        "plainReads": 200,
        "sameResults": True,
        "matched": 20,
        "collectionMatched": 100,
        "configMatched": 67,
    }
