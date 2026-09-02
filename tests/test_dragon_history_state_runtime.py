from __future__ import annotations

import json
import subprocess

from tests.frontend_test_support import REPO_ROOT


def test_history_list_state_survives_detail_round_trip_without_stale_backend_state() -> None:
    script = r"""
const store = await import('./web/static/js/dragon-ui/pages/history-list-state.js?runtime-test');
const collections = await import('./web/static/js/dragon-ui/pages/history-collections.js?runtime-test');

const workspace = collections.createHistoryCollectionWorkspace({ collection_order: ['old'] });
workspace.activeKey = 'collection:123';
workspace.selectedTaskIds.add('task-1');
workspace.expandedConfigKeys.add('config-a');
workspace.initializedExpansion = true;
workspace.dragTaskIds = ['task-1'];

store.saveHistoryListState({
  filters: { search: 'needle', status: 'error', sort: 'oldest', source: 'queue', obsolete: 'ignored' },
  workspace,
});

const defaults = { search: '', status: 'all', sort: 'newest', source: 'all' };
const freshWorkspace = collections.createHistoryCollectionWorkspace({ collection_order: ['123', 'new'] });
const restored = store.restoreHistoryListState(defaults, freshWorkspace);
const tasks = [
  {
    id: 'task-1', name: 'needle training', group: '123', job: 'training', state: 'failed',
    archived: false, from_queue: true, history_group_key: 'config-a', history_group_label: 'Config A',
  },
  {
    id: 'task-2', name: 'other training', group: '', job: 'training', state: 'done',
    archived: false, history_group_key: 'config-b', history_group_label: 'Config B',
  },
];
const html = collections.renderHistoryCollectionWorkbench(tasks, restored.filters, restored.workspace);

restored.filters.search = 'mutated';
restored.workspace.selectedTaskIds.add('task-2');
restored.workspace.expandedConfigKeys.clear();
const restoredAgain = store.restoreHistoryListState(defaults, collections.createHistoryCollectionWorkspace());

console.log(JSON.stringify({
  activeKey: restored.workspace.activeKey,
  selectedTaskIds: [...restored.workspace.selectedTaskIds],
  initializedExpansion: restored.workspace.initializedExpansion,
  currentCollectionSettings: restored.workspace.settings.collection_order,
  transientDragTaskIds: restored.workspace.dragTaskIds,
  obsoleteFilterRestored: Object.hasOwn(restored.filters, 'obsolete'),
  renderedActiveGroup: html.includes('<strong>当前：123</strong>'),
  restoredAgain: {
    filters: restoredAgain.filters,
    selectedTaskIds: [...restoredAgain.workspace.selectedTaskIds],
    expandedConfigKeys: [...restoredAgain.workspace.expandedConfigKeys],
  },
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
        "activeKey": "collection:123",
        "selectedTaskIds": ["task-1", "task-2"],
        "initializedExpansion": True,
        "currentCollectionSettings": ["123", "new"],
        "transientDragTaskIds": [],
        "obsoleteFilterRestored": False,
        "renderedActiveGroup": True,
        "restoredAgain": {
            "filters": {
                "search": "needle",
                "status": "error",
                "sort": "oldest",
                "source": "queue",
            },
            "selectedTaskIds": ["task-1"],
            "expandedConfigKeys": ["config-a"],
        },
    }


def test_history_page_saves_and_restores_list_state_during_route_changes() -> None:
    source = (REPO_ROOT / "web/static/js/dragon-ui/pages/history.js").read_text(encoding="utf-8")

    assert "restoreHistoryListState(" in source
    assert "filters: restored.filters" in source
    assert "workspace: restored.workspace" in source
    assert "saveHistoryListState(state);" in source
