from __future__ import annotations

import json
import subprocess

from tests.frontend_test_support import REPO_ROOT, STATIC_DIR
from web.services.training.history_config_chips import (
    history_config_chips_from_snapshot_text,
)


def test_history_config_chips_resolve_model_family() -> None:
    assert history_config_chips_from_snapshot_text(
        'model_family = "krea2_raw"\n'
    )["model_family"] == "krea2_raw"
    assert history_config_chips_from_snapshot_text(
        'model_family = "z_image"\n'
    )["model_family"] == "z_image"
    assert history_config_chips_from_snapshot_text(
        'network_module = "networks.lora_anima"\n'
    )["model_family"] == "anima"
    assert history_config_chips_from_snapshot_text(
        'model_family = "unknown"\n'
    )["model_family"] == ""


def test_classic_history_model_family_filter_is_wired() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert 'id="history-filter-model-family"' in html
    assert '<option value="anima">Anima</option>' in html
    assert '<option value="krea2_raw">Krea-2</option>' in html
    assert '<option value="z_image">Z-Image</option>' in html

    sources = {
        path: (STATIC_DIR / path).read_text(encoding="utf-8")
        for path in (
            "js/features/anima-app/state/history-state.js",
            "js/features/app-shell/event-listeners-setup.js",
            "js/features/app-shell/event-listeners-contract.js",
            "js/features/history-list/task-collections.js",
            "js/features/history-list/collections-workbench.js",
        )
    }
    assert "modelFamily: 'all'" in sources["js/features/anima-app/state/history-state.js"]
    assert "'history-filter-model-family': 'modelFamily'" in sources["js/features/app-shell/event-listeners-setup.js"]
    assert "'history-filter-model-family'" in sources["js/features/app-shell/event-listeners-contract.js"]
    assert "'history-filter-model-family': 'modelFamily'" in sources["js/features/history-list/task-collections.js"]
    assert "task?.model_family" in sources["js/features/history-list/collections-workbench.js"]


def test_dragon_history_filters_by_model_family() -> None:
    script = r"""
const model = await import('./web/static/js/dragon-ui/pages/history-model.js?model-family-test');
const view = await import('./web/static/js/dragon-ui/pages/history-list-view.js?model-family-test');
const tasks = [
  { id: 'a', job: 'training', state: 'done', archived: false, model_family: 'anima' },
  { id: 'k', job: 'training', state: 'done', archived: false, model_family: 'krea2_raw' },
  { id: 'z', job: 'training', state: 'done', archived: false, model_family: 'z_image' },
];
const ids = model.filterHistoryTasks(tasks, { modelFamily: 'krea2_raw' }).map((task) => task.id);
const html = view.renderHistoryPage({ tasks, filters: { modelFamily: 'krea2_raw' } });
console.log(JSON.stringify({ ids, hasLabel: html.includes('基座模型'), hasOptions: ['anima', 'krea2_raw', 'z_image'].every((value) => html.includes(`value="${value}"`)) }));
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
    assert json.loads(result.stdout) == {
        "ids": ["k"],
        "hasLabel": True,
        "hasOptions": True,
    }
