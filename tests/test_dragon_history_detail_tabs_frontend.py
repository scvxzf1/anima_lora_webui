from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_history_detail_routes_include_task_subview() -> None:
    entry = (REPO_ROOT / "web/static/js/dragon-ui/index.js").read_text(encoding="utf-8")
    router = (REPO_ROOT / "web/static/js/dragon-ui/router.js").read_text(encoding="utf-8")
    history = (REPO_ROOT / "web/static/js/dragon-ui/pages/history.js").read_text(encoding="utf-8")

    assert "sub: parts[2] || null" in entry
    assert "taskId: decodeHashPart(parts[1])" in entry
    assert "updateMountedPage(route.page, context)" in router
    assert "onRouteUpdate: (context) => updateHistoryDetailRoute(context, taskId)" in history
    assert "if (context.taskId !== mountedTaskId || !mountedRoot) return false" in history


def test_history_detail_tabs_render_and_switch_without_reload() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for Dragon history detail tab checks")
    jsdom_api = REPO_ROOT / "web/frontend-next/node_modules/jsdom/lib/api.js"
    if not jsdom_api.exists():
        pytest.skip("jsdom is required for Dragon history detail tab checks")

    script = r"""
import { JSDOM } from './web/frontend-next/node_modules/jsdom/lib/api.js';
import { renderHistoryDetailPage } from './web/static/js/dragon-ui/pages/history-view.js';
import {
  activateHistoryDetailTab,
  normalizeHistoryDetailTab,
} from './web/static/js/dragon-ui/pages/history-detail-tabs.js';

const model = {
  taskId: 'task / 1',
  activeTab: 'metrics',
  payload: {
    task: {
      id: 'task / 1', job: 'training', state: 'idle', variant: 'lora', preset: 'default',
      config_snapshot: '/tmp/config.toml', runtime_config_file: '/tmp/runtime.toml',
    },
    metrics: [{ step: 1, loss: 0.4, lr: 1e-5 }, { step: 2, loss: 0.2, lr: 0 }],
    logs: [{ line: 'one' }, { line: 'two' }],
    config_toml: '[training]',
  },
  images: { images: [{ name: 'a.png', url: '/a.png' }, { name: 'b.png', url: '/b.png' }] },
  weights: { weights: [{ name: 'a.safetensors', abs_path: '/tmp/output/a.safetensors', file: 'output/a.safetensors', download_url: '/download/a' }] },
  resume: { checkpoints: [{ path: '/tmp/state', resume_available: true }] },
  lossChart: '<svg data-history-chart></svg>',
};

const dom = new JSDOM(`<main id="root">${renderHistoryDetailPage(model)}</main>`);
const root = dom.window.document.querySelector('#root');
const initialVisible = [...root.querySelectorAll('[data-history-detail-panel]')]
  .filter((panel) => !panel.hidden)
  .map((panel) => panel.dataset.historyDetailPanel);
const hrefs = [...root.querySelectorAll('[data-history-detail-tab]')].map((link) => link.getAttribute('href'));
const labels = [...root.querySelectorAll('[data-history-detail-tab]')]
  .map((link) => link.querySelector('span')?.textContent.trim());

const switched = activateHistoryDetailTab(root, 'logs');
const visibleAfterSwitch = [...root.querySelectorAll('[data-history-detail-panel]')]
  .filter((panel) => !panel.hidden)
  .map((panel) => panel.dataset.historyDetailPanel);
const currentAfterSwitch = root.querySelector('[data-history-detail-tab][aria-current="page"]')
  ?.dataset.historyDetailTab;

console.log(JSON.stringify({
  defaultTab: normalizeHistoryDetailTab(null),
  invalidTab: normalizeHistoryDetailTab('unknown'),
  initialVisible,
  hrefs,
  labels,
  switched,
  visibleAfterSwitch,
  currentAfterSwitch,
  resumeShortcut: Boolean(root.querySelector('[data-history-resume-shortcut]')),
  weightCopyPath: root.querySelector('[data-history-weight-copy]')?.dataset.historyWeightCopy,
  weightCopyLabel: root.querySelector('[data-history-weight-copy] span')?.textContent.trim(),
  weightDownloadHref: root.querySelector('.dragon-history-weight-actions a')?.getAttribute('href'),
}));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    assert payload["defaultTab"] == "overview"
    assert payload["invalidTab"] == "overview"
    assert payload["initialVisible"] == ["metrics"]
    assert payload["switched"] == "logs"
    assert payload["visibleAfterSwitch"] == ["logs"]
    assert payload["currentAfterSwitch"] == "logs"
    assert payload["resumeShortcut"] is True
    assert payload["weightCopyPath"] == "/tmp/output/a.safetensors"
    assert payload["weightCopyLabel"] == "复制路径"
    assert payload["weightDownloadHref"] == "/download/a"
    assert len(payload["hrefs"]) == 5
    assert payload["hrefs"][0] == "#history/task%20%2F%201/overview"
    assert payload["labels"] == ["概览", "指标", "产物", "配置", "日志"]


def test_history_detail_binds_weight_path_copy_feedback() -> None:
    history = (REPO_ROOT / "web/static/js/dragon-ui/pages/history.js").read_text(encoding="utf-8")

    assert "button.addEventListener('click', () => copyHistoryWeightPath(button))" in history
    assert "await copyText(path);" in history
    assert "label.textContent = '已复制'" in history
    assert "label.textContent = '复制失败'" in history
