from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
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
      project_root_abs: '/workspace', history_dir_abs: '/workspace/configs/history/task',
      run_dir: 'output/runs/task', config_snapshot: 'configs/history/task/config.snapshot.toml',
      runtime_config_file: 'output/runs/task/config.runtime.toml',
      original_config_file: 'output/runs/task/config.original.toml',
      dataset_config_file: 'output/runs/task/dataset.runtime.toml',
      model_cache_dir: 'output/runs/task/model_cache', dataset_cache_dir: 'output/runs/task/dataset_cache',
      training_output_dir: 'output/runs/task/training_output',
      sample_dir: 'output/runs/task/training_output/sample', logs_dir: 'output/runs/task/model_cache/logs',
      logs_path: 'configs/history/task/logs.jsonl', metrics_path: 'configs/history/task/metrics.jsonl',
      system_path: 'configs/history/task/system.jsonl',
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
  previewOpenCount: root.querySelectorAll('[data-history-sample-open]').length,
  previewOverlayCount: root.querySelectorAll('.dragon-history-preview-open-label').length,
  pathPanelOpen: root.querySelector('[data-history-paths]')?.hasAttribute('open'),
  pathLabels: [...root.querySelectorAll('.dragon-history-path-list dt')].map((item) => item.textContent.trim()),
  pathValues: [...root.querySelectorAll('.dragon-history-path-list code')].map((item) => item.textContent.trim()),
  pathCopyCount: root.querySelectorAll('[data-history-path-copy]').length,
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
    assert payload["previewOpenCount"] == 2
    assert payload["previewOverlayCount"] == 0
    assert payload["pathPanelOpen"] is False
    assert payload["pathCopyCount"] == 15
    assert payload["pathLabels"][:3] == ["基础目录", "历史目录", "本次运行目录"]
    assert payload["pathValues"][0] == "/workspace/output/runs/task"
    assert payload["pathValues"][-1] == "/workspace/configs/history/task/config.snapshot.toml"
    assert len(payload["hrefs"]) == 5
    assert payload["hrefs"][0] == "#history/task%20%2F%201/overview"
    assert payload["labels"] == ["概览", "指标", "产物", "配置", "日志"]

