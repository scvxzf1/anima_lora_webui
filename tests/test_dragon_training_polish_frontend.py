from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _node(script: str) -> dict:
    if not shutil.which("node"):
        pytest.skip("node is required for Dragon frontend checks")
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    return json.loads(proc.stdout)


def test_training_trend_helpers_smooth_without_changing_length() -> None:
    payload = _node(
        """
import { areaSvgPath, emaValues, smoothSvgPath } from './web/static/js/dragon-ui/pages/trend-utils.js';
const values = emaValues([1, 0, 1], .5);
const path = smoothSvgPath([[0, 10], [10, 5], [20, 8]], .2);
console.log(JSON.stringify({ values, path, area: areaSvgPath(path, [[0, 10], [10, 5], [20, 8]], 12) }));
"""
    )
    assert payload["values"] == [1, 0.5, 0.75]
    assert payload["path"].startswith("M0.00 10.00C")
    assert payload["area"].endswith("L20.00 12.00L0.00 12.00Z")


def test_log_highlighter_is_safe_and_marks_training_tokens() -> None:
    payload = _node(
        """
import { highlightedLogHtml } from './web/static/js/dragon-ui/pages/log-highlighter.js';
const html = highlightedLogHtml('<script> 2026-08-25 06:10:42 INFO [12/100] 100% ok');
console.log(JSON.stringify({ html }));
"""
    )
    html = payload["html"]
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    for token in ('data-token="time"', 'data-token="meta"', 'data-token="step"', 'data-token="success"'):
        assert token in html


def test_history_chart_hover_exposes_glass_tooltip() -> None:
    jsdom_api = REPO_ROOT / "web/frontend-next/node_modules/jsdom/lib/api.js"
    if not jsdom_api.exists():
        pytest.skip("jsdom is required for Dragon chart interaction checks")
    payload = _node(
        r"""
import { JSDOM } from './web/frontend-next/node_modules/jsdom/lib/api.js';
import { bindHistoryChart } from './web/static/js/dragon-ui/pages/history-chart.js';
const dom = new JSDOM('<div id="root"><div data-history-chart-container></div></div>', { pretendToBeVisual: true });
globalThis.document = dom.window.document;
globalThis.DOMPoint = undefined;
const root = document.querySelector('#root');
const cleanup = bindHistoryChart(root, [{ loss: .8, lr: 1e-5 }, { loss: .4, lr: 2e-5 }]);
const svg = root.querySelector('[data-history-chart]');
svg.getBoundingClientRect = () => ({ left: 0, top: 0, width: 900, height: 300, right: 900, bottom: 300 });
const moveEvent = typeof dom.window.PointerEvent === 'function' ? 'pointermove' : 'mousemove';
root.querySelector('.dragon-history-chart-hitarea').dispatchEvent(new dom.window.MouseEvent(moveEvent, { clientX: 441, clientY: 150, bubbles: true }));
await new Promise((resolve) => dom.window.requestAnimationFrame(resolve));
const tooltip = root.querySelector('[data-history-chart-tooltip]');
console.log(JSON.stringify({ hidden: tooltip.hidden, text: tooltip.textContent.trim() }));
cleanup();
"""
    )
    assert payload["hidden"] is False
    assert "STEP" in payload["text"] and "Loss" in payload["text"] and "LR" in payload["text"]


def test_history_detail_renders_summary_final_weight_and_lightbox_navigation() -> None:
    payload = _node(
        r"""
import { renderHistoryDetailPage } from './web/static/js/dragon-ui/pages/history-view.js';
const html = renderHistoryDetailPage({
  taskId: 'task-1', activeTab: 'artifacts', lossChart: '', systemCharts: '', resume: {},
  payload: { task: { id: 'task-1', state: 'completed', job: 'training' }, metrics: [], logs: [], config_toml: 'model_family = "krea2"\ntrain_batch_size = 2\nlearning_rate = 2e-5\nnetwork_dim = 16\nnetwork_alpha = 16\noptimizer_type = "CAME"' },
  images: { images: [{ name: 'step.png', url: '/step.png', sample: { step: 2500 } }] },
  weights: { weights: [{ name: 'adapter-final.safetensors', abs_path: '/tmp/adapter-final.safetensors', download_url: '/download' }] },
});
console.log(JSON.stringify({ html }));
"""
    )["html"]
    assert "dragon-history-preview-step" in payload
    assert "Step 2500" in payload
    assert "Final Model" in payload
    assert 'data-history-sample-action="previous"' in payload
    assert 'data-history-sample-action="next"' in payload


def test_training_polish_styles_are_reachable_and_responsive() -> None:
    route_styles = (REPO_ROOT / "web/static/js/dragon-ui/route-styles.js").read_text(encoding="utf-8")
    css = (REPO_ROOT / "web/static/css/dragon/07-dragon-training-polish.css").read_text(encoding="utf-8")
    workbench = (REPO_ROOT / "web/static/css/dragon/07a-dragon-live-workbench.css").read_text(encoding="utf-8")
    tokens = (REPO_ROOT / "web/static/css/dragon/00-dragon-tokens.css").read_text(encoding="utf-8")
    assert "07-dragon-training-polish.css" in route_styles
    assert "max-width: 1600px" in css
    width_rules = css.split(".dragon-tool-panel", 1)[0]
    assert ".dragon-tool-page" in width_rules
    assert ".dragon-history-page" not in width_rules
    assert ".dragon-history-detail-page" not in width_rules
    assert ".dragon-live-log-stream" in css and "#0f172a" in css
    assert ".dragon-history-config-summary" in css
    assert "@media (max-width: 460px)" in css
    assert "07a-dragon-live-workbench.css" in route_styles
    assert "grid-template-columns: 284px minmax(0, 1fr)" in workbench
    assert "@media (max-width: 1199px)" in workbench
    assert "white-space: pre" in workbench and "word-break: normal" in workbench
    assert ".dragon-live-hardware-card[data-tone=\"warning\"]" in workbench
    assert '@font-face' in tokens
    assert '"Inter"' in tokens and '"JetBrains Mono"' in tokens


def test_live_workbench_renders_queue_sidebar_and_all_runtime_states() -> None:
    payload = _node(
        r"""
import { createLiveModel } from './web/static/js/dragon-ui/pages/live-training-state.js';
import { applyWorkspaceSnapshot, liveWorkspaceMode } from './web/static/js/dragon-ui/pages/live-training-workspace.js';
import { renderLiveTrainingPage } from './web/static/js/dragon-ui/pages/live-training-view.js';
const model = createLiveModel({
  status: 'running', task_id: 'task-110', variant: '8-24-test', preset: 'default',
  latest_progress: { current: 110, total: 1100, rate: '9.98s/it', epoch: 11 },
  latest_metric: { loss: .4153, lr: 1.986e-5 },
  latest_system: { vram_used_gb: 21.8, vram_total_gb: 64, gpu_util: 100, gpu_temp: 82 },
}, [{ loss: .44 }, { loss: .4153 }], { records: [{ id: 1, level: 'info', message: '/long/path/model.safetensors' }] });
applyWorkspaceSnapshot(model, { summary: { queued: 2, running: 1 }, items: [{ id: 'q1', state: 'queued', variant: 'next-job' }] }, { tasks: [{ id: 'h1', state: 'completed', variant: 'done-job' }] });
const html = renderLiveTrainingPage(model, () => '<svg data-chart></svg>');
console.log(JSON.stringify({ html, modes: ['idle', 'running', 'error', 'interrupted'].map(liveWorkspaceMode) }));
"""
    )
    html = payload["html"]
    assert payload["modes"] == ["idle", "running", "error", "error"]
    assert "dragon-live-sidebar" in html and "排队中" in html and "next-job" in html
    assert "等待任务估算" in html
    assert "实时损失" in html and "实时学习率" in html and "速度与 ETA" in html
    assert "FINAL LOSS" not in html
    assert 'data-live-section="idle"' in html and 'data-live-section="error"' in html
    assert 'data-live-section="running"' in html
    assert 'data-tone="warning"' in html and "高温预警" in html
    assert "dragon-live-stop-dialog" in html
