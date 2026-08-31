from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dragon_history_chart_hover_tracks_each_series() -> None:
    """Loss and LR hover markers use their own scales and visibility toggles."""
    if not shutil.which("node"):
        pytest.skip("node is required for Dragon history chart checks")
    jsdom_api = REPO_ROOT / "web/frontend-next/node_modules/jsdom/lib/api.js"
    if not jsdom_api.exists():
        pytest.skip("jsdom is required for Dragon history chart interaction checks")

    script = r"""
import { JSDOM } from './web/frontend-next/node_modules/jsdom/lib/api.js';
import { bindHistoryChart } from './web/static/js/dragon-ui/pages/history-chart.js';

const dom = new JSDOM(`<div id="root">
  <input type="checkbox" data-history-chart-toggle="lossCurve" checked>
  <input type="checkbox" data-history-chart-toggle="lrCurve" checked>
  <input type="checkbox" data-history-chart-toggle="lossValue" checked>
  <input type="checkbox" data-history-chart-toggle="lrValue" checked>
  <div data-history-chart-container></div>
</div>`, { pretendToBeVisual: true });
globalThis.document = dom.window.document;
globalThis.DOMPoint = undefined;

const root = document.querySelector('#root');
const cleanup = bindHistoryChart(root, [
  { loss: 0, lr: 0 },
  { loss: 0.5, lr: 0.25 },
  { loss: 1, lr: 1 },
]);

async function hoverMiddle() {
  const svg = root.querySelector('[data-history-chart]');
  svg.getBoundingClientRect = () => ({
    left: 0, top: 0, width: 900, height: 300, right: 900, bottom: 300,
  });
  const moveEvent = typeof dom.window.PointerEvent === 'function' ? 'pointermove' : 'mousemove';
  root.querySelector('.dragon-history-chart-hitarea').dispatchEvent(
    new dom.window.MouseEvent(moveEvent, { clientX: 441, clientY: 150, bubbles: true }),
  );
  await new Promise((resolve) => dom.window.requestAnimationFrame(resolve));
}

await hoverMiddle();
const initialLossPoint = root.querySelector('[data-history-hover-loss-point]');
const initialLrPoint = root.querySelector('[data-history-hover-lr-point]');
const initial = {
  lossX: initialLossPoint.getAttribute('cx'),
  lossY: initialLossPoint.getAttribute('cy'),
  lrX: initialLrPoint.getAttribute('cx'),
  lrY: initialLrPoint.getAttribute('cy'),
  lossHidden: initialLossPoint.hasAttribute('hidden'),
  lrHidden: initialLrPoint.hasAttribute('hidden'),
};

const lossCurve = root.querySelector('[data-history-chart-toggle="lossCurve"]');
lossCurve.checked = false;
lossCurve.dispatchEvent(new dom.window.Event('change', { bubbles: true }));
await hoverMiddle();
const curveToggle = {
  lossHidden: root.querySelector('[data-history-hover-loss-point]').hasAttribute('hidden'),
  lrHidden: root.querySelector('[data-history-hover-lr-point]').hasAttribute('hidden'),
};

for (const key of ['lossValue', 'lrValue']) {
  const input = root.querySelector(`[data-history-chart-toggle="${key}"]`);
  input.checked = false;
  input.dispatchEvent(new dom.window.Event('change', { bubbles: true }));
}
await hoverMiddle();
const valuesHidden = root.querySelector('[data-history-chart-values]').hasAttribute('hidden');
const lrPointStillVisible = !root.querySelector('[data-history-hover-lr-point]').hasAttribute('hidden');
cleanup();

console.log(JSON.stringify({ initial, curveToggle, valuesHidden, lrPointStillVisible }));
"""
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    payload = json.loads(proc.stdout)
    initial = payload["initial"]
    assert initial["lossX"] == initial["lrX"] == "441"
    assert initial["lossY"] != initial["lrY"]
    assert initial["lossHidden"] is False
    assert initial["lrHidden"] is False
    assert payload["curveToggle"] == {"lossHidden": True, "lrHidden": False}
    assert payload["valuesHidden"] is True
    assert payload["lrPointStillVisible"] is True


def test_dragon_history_system_chart_filters_series_and_renders_hover_values() -> None:
    """The full-width single-GPU chart filters series and links hover values."""
    if not shutil.which("node"):
        pytest.skip("node is required for Dragon history system chart checks")
    jsdom_api = REPO_ROOT / "web/frontend-next/node_modules/jsdom/lib/api.js"
    if not jsdom_api.exists():
        pytest.skip("jsdom is required for Dragon history system chart checks")

    script = r"""
import { JSDOM } from './web/frontend-next/node_modules/jsdom/lib/api.js';
import {
  bindHistorySystemCharts,
  renderHistorySystemCharts,
} from './web/static/js/dragon-ui/pages/history-system-charts.js';

const records = [
  { ts: 100, vram_used_gb: 4, vram_total_gb: 32, gpu_util: 20, gpu_temp: 60 },
  { ts: 200, vram_used_gb: 16, vram_total_gb: 32, gpu_util: 80, gpu_temp: 90 },
];
const html = renderHistorySystemCharts(records, { system_total: 10 });
const emptyHtml = renderHistorySystemCharts([], {});
const dom = new JSDOM(`<main id="root">${html}</main>`, { pretendToBeVisual: true });
globalThis.document = dom.window.document;
globalThis.DOMPoint = undefined;

const root = document.querySelector('#root');
const cleanup = bindHistorySystemCharts(root, records);
async function hoverLast() {
  const chart = root.querySelector('[data-history-system-chart]');
  chart.getBoundingClientRect = () => ({
    left: 0, top: 0, width: 900, height: 300, right: 900, bottom: 300,
  });
  const moveEvent = typeof dom.window.PointerEvent === 'function' ? 'pointermove' : 'mousemove';
  chart.querySelector('.dragon-history-system-hitarea').dispatchEvent(
    new dom.window.MouseEvent(moveEvent, { clientX: 826, clientY: 150, bubbles: true }),
  );
  await new Promise((resolve) => dom.window.requestAnimationFrame(resolve));
}

await hoverLast();
const initial = {
  chartCount: root.querySelectorAll('[data-history-system-chart]').length,
  series: [...root.querySelectorAll('[data-history-system-series]')]
    .map((series) => series.dataset.historySystemSeries),
  controls: [...root.querySelectorAll('[data-history-system-toggle]')]
    .map((input) => input.dataset.historySystemToggle),
  points: [...root.querySelectorAll('[data-history-system-hover-point]')]
    .map((point) => ({ id: point.dataset.historySystemHoverPoint, hidden: point.hasAttribute('hidden'), x: point.getAttribute('cx') })),
  tooltip: root.querySelector('[data-history-system-tooltip]').textContent.replace(/\s+/g, ' ').trim(),
  tooltipHidden: root.querySelector('[data-history-system-tooltip]').hasAttribute('hidden'),
  sampleLabel: root.querySelector('.dragon-history-panel-head > span')?.textContent.trim(),
  summaryValues: [...root.querySelectorAll('.dragon-history-system-summary strong')]
    .map((label) => label.textContent.trim()),
};

const gpuToggle = root.querySelector('[data-history-system-toggle="gpu"]');
gpuToggle.checked = false;
gpuToggle.dispatchEvent(new dom.window.Event('change', { bubbles: true }));
await hoverLast();
const filtered = {
  series: [...root.querySelectorAll('[data-history-system-series]')]
    .map((series) => series.dataset.historySystemSeries),
  points: [...root.querySelectorAll('[data-history-system-hover-point]')]
    .map((point) => point.dataset.historySystemHoverPoint),
  tooltip: root.querySelector('[data-history-system-tooltip]').textContent.replace(/\s+/g, ' ').trim(),
};
for (const id of ['vram', 'temp']) {
  const toggle = root.querySelector(`[data-history-system-toggle="${id}"]`);
  toggle.checked = false;
  toggle.dispatchEvent(new dom.window.Event('change', { bubbles: true }));
}
await hoverLast();
const allHidden = {
  seriesCount: root.querySelectorAll('[data-history-system-series]').length,
  tooltipHidden: root.querySelector('[data-history-system-tooltip]').hasAttribute('hidden'),
  hoverHidden: root.querySelector('[data-history-system-hover]').hasAttribute('hidden'),
};
const empty = {
  hasTitle: emptyHtml.includes('VRAM 与 GPU 状态'),
  hasMessage: emptyHtml.includes('当前任务没有可用的 GPU 系统采样'),
  chartCount: (emptyHtml.match(/data-history-system-chart/g) || []).length,
};
cleanup();
console.log(JSON.stringify({ initial, filtered, allHidden, empty }));
"""
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    payload = json.loads(proc.stdout)
    initial = payload["initial"]
    assert initial["chartCount"] == 1
    assert initial["series"] == ["vram-area", "vram", "gpu", "temp"]
    assert initial["controls"] == ["vram", "gpu", "temp"]
    assert [point["id"] for point in initial["points"]] == ["vram", "gpu", "temp"]
    assert all(point["hidden"] is False for point in initial["points"])
    assert len({point["x"] for point in initial["points"]}) == 1
    assert initial["tooltipHidden"] is False
    assert "VRAM16 GB" in initial["tooltip"]
    assert "GPU80%" in initial["tooltip"]
    assert "温度90°C" in initial["tooltip"]
    assert "总显存 32 GB" in initial["tooltip"]
    assert initial["sampleLabel"] == "最近 2 / 10 个采样"
    assert initial["summaryValues"] == ["16 GB", "80%", "90°C"]
    assert payload["filtered"]["series"] == ["vram-area", "vram", "temp"]
    assert payload["filtered"]["points"] == ["vram", "temp"]
    assert "GPU80%" not in payload["filtered"]["tooltip"]
    assert payload["allHidden"] == {"seriesCount": 0, "tooltipHidden": True, "hoverHidden": True}
    assert payload["empty"] == {"hasTitle": True, "hasMessage": True, "chartCount": 0}


def test_dragon_history_system_charts_are_wired_to_detail_payload() -> None:
    controller = (REPO_ROOT / "web/static/js/dragon-ui/pages/history-detail.js").read_text(encoding="utf-8")
    metrics_controller = (REPO_ROOT / "web/static/js/dragon-ui/pages/history-metrics-controller.js").read_text(encoding="utf-8")
    view = (REPO_ROOT / "web/static/js/dragon-ui/pages/history-view.js").read_text(encoding="utf-8")
    css = (REPO_ROOT / "web/static/css/dragon/03a-dragon-history-workbench.css").read_text(encoding="utf-8")

    assert "renderHistorySystemCharts(model.payload.system, model.payload.limits)" in metrics_controller
    assert "bindHistorySystemCharts(root, model.payload.system)" in metrics_controller
    assert "renderHistoryMetrics(metrics, lossChart, systemCharts)" in view
    assert "dragon-history-system-chart-shell" in css
    assert "aspect-ratio:3/1" in css
    assert "grid-template-columns:repeat(3,minmax(0,1fr))" in css
    assert "@container dragon-history-detail (min-width:1600px)" not in css
