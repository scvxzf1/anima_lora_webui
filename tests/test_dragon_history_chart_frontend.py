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

function hoverMiddle() {
  const svg = root.querySelector('[data-history-chart]');
  svg.getBoundingClientRect = () => ({
    left: 0, top: 0, width: 900, height: 300, right: 900, bottom: 300,
  });
  root.querySelector('.dragon-history-chart-hitarea').dispatchEvent(
    new dom.window.MouseEvent('mousemove', { clientX: 441, clientY: 150, bubbles: true }),
  );
}

hoverMiddle();
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
hoverMiddle();
const curveToggle = {
  lossHidden: root.querySelector('[data-history-hover-loss-point]').hasAttribute('hidden'),
  lrHidden: root.querySelector('[data-history-hover-lr-point]').hasAttribute('hidden'),
};

for (const key of ['lossValue', 'lrValue']) {
  const input = root.querySelector(`[data-history-chart-toggle="${key}"]`);
  input.checked = false;
  input.dispatchEvent(new dom.window.Event('change', { bubbles: true }));
}
hoverMiddle();
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


def test_dragon_history_system_charts_render_and_hover_independently() -> None:
    """History metrics renders three real system series with matching hover scales."""
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
const vram = root.querySelector('[data-history-system-chart="vram"]');
vram.getBoundingClientRect = () => ({
  left: 0, top: 0, width: 420, height: 180, right: 420, bottom: 180,
});
vram.querySelector('.dragon-history-system-hitarea').dispatchEvent(
  new dom.window.MouseEvent('mousemove', { clientX: 406, clientY: 90, bubbles: true }),
);
const hover = vram.querySelector('[data-history-system-hover]');
const point = vram.querySelector('[data-history-system-hover-point]');
const result = {
  chartIds: [...root.querySelectorAll('[data-history-system-chart]')]
    .map((chart) => chart.dataset.historySystemChart),
  cardLabels: [...root.querySelectorAll('.dragon-history-system-card-head strong')]
    .map((label) => label.textContent.trim()),
  sampleLabel: root.querySelector('.dragon-history-system-head > span')?.textContent.trim(),
  summaries: [...root.querySelectorAll('.dragon-history-system-card-head span')]
    .map((label) => label.textContent.trim()),
  hoverHidden: hover.hasAttribute('hidden'),
  hoverX: point.getAttribute('cx'),
  hoverY: point.getAttribute('cy'),
  hoverText: vram.querySelector('[data-history-system-hover-text]').textContent,
  emptyHasTitle: emptyHtml.includes('系统趋势'),
  emptyHasMessage: emptyHtml.includes('当前任务没有可用的 GPU 系统采样'),
  emptyChartCount: (emptyHtml.match(/data-history-system-chart=/g) || []).length,
};
cleanup();
console.log(JSON.stringify(result));
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
    assert payload["chartIds"] == ["vram", "gpu", "temp"]
    assert payload["cardLabels"] == ["VRAM", "GPU 占用", "温度"]
    assert payload["sampleLabel"] == "最近 2 / 10 个采样"
    assert payload["summaries"] == [
        "最后 16 GB · 峰值 16 GB",
        "最后 80% · 峰值 80%",
        "最后 90°C · 峰值 90°C",
    ]
    assert payload["hoverHidden"] is False
    assert float(payload["hoverX"]) == pytest.approx(406)
    assert float(payload["hoverY"]) == pytest.approx(96)
    assert "16 GB" in payload["hoverText"]
    assert payload["emptyHasTitle"] is True
    assert payload["emptyHasMessage"] is True
    assert payload["emptyChartCount"] == 0


def test_dragon_history_system_charts_are_wired_to_detail_payload() -> None:
    controller = (REPO_ROOT / "web/static/js/dragon-ui/pages/history.js").read_text(encoding="utf-8")
    view = (REPO_ROOT / "web/static/js/dragon-ui/pages/history-view.js").read_text(encoding="utf-8")
    css = (REPO_ROOT / "web/static/css/dragon/03a-dragon-history-workbench.css").read_text(encoding="utf-8")

    assert "renderHistorySystemCharts(payload.system, payload.limits)" in controller
    assert "bindHistorySystemCharts(root, model.payload.system)" in controller
    assert "renderHistoryMetrics(metrics, lossChart, systemCharts)" in view
    assert "grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr))" in css
    assert "@container dragon-history-detail (min-width:1600px)" in css
