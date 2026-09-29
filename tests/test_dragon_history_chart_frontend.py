from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_dragon_history_chart_hover_tracks_each_series() -> None:
    """The Next chart has no equivalent multi-series controls yet."""
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
