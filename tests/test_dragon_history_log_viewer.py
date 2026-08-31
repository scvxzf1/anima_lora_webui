from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_history_log_viewer_windows_segments_across_full_scroll_range() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for Dragon history log viewer checks")
    jsdom_api = REPO_ROOT / "web/frontend-next/node_modules/jsdom/lib/api.js"
    if not jsdom_api.exists():
        pytest.skip("jsdom is required for Dragon history log viewer checks")

    script = r"""
import { JSDOM } from './web/frontend-next/node_modules/jsdom/lib/api.js';
import { bindHistoryLogViewer } from './web/static/js/dragon-ui/pages/history-log-viewer.js';

const dom = new JSDOM(`<section data-history-detail-panel="logs" hidden>
  <span data-history-log-window-status></span>
  <div data-history-log-viewer role="list" style="--dragon-history-log-row-height: 28px; line-height: 28px"></div>
</section>`, { pretendToBeVisual: true });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.MutationObserver = dom.window.MutationObserver;
let triggerResize = () => {};
globalThis.ResizeObserver = class {
  constructor(callback) { triggerResize = callback; }
  observe() {}
  disconnect() {}
};
globalThis.requestAnimationFrame = (callback) => { callback(); return 1; };
globalThis.cancelAnimationFrame = () => {};

const panel = document.querySelector('[data-history-detail-panel]');
const viewer = document.querySelector('[data-history-log-viewer]');
Object.defineProperty(viewer, 'clientHeight', {
  configurable: true,
  get: () => panel.hidden ? 0 : 280,
});
const logs = Array.from({ length: 1000 }, (_, index) => ({
  line: `line-${index}${index === 500 ? '\ncontinued' : ''}`,
  level: index === 999 ? 'error' : 'info',
}));
const cleanup = bindHistoryLogViewer(panel, logs, { total: 1300 });

const hiddenState = {
  start: viewer.dataset.historyLogRenderStart,
  end: viewer.dataset.historyLogRenderEnd,
  mounted: viewer.querySelectorAll('.dragon-history-log-line').length,
};

panel.hidden = false;
await new Promise((resolve) => setTimeout(resolve, 0));
const latestState = {
  scrollTop: viewer.scrollTop,
  start: viewer.dataset.historyLogRenderStart,
  end: viewer.dataset.historyLogRenderEnd,
  mounted: viewer.querySelectorAll('.dragon-history-log-line').length,
  first: viewer.querySelector('.dragon-history-log-line')?.textContent,
  last: [...viewer.querySelectorAll('.dragon-history-log-line')].at(-1)?.textContent,
  status: document.querySelector('[data-history-log-window-status]').textContent,
  topSpacer: viewer.querySelector('[data-log-spacer="top"]')?.style.height,
  bottomSpacer: viewer.querySelector('[data-log-spacer="bottom"]')?.style.height,
};

viewer.scrollTop = 0;
viewer.dispatchEvent(new dom.window.Event('scroll'));
const topState = {
  start: viewer.dataset.historyLogRenderStart,
  end: viewer.dataset.historyLogRenderEnd,
  mounted: viewer.querySelectorAll('.dragon-history-log-line').length,
  first: viewer.querySelector('.dragon-history-log-line')?.textContent,
  status: document.querySelector('[data-history-log-window-status]').textContent,
};

viewer.scrollTop = 14000;
viewer.dispatchEvent(new dom.window.Event('scroll'));
const middleState = {
  start: viewer.dataset.historyLogRenderStart,
  end: viewer.dataset.historyLogRenderEnd,
  mounted: viewer.querySelectorAll('.dragon-history-log-line').length,
  topSpacer: viewer.querySelector('[data-log-spacer="top"]')?.style.height,
  bottomSpacer: viewer.querySelector('[data-log-spacer="bottom"]')?.style.height,
  hasContinuedLine: [...viewer.querySelectorAll('.dragon-history-log-line')]
    .some((line) => line.textContent === 'line-500 ↩ continued'),
};

viewer.style.setProperty('--dragon-history-log-row-height', '32px');
viewer.style.lineHeight = '32px';
triggerResize();
const resizedState = {
  scrollTop: viewer.scrollTop,
  start: viewer.dataset.historyLogRenderStart,
  end: viewer.dataset.historyLogRenderEnd,
  topSpacer: viewer.querySelector('[data-log-spacer="top"]')?.style.height,
  bottomSpacer: viewer.querySelector('[data-log-spacer="bottom"]')?.style.height,
};

cleanup();
const startBeforeDetachedScroll = viewer.dataset.historyLogRenderStart;
viewer.scrollTop = 0;
viewer.dispatchEvent(new dom.window.Event('scroll'));
const startAfterDetachedScroll = viewer.dataset.historyLogRenderStart;

console.log(JSON.stringify({
  hiddenState,
  latestState,
  topState,
  middleState,
  resizedState,
  startBeforeDetachedScroll,
  startAfterDetachedScroll,
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

    assert payload["hiddenState"] == {"start": "0", "end": "360", "mounted": 360}
    assert payload["latestState"] == {
        "scrollTop": 27720,
        "start": "840",
        "end": "1000",
        "mounted": 160,
        "first": "line-840",
        "last": "line-999",
        "status": "第 1291–1300 / 1300 行",
        "topSpacer": "23520px",
        "bottomSpacer": "0px",
    }
    assert payload["topState"] == {
        "start": "0",
        "end": "360",
        "mounted": 360,
        "first": "line-0",
        "status": "第 301–310 / 1300 行",
    }
    assert payload["middleState"] == {
        "start": "360",
        "end": "720",
        "mounted": 360,
        "topSpacer": "10080px",
        "bottomSpacer": "7840px",
        "hasContinuedLine": True,
    }
    assert payload["resizedState"] == {
        "scrollTop": 16000,
        "start": "360",
        "end": "720",
        "topSpacer": "11520px",
        "bottomSpacer": "8960px",
    }
    assert payload["startAfterDetachedScroll"] == payload["startBeforeDetachedScroll"]


def test_history_log_viewer_is_wired_without_twelve_line_slice() -> None:
    controller = (REPO_ROOT / "web/static/js/dragon-ui/pages/history-detail.js").read_text(encoding="utf-8")
    log_controller = (REPO_ROOT / "web/static/js/dragon-ui/pages/history-log-controller.js").read_text(encoding="utf-8")
    view = (REPO_ROOT / "web/static/js/dragon-ui/pages/history-view.js").read_text(encoding="utf-8")
    css = (REPO_ROOT / "web/static/css/dragon/03a-dragon-history-workbench.css").read_text(encoding="utf-8")

    assert "from './history-log-viewer.js" not in controller
    assert "import('./history-log-viewer.js?v=" in log_controller
    assert "module.bindHistoryLogViewer(root, model.payload.logs" in log_controller
    assert "activeTab === 'logs' ? loadHistoryLogPage(taskId)" in controller
    assert "detailController?.activateTab(model.activeTab)" in controller
    assert "logs.slice(-12)" not in view
    assert 'data-history-log-viewer tabindex="0" role="list"' in view
    assert 'data-history-log-search' in view
    assert 'data-history-log-search-previous' in view
    assert 'data-history-log-search-next' in view
    assert "height:clamp(240px,66vh,820px);height:clamp(240px,66dvh,820px)" in css
    assert "--dragon-history-log-row-height:22px" in css
    assert "height:var(--dragon-history-log-row-height)" in css
    assert "white-space:pre" in css
    viewer = (REPO_ROOT / "web/static/js/dragon-ui/pages/history-log-viewer.js").read_text(encoding="utf-8")
    assert "resolveLogRowHeight(viewer)" in viewer
    assert "line.dataset.searchActive = 'true'" in viewer


@pytest.mark.integration
def test_history_log_viewer_loads_arbitrary_pages_without_mounting_all_rows() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for Dragon history log viewer checks")
    jsdom_api = REPO_ROOT / "web/frontend-next/node_modules/jsdom/lib/api.js"
    if not jsdom_api.exists():
        pytest.skip("jsdom is required for Dragon history log viewer checks")

    script = r"""
import { JSDOM } from './web/frontend-next/node_modules/jsdom/lib/api.js';
import { bindHistoryLogViewer } from './web/static/js/dragon-ui/pages/history-log-viewer.js';

const dom = new JSDOM(`<section data-history-detail-panel="logs">
  <input type="search" data-history-log-search>
  <span data-history-log-search-status></span>
  <button data-history-log-search-previous></button>
  <button data-history-log-search-next></button>
  <span data-history-log-window-status></span>
  <div data-history-log-viewer role="list"></div>
</section>`, { pretendToBeVisual: true });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.MutationObserver = dom.window.MutationObserver;
globalThis.ResizeObserver = undefined;
globalThis.requestAnimationFrame = (callback) => { callback(); return 1; };
globalThis.cancelAnimationFrame = () => {};

const viewer = document.querySelector('[data-history-log-viewer]');
Object.defineProperty(viewer, 'clientHeight', { configurable: true, value: 220 });
const total = 10000;
const initialOffset = total - 360;
const initial = Array.from({ length: 360 }, (_, index) => ({ line: `line-${initialOffset + index}` }));
const requests = [];
const searchRequests = [];
const loadRange = async (offset, limit) => {
  requests.push({ offset, limit });
  return {
    ok: true,
    offset,
    total,
    logs: Array.from({ length: limit }, (_, index) => ({ line: `line-${offset + index}` })),
  };
};
const searchMatch = async (query, cursor, direction) => {
  searchRequests.push({ query, cursor, direction });
  return { ok: true, match_index: 20, match_ordinal: 1, matches_total: 1, total };
};
const cleanup = bindHistoryLogViewer(document, initial, {
  total,
  offset: initialOffset,
  loadRange,
  searchMatch,
});
await new Promise((resolve) => setTimeout(resolve, 0));
const latest = {
  last: [...viewer.querySelectorAll('.dragon-history-log-line')].at(-1)?.textContent,
  mounted: viewer.querySelectorAll('.dragon-history-log-line').length,
  requests: requests.length,
};

viewer.scrollTop = 0;
viewer.dispatchEvent(new dom.window.Event('scroll'));
await new Promise((resolve) => setTimeout(resolve, 0));
const top = {
  first: viewer.querySelector('.dragon-history-log-line')?.textContent,
  mounted: viewer.querySelectorAll('.dragon-history-log-line').length,
  status: document.querySelector('[data-history-log-window-status]').textContent,
  request: requests.find((item) => item.offset === 0),
};

viewer.scrollTop = 5000 * 22;
viewer.dispatchEvent(new dom.window.Event('scroll'));
await new Promise((resolve) => setTimeout(resolve, 0));
const middle = {
  first: viewer.querySelector('.dragon-history-log-line')?.textContent,
  mounted: viewer.querySelectorAll('.dragon-history-log-line').length,
  start: viewer.dataset.historyLogRenderStart,
  requestedOffsets: requests.map((item) => item.offset),
};
const input = document.querySelector('[data-history-log-search]');
input.value = 'needle';
input.dispatchEvent(new dom.window.Event('input'));
await new Promise((resolve) => setTimeout(resolve, 250));
const search = {
  status: document.querySelector('[data-history-log-search-status]').textContent,
  activeIndex: viewer.querySelector('[data-search-active="true"]')?.dataset.logIndex,
  request: searchRequests[0],
};
cleanup();
console.log(JSON.stringify({ latest, top, middle, search }));
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

    assert payload["latest"] == {"last": "line-9999", "mounted": 160, "requests": 0}
    assert payload["top"] == {
        "first": "line-0",
        "mounted": 360,
        "status": "第 1–10 / 10000 行",
        "request": {"offset": 0, "limit": 360},
    }
    assert payload["middle"]["first"] == "line-4800"
    assert payload["middle"]["mounted"] == 360
    assert payload["middle"]["start"] == "4800"
    assert 4680 in payload["middle"]["requestedOffsets"]
    assert 5040 in payload["middle"]["requestedOffsets"]
    assert payload["search"] == {
        "status": "1 / 1",
        "activeIndex": "20",
        "request": {"query": "needle", "cursor": 5000, "direction": "forward"},
    }


@pytest.mark.integration
def test_history_log_search_navigates_full_virtualized_range() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for Dragon history log search checks")
    jsdom_api = REPO_ROOT / "web/frontend-next/node_modules/jsdom/lib/api.js"
    if not jsdom_api.exists():
        pytest.skip("jsdom is required for Dragon history log search checks")

    script = r"""
import { JSDOM } from './web/frontend-next/node_modules/jsdom/lib/api.js';
import { bindHistoryLogViewer } from './web/static/js/dragon-ui/pages/history-log-viewer.js';

const dom = new JSDOM(`<section data-history-detail-panel="logs">
  <input type="search" data-history-log-search>
  <span data-history-log-search-status></span>
  <button data-history-log-search-previous></button>
  <button data-history-log-search-next></button>
  <span data-history-log-window-status></span>
  <div data-history-log-viewer role="list"></div>
</section>`, { pretendToBeVisual: true });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.MutationObserver = dom.window.MutationObserver;
globalThis.ResizeObserver = undefined;
globalThis.requestAnimationFrame = (callback) => { callback(); return 1; };
globalThis.cancelAnimationFrame = () => {};

const viewer = document.querySelector('[data-history-log-viewer]');
Object.defineProperty(viewer, 'clientHeight', { configurable: true, value: 220 });
const logs = Array.from({ length: 1000 }, (_, index) => ({
  line: index === 20 ? 'first Needle match' : index === 500 ? 'middle needle match' : index === 999 ? 'last NEEDLE match' : `line-${index}`,
}));
const cleanup = bindHistoryLogViewer(document, logs, { total: 1300 });
const input = document.querySelector('[data-history-log-search]');
const previous = document.querySelector('[data-history-log-search-previous]');
const next = document.querySelector('[data-history-log-search-next]');
const status = document.querySelector('[data-history-log-search-status]');

input.value = 'needle';
input.dispatchEvent(new dom.window.Event('input'));
const latestMatch = {
  status: status.textContent,
  row: viewer.querySelector('[data-search-active="true"]')?.dataset.logIndex,
  mark: viewer.querySelector('[data-search-active="true"] mark')?.textContent,
  disabled: next.disabled,
};

next.click();
const wrappedMatch = {
  status: status.textContent,
  row: viewer.querySelector('[data-search-active="true"]')?.dataset.logIndex,
  text: viewer.querySelector('[data-search-active="true"]')?.textContent,
};

input.dispatchEvent(new dom.window.KeyboardEvent('keydown', { key: 'Enter' }));
const middleMatch = {
  status: status.textContent,
  row: viewer.querySelector('[data-search-active="true"]')?.dataset.logIndex,
  mounted: viewer.querySelectorAll('.dragon-history-log-line').length,
};

previous.click();
input.dispatchEvent(new dom.window.KeyboardEvent('keydown', { key: 'Escape' }));
const cleared = {
  value: input.value,
  status: status.textContent,
  active: viewer.querySelector('[data-search-active="true"]') !== null,
  marks: viewer.querySelectorAll('mark').length,
  disabled: next.disabled && previous.disabled,
};
cleanup();

console.log(JSON.stringify({ latestMatch, wrappedMatch, middleMatch, cleared }));
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

    assert payload["latestMatch"] == {
        "status": "3 / 3",
        "row": "1299",
        "mark": "NEEDLE",
        "disabled": False,
    }
    assert payload["wrappedMatch"] == {
        "status": "1 / 3",
        "row": "320",
        "text": "first Needle match",
    }
    assert payload["middleMatch"] == {
        "status": "2 / 3",
        "row": "800",
        "mounted": 360,
    }
    assert payload["cleared"] == {
        "value": "",
        "status": "0 个匹配",
        "active": False,
        "marks": 0,
        "disabled": True,
    }
