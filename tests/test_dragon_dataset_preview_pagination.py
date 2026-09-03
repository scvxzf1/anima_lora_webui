from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from tests.frontend_test_support import REPO_ROOT, STATIC_DIR, node_syntax_check


def _read(relative: str) -> str:
    return (STATIC_DIR / relative).read_text(encoding="utf-8")


def test_dragon_dataset_preview_uses_bounded_server_pagination() -> None:
    preview = _read("js/dragon-ui/pages/dataset-editor-preview.js")
    windowing = _read("js/dragon-ui/pages/dataset-preview-window.js")
    controller = _read("js/dragon-ui/pages/dataset-preview-controller.js")

    assert "const DATASET_PREVIEW_PAGE_SIZE = 24;" in preview
    assert "limit: String(DATASET_PREVIEW_PAGE_SIZE)" in preview
    assert "offset: String(Math.max(0, Number(offset) || 0))" in preview
    assert "requestSequence !== state.datasetPreviewRequestSequence" in preview
    assert 'data-dataset-preview-page="previous"' in preview
    assert 'data-dataset-preview-page="next"' in preview
    assert 'data-dataset-preview-page-input' in preview
    assert 'data-dataset-preview-jump' in preview
    assert "installDirectionalLoadObserver" in preview
    assert "DATASET_PREVIEW_BEFORE_ROOT_MARGIN" in preview
    assert "DATASET_PREVIEW_AFTER_ROOT_MARGIN" in preview
    assert "ensureDirectionalLoadControl(results, grid, 'before')" in preview
    assert 'data-dataset-preview-load-direction="${direction}"' in preview
    assert "payload.has_more_before" in preview
    assert "payload.has_more_after" in preview
    assert "DATASET_PREVIEW_MAX_RESIDENT_PAGES = 3" in windowing
    assert "preserveDatasetPreviewAnchor" in windowing
    assert "visibleDatasetPreviewOffset" in windowing
    assert "session.pageCache" in windowing
    assert "image.thumbnail_url || image.url" in preview
    assert "dialog.setAttribute('open', 'open')" in preview
    assert "await copyText(text)" in preview
    assert "renderIcon('copy', 'dragon-btn-icon')" in preview
    assert "dataset-editor-preview.js?v=dragon-ui-20260902v53" in controller
    assert "dataset-preview-window.js?v=dragon-ui-20260831v3" in preview


def test_dragon_dataset_preview_normalizes_new_and_legacy_pagination() -> None:
    script = r"""
const mod = await import('./web/static/js/dragon-ui/pages/dataset-editor-preview.js?pagination-runtime-test');
const current = mod.normalizeDatasetPreviewPayload({
  ok: true,
  images: [{ file: '048.png' }, { file: '049.png' }],
  total: 130,
  offset: 48,
  limit: 48,
  returned: 2,
  next_offset: 50,
  has_more_before: true,
  has_more_after: true,
}, 48, 48);
const legacy = mod.normalizeDatasetPreviewPayload({
  ok: true,
  images: [{ file: '000.png' }, { file: '001.png' }],
  total: 3,
  limit: 2,
}, 0, 2);
let legacyError = '';
try {
  mod.normalizeDatasetPreviewPayload({
    ok: true,
    images: [{ file: '000.png' }, { file: '001.png' }],
    total: 3,
    limit: 2,
  }, 2, 2);
} catch (error) {
  legacyError = error.message;
}
console.log(JSON.stringify({ current, legacy, legacyError }));
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
    assert payload["current"] == {
        "ok": True,
        "images": [{"file": "048.png"}, {"file": "049.png"}],
        "total": 130,
        "offset": 48,
        "limit": 48,
        "returned": 2,
        "next_offset": 50,
        "has_more_before": True,
        "has_more_after": True,
        "legacy_pagination": False,
    }
    assert payload["legacy"]["offset"] == 0
    assert payload["legacy"]["next_offset"] == 2
    assert payload["legacy"]["has_more_after"] is True
    assert payload["legacy"]["legacy_pagination"] is True
    assert payload["legacyError"] == "分页接口尚未生效，请重启 WebUI 服务后再翻页或继续加载。"


def test_dragon_dataset_preview_module_has_valid_syntax() -> None:
    result = node_syntax_check("js/dragon-ui/pages/dataset-editor-preview.js")
    assert result.returncode == 0, result.stderr or result.stdout
    result = node_syntax_check("js/dragon-ui/pages/dataset-preview-window.js")
    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.integration
def test_dragon_dataset_preview_window_keeps_anchor_and_bounds_resident_pages() -> None:
    script = r"""
import {
  createDatasetPreviewWindow,
  datasetPreviewRequestOffset,
  getCachedDatasetPreviewPage,
  mergeDatasetPreviewPage,
  preserveDatasetPreviewAnchor,
  trimDatasetPreviewWindow,
} from './web/static/js/dragon-ui/pages/dataset-preview-window.js?window-runtime-test';

const key = (image) => image.file;
const page = (offset, total = 120) => ({
  images: Array.from({ length: Math.min(24, total - offset) }, (_, index) => ({ file: `${offset + index}.png` })),
  total,
  offset,
  limit: 24,
  next_offset: Math.min(total, offset + 24),
  has_more_before: offset > 0,
  has_more_after: offset + 24 < total,
  legacy_pagination: false,
});
const session = createDatasetPreviewWindow(page(48), key);
const before = mergeDatasetPreviewPage(session, page(24), 'before', key);
mergeDatasetPreviewPage(session, page(72), 'after', key);
mergeDatasetPreviewPage(session, page(96), 'after', key);
const removed = trimDatasetPreviewWindow(session, 'after', key);
const cachedBefore = getCachedDatasetPreviewPage(session, 24);

let inserted = false;
const scroller = { scrollTop: 300 };
const anchor = { getBoundingClientRect: () => ({ top: inserted ? 450 : 100 }) };
preserveDatasetPreviewAnchor(scroller, anchor, () => { inserted = true; });

console.log(JSON.stringify({
  beforeOffset: before.offset,
  activeOffset: session.activeOffset,
  requestBefore: datasetPreviewRequestOffset(session, 'before'),
  requestAfter: datasetPreviewRequestOffset(session, 'after'),
  residentOffsets: [...session.pages.keys()].sort((a, b) => a - b),
  removed,
  cachedBeforeOffset: cachedBefore?.offset,
  scrollTop: scroller.scrollTop,
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
    assert json.loads(result.stdout) == {
        "beforeOffset": 24,
        "activeOffset": 48,
        "requestBefore": 24,
        "requestAfter": 120,
        "residentOffsets": [48, 72, 96],
        "removed": [24],
        "cachedBeforeOffset": 24,
        "scrollTop": 650,
    }


@pytest.mark.integration
def test_dragon_dataset_preview_detail_restores_list_state() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for Dragon dataset preview detail checks")
    jsdom_api = REPO_ROOT / "web/frontend-next/node_modules/jsdom/lib/api.js"
    if not jsdom_api.exists():
        pytest.skip("jsdom is required for Dragon dataset preview detail checks")

    script = r"""
import { JSDOM } from './web/frontend-next/node_modules/jsdom/lib/api.js';
import { createDatasetPreviewDetailController } from './web/static/js/dragon-ui/pages/dataset-preview-detail.js?detail-runtime-test';

const dom = new JSDOM(`
  <dialog id="dataset-preview-dialog">
    <form>
      <div class="dataset-preview-dialog-body">
        <div class="dataset-preview-results"><div id="dataset-preview-grid"></div></div>
        <button type="button" data-trigger>图片</button>
      </div>
    </form>
  </dialog>
`, { pretendToBeVisual: true });
globalThis.window = dom.window;
globalThis.document = dom.window.document;

const list = document.querySelector('.dataset-preview-dialog-body');
list.scrollTop = 240;
const trigger = document.querySelector('[data-trigger]');
trigger.focus();
const controller = createDatasetPreviewDetailController(document.querySelector('#dataset-preview-dialog'));
const image = {
  name: 'sample.png',
  file: '/datasets/sample.png',
  url: '/api/image/sample.png',
  width: 1024,
  height: 768,
  size_bytes: 2048,
  mtime_text: '2026-08-31 12:00:00',
  caption: { ok: true, format_label: '同名 TXT', file: '/datasets/sample.txt', text: 'one girl, blue sky' },
};
controller.open(image, trigger);
const opened = {
  listHidden: list.hidden,
  detailVisible: !document.querySelector('[data-dataset-preview-detail]').hidden,
  path: document.querySelectorAll('[data-dataset-preview-detail-meta] dd')[1]?.textContent,
  caption: document.querySelector('[data-dataset-preview-detail-caption-text]')?.textContent,
};
list.scrollTop = 0;
controller.restore({ defer: false });
const restored = {
  listHidden: list.hidden,
  detailHidden: document.querySelector('[data-dataset-preview-detail]').hidden,
  scrollTop: list.scrollTop,
  isOpen: controller.isOpen(),
};
list.scrollTop = 180;
controller.open(image, trigger);
const escapeEvent = new window.KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true });
document.querySelector('[data-dataset-preview-detail-back]').dispatchEvent(escapeEvent);
await new Promise((resolve) => window.requestAnimationFrame(() => window.requestAnimationFrame(resolve)));
const escaped = {
  defaultPrevented: escapeEvent.defaultPrevented,
  listHidden: list.hidden,
  detailHidden: document.querySelector('[data-dataset-preview-detail]').hidden,
  scrollTop: list.scrollTop,
  isOpen: controller.isOpen(),
};
console.log(JSON.stringify({ opened, restored, escaped }));
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
    assert payload["opened"] == {
        "listHidden": True,
        "detailVisible": True,
        "path": "/datasets/sample.png",
        "caption": "one girl, blue sky",
    }
    assert payload["restored"] == {
        "listHidden": False,
        "detailHidden": True,
        "scrollTop": 240,
        "isOpen": False,
    }
    assert payload["escaped"] == {
        "defaultPrevented": True,
        "listHidden": False,
        "detailHidden": True,
        "scrollTop": 180,
        "isOpen": False,
    }


def test_dragon_dataset_preview_pager_has_stable_responsive_styles() -> None:
    css = _read("css/dragon/06a-dragon-shared-dialogs.css")
    pages_css = _read("css/dragon/06-dragon-pages.css")
    route_styles = _read("js/dragon-ui/route-styles.js")

    pager_rule = css[
        css.index(".dataset-preview-pagination {"):
        css.index(".dataset-preview-pagination[hidden]")
    ]
    assert "position: sticky;" in pager_rule
    assert "display: grid;" in pager_rule
    assert "grid-template-columns: 32px minmax(0, auto) 32px;" in pager_rule
    assert "min-height: 36px;" in pager_rule
    assert "font-variant-numeric: tabular-nums;" in css
    assert "content-visibility: auto;" in css
    assert "contain-intrinsic-size: 380px;" in css
    image_rule = css[
        css.index(".dataset-preview-image-btn img {"):
        css.index(".dataset-preview-card-body {")
    ]
    assert "position: absolute;" in image_rule
    assert "inset: 0;" in image_rule
    assert "min-height: 0;" in image_rule
    assert "flex-direction: row;" in css
    assert "06a-dragon-shared-dialogs.css?v=dragon-ui-20260902v78" in route_styles
    assert "width: min(80vw, 1600px);" in pages_css
    assert "height: min(80vh, 960px);" in pages_css
    assert "grid-template-columns: repeat(auto-fit, minmax(min(240px, 100%), 1fr));" in css
    assert ".dataset-preview-detail-content" in css
    assert ".dataset-preview-dialog-body[hidden]" in css
