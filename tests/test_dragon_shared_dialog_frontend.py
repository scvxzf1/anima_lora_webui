from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from tests.frontend_test_support import REPO_ROOT, STATIC_DIR, node_syntax_check


def _read(relative: str) -> str:
    return (STATIC_DIR / relative).read_text(encoding="utf-8")


def test_dragon_actions_use_shared_dialog_instead_of_browser_prompts() -> None:
    dataset = _read("js/dragon-ui/pages/dataset-editor.js")
    config = _read("js/dragon-ui/pages/config-page.js")
    training_library = _read("js/dragon-ui/pages/training-preset-library.js")
    preview = _read("js/features/dataset-editor/preview.js")
    stage = _read("js/features/config-form/stage-resolution-ui-dialog.js")
    shared = _read("js/shared/dialog.js")
    shared_css = _read("css/dragon/06a-dragon-shared-dialogs.css")
    route_styles = _read("js/dragon-ui/route-styles.js")

    for source in (dataset, config, training_library):
        assert "window.prompt" not in source
        assert "window.confirm" not in source
        assert "window.alert" not in source
    assert "window.alert" not in preview
    assert "window.prompt" not in stage
    assert "alertDragonDialog" in config
    assert "confirmDragonDialog" in config
    assert "confirmDragonDialog" in training_library
    assert "promptDragonDialog" in training_library
    for export_name in ("openDragonDialog", "confirmDragonDialog", "promptDragonDialog", "alertDragonDialog"):
        assert f"export function {export_name}" in shared
    assert "06a-dragon-shared-dialogs.css?v=dragon-ui-20260902v78" in route_styles
    assert "dragon-dialog-host" in shared
    assert "aria-labelledby" in shared
    assert ".dragon-dialog-host [hidden]" in shared_css
    assert node_syntax_check("js/shared/dialog.js").returncode == 0


def test_training_dialog_modules_share_one_context_cache_token() -> None:
    token = "dragon-ui-20260901v115"
    config = _read("js/dragon-ui/pages/config-page.js")
    training_library = _read("js/dragon-ui/pages/training-preset-library.js")

    for relative in (
        "js/dragon-ui/pages/config-page.js",
        "js/dragon-ui/pages/training-preset-library.js",
        "js/dragon-ui/pages/dataset-editor.js",
        "js/dragon-ui/pages/image-test.js",
        "js/dragon-ui/pages/tagging.js",
    ):
        assert f"training-controls.js?v={token}" in _read(relative)
    assert "training-preset-library.js?v=dragon-ui-20260901v116" in config
    assert "shared/dialog.js?v=module-bootstrap-20260901-dialog-v1" in config
    assert "shared/dialog.js?v=module-bootstrap-20260901-dialog-v1" in training_library


@pytest.mark.integration
def test_dragon_dialog_prompt_confirm_queue_and_focus() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for Dragon dialog interaction checks")
    if not (REPO_ROOT / "web/frontend-next/node_modules/jsdom/lib/api.js").exists():
        pytest.skip("jsdom is required for Dragon dialog interaction checks")

    script = r"""
import { JSDOM } from './web/frontend-next/node_modules/jsdom/lib/api.js';
import {
  alertDragonDialog,
  bindDialogBackdropClose,
  confirmDragonDialog,
  promptDragonDialog,
} from './web/static/js/shared/dialog.js?dialog-runtime-test';

const dom = new JSDOM(`<!doctype html><html><body data-dragon-ui><button id="trigger">open</button></body></html>`, {
  pretendToBeVisual: true,
  url: 'http://localhost/',
});
globalThis.window = dom.window;
globalThis.document = dom.window.document;

const trigger = document.querySelector('#trigger');
trigger.focus();
window.confirm = () => { throw new Error('native confirm must not be used'); };
window.prompt = () => { throw new Error('native prompt must not be used'); };
window.alert = () => { throw new Error('native alert must not be used'); };

const promptPromise = promptDragonDialog({
  title: '新建预设分组',
  message: '请输入名称。',
  label: '分组名称',
  value: 'dataset',
});
await Promise.resolve();
const host = document.querySelector('[data-dragon-dialog-host]');
const promptOpened = {
  mode: host.dataset.mode,
  value: host.querySelector('input').value,
  fallback: host.dataset.fallbackOpen === 'true',
  focused: document.activeElement === host.querySelector('input'),
};
host.querySelector('input').value = 'new-dataset';
host.querySelector('[data-dragon-dialog-confirm]').click();
const promptValue = await promptPromise;

const confirmPromise = confirmDragonDialog({ title: '确认删除？', tone: 'danger' });
await Promise.resolve();
const confirmHost = document.querySelector('[data-dragon-dialog-host]');
const confirmFieldHidden = confirmHost.querySelector('[data-dragon-dialog-field]').hidden;
const escapeEvent = new window.KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true });
document.dispatchEvent(escapeEvent);
const confirmValue = await confirmPromise;

const first = confirmDragonDialog({ title: 'first' });
const second = alertDragonDialog({ title: 'second' });
await Promise.resolve();
const queueHost = document.querySelector('[data-dragon-dialog-host]');
queueHost.querySelector('[data-dragon-dialog-confirm]').click();
const firstValue = await first;
const secondHost = document.querySelector('[data-dragon-dialog-host]');
const alertCancelHidden = secondHost.querySelector('[data-dragon-dialog-cancel]').hidden;
secondHost.querySelector('[data-dragon-dialog-close]').click();
await second;

const browseDialog = document.createElement('dialog');
browseDialog.setAttribute('open', '');
document.body.appendChild(browseDialog);
bindDialogBackdropClose(browseDialog);
browseDialog.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
const browseFallbackClosed = {
  hidden: browseDialog.hidden,
  open: browseDialog.hasAttribute('open'),
  ariaHidden: browseDialog.getAttribute('aria-hidden'),
};

console.log(JSON.stringify({
  promptOpened,
  promptValue,
  confirmFieldHidden,
  confirmValue,
  escapePrevented: escapeEvent.defaultPrevented,
  firstValue,
  alertCancelHidden,
  focusRestored: document.activeElement === trigger,
  hostHidden: secondHost.hidden,
  browseFallbackClosed,
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
        "promptOpened": {
            "mode": "prompt",
            "value": "dataset",
            "fallback": True,
            "focused": True,
        },
        "promptValue": "new-dataset",
        "confirmFieldHidden": True,
        "confirmValue": False,
        "escapePrevented": True,
        "firstValue": True,
        "alertCancelHidden": True,
        "focusRestored": True,
        "hostHidden": True,
        "browseFallbackClosed": {
            "hidden": True,
            "open": False,
            "ariaHidden": "true",
        },
    }


@pytest.mark.integration
def test_training_context_select_waits_for_async_discard_confirmation() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for Dragon training context interaction checks")
    jsdom_api = REPO_ROOT / "web/frontend-next/node_modules/jsdom/lib/api.js"
    if not jsdom_api.exists():
        pytest.skip("jsdom is required for Dragon training context interaction checks")

    script = r"""
import { pathToFileURL } from 'node:url';
const { JSDOM } = await import(pathToFileURL(process.argv[1]).href);
const dom = new JSDOM('<!doctype html><html><body data-dragon-ui></body></html>', {
  pretendToBeVisual: true,
  url: 'http://localhost/',
});
dom.window.fetch = globalThis.fetch.bind(globalThis);
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.CustomEvent = dom.window.CustomEvent;
const moduleUrl = pathToFileURL(process.argv[2]).href + '?training-context-dialog-test';
const controls = await import(moduleUrl);

const context = {
  files: [
    { path: 'configs/imported/current.toml', label: 'current', trainable: true },
    { path: 'configs/imported/next.toml', label: 'next', trainable: true },
  ],
  presets: ['default', 'fast'],
  gpus: [],
  gpuWhitelist: [],
  configFile: 'configs/imported/current.toml',
  preset: 'default',
};
const root = document.createElement('main');
root.innerHTML = controls.renderTrainingControls(context);
document.body.appendChild(root);

let settleConfirmation;
const transitions = [];
controls.bindTrainingControls(root, context, {
  beforeContextChange: () => new Promise((resolve) => { settleConfirmation = resolve; }),
  onConfigFileChange: (file) => transitions.push(file.path),
});

const select = root.querySelector('[data-training-context="file"]');
select.value = 'configs/imported/next.toml';
select.dispatchEvent(new window.Event('change', { bubbles: true }));
const valueWhilePending = select.value;
settleConfirmation(false);
await new Promise((resolve) => setTimeout(resolve, 0));
const cancelled = { value: select.value, transitions: [...transitions] };

select.value = 'configs/imported/next.toml';
select.dispatchEvent(new window.Event('change', { bubbles: true }));
settleConfirmation(true);
await new Promise((resolve) => setTimeout(resolve, 0));

console.log(JSON.stringify({
  valueWhilePending,
  cancelled,
  approved: { value: select.value, transitions },
}));
"""
    result = subprocess.run(
        [
            "node",
            "--input-type=module",
            "--eval",
            script,
            str(jsdom_api),
            str(STATIC_DIR / "js/dragon-ui/pages/training-controls.js"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert json.loads(result.stdout) == {
        "valueWhilePending": "configs/imported/current.toml",
        "cancelled": {
            "value": "configs/imported/current.toml",
            "transitions": [],
        },
        "approved": {
            "value": "configs/imported/current.toml",
            "transitions": ["configs/imported/next.toml"],
        },
    }
