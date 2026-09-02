from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from tests.frontend_test_support import REPO_ROOT, STATIC_DIR, node_syntax_check


def _read(relative: str) -> str:
    return (STATIC_DIR / relative).read_text(encoding="utf-8")


def _run_node(script: str) -> dict:
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return json.loads(result.stdout)


def test_sample_prompts_field_uses_a_scoped_editor_dialog() -> None:
    page = _read("js/dragon-ui/pages/config-page.js")
    editor = _read("js/dragon-ui/pages/sample-prompts-dialog.js")
    css = _read("css/dragon/04e-dragon-sample-prompts.css")
    route_styles = _read("js/dragon-ui/route-styles.js")

    assert "renderSamplePromptsFieldControl" in page
    assert "bindSamplePromptsDialog(wrapper, { trainingContext })" in page
    assert "sample-prompts-dialog.js?v=dragon-ui-20260902-sample-prompts-v4" in page
    assert "state.samplePromptsCleanup?.()" in page
    assert 'key === \'sample_prompts\'' in page
    assert 'aria-haspopup="dialog"' in editor
    assert 'data-sample-prompts-action="add"' in editor
    assert 'data-sample-prompts-action="remove"' in editor
    assert 'data-sample-prompts-action="apply-uniform"' in editor
    assert "添加提示词" in editor
    assert "删除提示词" in editor
    assert "风格" not in editor
    assert 'data-sample-prompts-mode="raw"' in editor
    assert 'role="tabpanel"' in editor
    assert "handleModeKeydown" in editor
    assert "requestIsCurrent" in editor
    assert "document.body.appendChild(this.dialog)" in editor
    assert "this.dialog.remove()" in editor
    assert "/api/config/sample-prompts" in editor
    assert "train_config_file: this.trainingContext.configFile || null" in editor
    assert "04e-dragon-sample-prompts.css?v=dragon-ui-20260902-sample-prompts-v2" in route_styles
    assert "@media (max-width: 560px)" in css
    assert ".dragon-sample-prompts-uniform[hidden]" in css
    assert node_syntax_check("js/dragon-ui/pages/sample-prompts-dialog.js").returncode == 0


def test_sample_prompts_model_unifies_rows_preserves_extensions_and_validates() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for sample prompt model checks")
    module_uri = (STATIC_DIR / "js/dragon-ui/pages/sample-prompts-dialog.js").resolve().as_uri()
    script = f"""
const mod = await import({json.dumps(module_uri + '?sample-prompts-model-test')});
const rows = [
  {{ prompt: 'style one', width: '512', height: '512', steps: '20', cfg: '3', extra: '--custom keep' }},
  {{ prompt: 'style two', width: '768', height: '512', steps: '30', cfg: '5' }},
];
const unified = mod.applyUniformSamplePromptValues(rows, {{ width: '1024', height: '', steps: '28', cfg: '4' }});
console.log(JSON.stringify({{
  unified,
  mixedWidth: mod.commonSamplePromptValue(rows, 'width'),
  commonHeight: mod.commonSamplePromptValue(rows, 'height'),
  serialized: mod.serializeStructuredSamplePrompts(unified, '# keep this comment\\nold prompt'),
  valid: mod.validateSamplePromptRows(unified),
  invalidWidth: mod.validateSamplePromptRows([{{ prompt: 'bad', width: '32' }}]),
  invalidSteps: mod.validateSamplePromptRows([{{ prompt: 'bad', steps: '1001' }}]),
  invalidFlowShift: mod.validateSamplePromptRows([{{ prompt: 'bad', flow_shift: '-1' }}]),
  scientificFlowShift: mod.validateSamplePromptRows([{{ prompt: 'bad', flow_shift: '1e-3' }}]),
  missingPrompt: mod.validateSamplePromptRows([{{ prompt: '', cfg: '4' }}]),
  legacyField: mod.renderSamplePromptsFieldControl({{
    fieldId: 'legacy-prompts', name: 'sample_prompts', value: 'configs/prompts.toml',
  }}),
}}));
"""
    payload = _run_node(script)

    assert [row["width"] for row in payload["unified"]] == ["1024", "1024"]
    assert [row["height"] for row in payload["unified"]] == ["512", "512"]
    assert [row["steps"] for row in payload["unified"]] == ["28", "28"]
    assert [row["cfg"] for row in payload["unified"]] == ["4", "4"]
    assert payload["mixedWidth"] == {"value": "", "mixed": True}
    assert payload["commonHeight"] == {"value": "512", "mixed": False}
    assert payload["serialized"].startswith("# keep this comment\n\n")
    assert "style one --w 1024 --h 512 --s 28 --g 4 --custom keep" in payload["serialized"]
    assert payload["valid"] == {"ok": True}
    assert payload["invalidWidth"]["field"] == "width"
    assert payload["invalidSteps"]["field"] == "steps"
    assert payload["invalidFlowShift"]["field"] == "flow_shift"
    assert payload["scientificFlowShift"]["field"] == "flow_shift"
    assert payload["missingPrompt"]["field"] == "prompt"
    assert 'data-key="sample_prompts"' in payload["legacyField"]
    assert "data-sample-prompts-open" not in payload["legacyField"]


@pytest.mark.integration
def test_sample_prompts_dialog_loads_unifies_adds_and_saves() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for sample prompt dialog checks")
    jsdom = REPO_ROOT / "web/frontend-next/node_modules/jsdom/lib/api.js"
    if not jsdom.exists():
        pytest.skip("jsdom is required for sample prompt dialog checks")

    script = r"""
import { JSDOM } from './web/frontend-next/node_modules/jsdom/lib/api.js';

const dom = new JSDOM('<!doctype html><html><body data-dragon-ui><main id="root"></main></body></html>', {
  pretendToBeVisual: true,
  url: 'http://localhost/',
});
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.Event = dom.window.Event;
globalThis.CustomEvent = dom.window.CustomEvent;

const mod = await import('./web/static/js/dragon-ui/pages/sample-prompts-dialog.js?sample-prompts-runtime-test');
const root = document.querySelector('#root');
root.innerHTML = mod.renderSamplePromptsFieldControl({
  fieldId: 'sample-prompts',
  name: 'sample_prompts',
  value: 'configs/sample-prompts/imported/original.txt',
}) + mod.renderSamplePromptsDialog();

const dialog = root.querySelector('[data-sample-prompts-dialog]');
dialog.showModal = function showModal() { this.setAttribute('open', ''); };
dialog.close = function close() { this.removeAttribute('open'); };

const calls = [];
const initialContent = [
  'style one --w 512 --h 512 --s 20 --g 3 --custom keep',
  'style two --w 768 --h 512 --s 30 --g 5',
].join('\n');
const api = async (url, options = {}) => {
  calls.push({ url, options });
  if (options.method === 'PUT') {
    const body = JSON.parse(options.body);
    return {
      ok: true,
      file: 'configs/sample-prompts/imported/training-config.txt',
      content: body.content,
      message: 'saved',
    };
  }
  return {
    ok: true,
    file: 'configs/sample-prompts/imported/original.txt',
    content: initialContent,
  };
};

let pathInputEvents = 0;
const pathInput = root.querySelector('[data-sample-prompts-path]');
pathInput.addEventListener('input', () => { pathInputEvents += 1; });
const dispose = mod.bindSamplePromptsDialog(root, {
  trainingContext: { configFile: 'configs/imported/training-config.toml' },
  apiClient: api,
});

root.querySelector('button[data-sample-prompts-open]').click();
await new Promise((resolve) => setTimeout(resolve, 0));

const portaledToBody = dialog.parentElement === document.body;
const loadedRows = dialog.querySelectorAll('[data-sample-prompt-row]').length;
const saveInitiallyEnabled = !dialog.querySelector('[data-sample-prompts-action="save"]').disabled;
const uniforms = Object.fromEntries([...dialog.querySelectorAll('[data-sample-prompts-uniform]')]
  .map((input) => [input.dataset.samplePromptsUniform, input]));
uniforms.width.value = '1024';
uniforms.height.value = '1024';
uniforms.steps.value = '28';
uniforms.cfg.value = '4';
dialog.querySelector('[data-sample-prompts-action="apply-uniform"]').click();

dialog.querySelector('[data-sample-prompts-action="add"]').click();
const promptInputs = dialog.querySelectorAll('[data-sample-prompt-field="prompt"]');
promptInputs[promptInputs.length - 1].value = 'style three';
promptInputs[promptInputs.length - 1].dispatchEvent(new Event('input', { bubbles: true }));

dialog.querySelector('[data-sample-prompts-action="save"]').click();
await new Promise((resolve) => setTimeout(resolve, 0));
const put = calls.find((call) => call.options.method === 'PUT');
const saved = JSON.parse(put.options.body);
const widths = [...dialog.querySelectorAll('[data-sample-prompt-field="width"]')].map((input) => input.value);
const count = dialog.querySelector('[data-sample-prompts-count]').textContent;
dispose();

console.log(JSON.stringify({
  loadedRows,
  saveInitiallyEnabled,
  widths,
  count,
  getUrl: calls[0].url,
  saved,
  path: pathInput.value,
  pathInputEvents,
  disposedOpen: dialog.hasAttribute('open'),
  portaledToBody,
  disposedConnected: dialog.isConnected,
}));
"""
    payload = _run_node(script)

    assert payload["loadedRows"] == 2
    assert payload["saveInitiallyEnabled"] is False
    assert payload["widths"] == ["1024", "1024", "1024"]
    assert payload["count"] == "3 条"
    assert "file=configs%2Fsample-prompts%2Fimported%2Foriginal.txt" in payload["getUrl"]
    assert payload["saved"]["train_config_file"] == "configs/imported/training-config.toml"
    assert payload["saved"]["content"].count("--w 1024 --h 1024 --s 28 --g 4") == 3
    assert "--custom keep" in payload["saved"]["content"]
    assert payload["path"] == "configs/sample-prompts/imported/training-config.txt"
    assert payload["pathInputEvents"] == 1
    assert payload["disposedOpen"] is False
    assert payload["portaledToBody"] is True
    assert payload["disposedConnected"] is False


@pytest.mark.integration
def test_sample_prompts_dialog_ignores_a_stale_load_after_reopen() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for sample prompt dialog race checks")
    jsdom = REPO_ROOT / "web/frontend-next/node_modules/jsdom/lib/api.js"
    if not jsdom.exists():
        pytest.skip("jsdom is required for sample prompt dialog race checks")

    script = r"""
import { JSDOM } from './web/frontend-next/node_modules/jsdom/lib/api.js';

const dom = new JSDOM('<!doctype html><html><body data-dragon-ui><main id="root"></main></body></html>', {
  pretendToBeVisual: true,
  url: 'http://localhost/',
});
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.Event = dom.window.Event;

const mod = await import('./web/static/js/dragon-ui/pages/sample-prompts-dialog.js?sample-prompts-race-test');
const root = document.querySelector('#root');
root.innerHTML = mod.renderSamplePromptsFieldControl({
  fieldId: 'sample-prompts', name: 'sample_prompts', value: 'configs/sample-prompts/race.txt',
}) + mod.renderSamplePromptsDialog();
const dialog = root.querySelector('[data-sample-prompts-dialog]');
dialog.showModal = function showModal() { this.setAttribute('open', ''); };
dialog.close = function close() { this.removeAttribute('open'); };

const pending = [];
const api = () => new Promise((resolve, reject) => pending.push({ resolve, reject }));
const dispose = mod.bindSamplePromptsDialog(root, { apiClient: api });
const open = root.querySelector('button[data-sample-prompts-open]');
open.click();
await new Promise((resolve) => setTimeout(resolve, 0));
dialog.querySelector('[data-sample-prompts-action="close"]').click();
await new Promise((resolve) => setTimeout(resolve, 0));
open.click();
await new Promise((resolve) => setTimeout(resolve, 0));

pending[1].resolve({ ok: true, file: 'configs/sample-prompts/new.txt', content: 'new style --w 1024' });
await new Promise((resolve) => setTimeout(resolve, 0));
pending[0].resolve({ ok: true, file: 'configs/sample-prompts/old.txt', content: 'old style --w 512' });
await new Promise((resolve) => setTimeout(resolve, 0));

const prompt = dialog.querySelector('[data-sample-prompt-field="prompt"]').value;
const file = dialog.querySelector('[data-sample-prompts-file]').textContent;
const status = dialog.querySelector('[data-sample-prompts-status]').textContent;
dispose();
pending.forEach(({ reject }) => reject(new Error('late failure')));
await new Promise((resolve) => setTimeout(resolve, 0));
console.log(JSON.stringify({ prompt, file, status }));
"""
    payload = _run_node(script)

    assert payload == {
        "prompt": "new style",
        "file": "configs/sample-prompts/new.txt",
        "status": "已加载",
    }
