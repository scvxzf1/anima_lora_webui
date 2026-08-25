from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from tests.frontend_test_support import INDEX_HTML, REPO_ROOT, STATIC_DIR


def _read(relative: str) -> str:
    return (STATIC_DIR / relative).read_text(encoding="utf-8")


def test_global_model_config_page_has_master_detail_management_contract() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    css = _read("css/43-model-configs.css")
    page = _read("js/features/model-configs/page.js")

    assert html.index('data-tab="settings"') < html.index('data-tab="model-config"') < html.index('data-tab="environment"')
    section = html[html.index('id="tab-model-config"'):html.index('id="tab-environment"')]
    assert section.index('class="model-config-editor"') < section.index('class="model-config-sidebar"')
    for dom_id in (
        "model-config-editor-form",
        "model-config-list",
        "model-config-search",
        "btn-model-config-create",
        "btn-model-config-manage",
        "btn-model-config-save",
        "btn-model-config-set-default",
    ):
        assert f'id="{dom_id}"' in section
    assert "grid-template-columns: minmax(0, 1fr) minmax(300px, 360px);" in css
    assert "row.addEventListener('dragstart'" in page
    assert "row.addEventListener('drop'" in page
    assert "model-config-drag-handle" in page
    assert "moveModelConfigByOffset" in page
    assert "deleteModelConfig" in page


def test_model_config_picker_replaces_global_path_confirmation() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    css = _read("css/43-model-configs.css")
    resource = _read("js/features/config-form/resource-values.js")
    family_defaults = _read("js/features/config-form/model-family-defaults.js")
    picker = _read("js/features/model-configs/picker-dialog.js")

    assert 'id="global-model-config-picker-dialog"' in html
    assert "openModelConfigPickerDialog" in resource
    assert "showAppConfirmDialog" not in resource
    for key in ("model_family", "pretrained_model_name_or_path", "qwen3", "vae"):
        assert f"['{key}', selected." in resource
    assert "if (!input && key === 'model_family') {" in resource
    assert "configFormState.draftValues.set(key, next);" in resource
    assert (
        "familyInput || { dataset: { key: 'model_family' }, value: selected.model_family }"
        in resource
    )
    assert "modelFamilyFormDefaults(selected.model_family)" in resource
    assert "setFieldInputValue(key, value)" in resource
    for contract in (
        "attn_mode: 'torch'",
        "torch_compile: false",
        "discrete_flow_shift: 6.0",
        "timestep_sampling: 'uniform'",
        "caption_dropout_rate: 0.0",
    ):
        assert contract in family_defaults
    assert "fetchModelConfigLibrary" in picker
    assert "应用此模型配置" in picker
    assert "btn-model-config-picker-manage" in html
    for class_name in (
        "model-config-picker-toolbar-summary",
        "model-config-picker-option-head",
        "model-config-picker-detail-row",
        "model-config-picker-selection-status",
    ):
        assert class_name in picker
        assert f".{class_name}" in css
    picker_css = css[css.index(".model-config-picker-dialog {"):]
    assert "--model-panel-bg: #161f22;" in picker_css
    assert ':root[data-theme="light"] .model-config-picker-dialog' in picker_css
    assert "box-shadow: inset 3px 0 0 var(--model-accent);" in picker_css


def test_model_config_frontend_api_and_tab_loading_are_wired() -> None:
    api = _read("js/features/model-configs/api.js")
    tabs = _read("js/features/app-shell/tabs.js")
    startup = _read("js/features/app-shell/startup.js")
    listeners = _read("js/features/app-shell/event-listeners-setup.js")

    assert "api('/api/settings/model-configs')" in api
    assert "method: 'PUT'" in api
    assert "loadModelConfigsPage?.();" in tabs
    assert "loadModelConfigsPage," in startup
    assert "bindModelConfigEvents();" in listeners
    assert "bindModelConfigPickerEvents();" in listeners


def test_model_config_reorder_helpers_with_node() -> None:
    if not shutil.which("node"):
        pytest.skip("node is optional for model config helper checks")
    script = r"""
const mod = await import('./web/static/js/features/model-configs/model-config-data.js');
const items = [
  {id: 'a', name: 'A', model_family: 'anima', pretrained_model_name_or_path: 'a', qwen3: 'q', vae: 'v'},
  {id: 'b', name: 'B', model_family: 'krea2_raw', pretrained_model_name_or_path: 'b', qwen3: 'q', vae: 'v'},
  {id: 'c', name: 'C', model_family: 'anima', pretrained_model_name_or_path: 'c', qwen3: 'q', vae: 'v'},
];
console.log(JSON.stringify({
  drop: mod.moveModelConfig(items, 'c', 'a', 'before').map((item) => item.id),
  offset: mod.moveModelConfigByOffset(items, 'a', 1).map((item) => item.id),
  valid: mod.modelConfigValidationError(items[0], items),
  invalid: mod.modelConfigValidationError({...items[0], vae: ''}, items),
}));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "drop": ["c", "a", "b"],
        "offset": ["b", "a", "c"],
        "valid": "",
        "invalid": "请填写VAE 模型",
    }
