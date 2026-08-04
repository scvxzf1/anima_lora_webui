# Split from test_training_frontend_state.py (dom)

from __future__ import annotations

from tests.frontend_test_support import *  # noqa: F403


import tests.frontend_test_support as _frontend_support
for _k, _v in vars(_frontend_support).items():
    if not _k.startswith("__"):
        globals()[_k] = _v

def test_setup_event_dom_contract_matches_index_html() -> None:
    source = _frontend_feature_text("js/features/app-shell/event-listeners.js", "js/features/app-shell/event-listeners-contract.js", "js/features/app-shell/event-listeners-setup.js", "js/features/app-shell/beginner-tooltips.js")
    dom_source = _frontend_module_text("js/shared/dom.js")
    html = INDEX_HTML.read_text(encoding="utf-8")
    listener_section = _section(source, "function setupEventListeners", "function installBeginnerTooltips")
    contract = _setup_event_dom_contract()

    assert contract["required"] == {
        "method-select",
        "variant-select",
        "preset-select",
        "btn-load-config",
        "btn-start-from-config",
        "btn-queue-from-config",
        "btn-save-toml",
        "toml-file-select",
        "toml-editor",
    }
    assert contract["optional"]
    missing = [
        dom_id
        for dom_id in sorted(contract["required"] | contract["optional"])
        if f'id="{dom_id}"' not in html
    ]
    assert not missing
    assert "export const SETUP_EVENT_DOM_CONTRACT = Object.freeze({" in source
    assert "globalThis.SETUP_EVENT_DOM_CONTRACT" not in source
    assert "export function setupEventListeners()" in source
    assert "export function installBeginnerTooltips()" in source
    assert "ctx.dom.bindEvent(id, eventName, handler" in listener_section
    assert "[webui-dom-contract] missing required DOM node" in dom_source
    assert not re.search(r"document\.getElementById\([^\n]+?\)\??\.addEventListener", listener_section)


def test_config_training_source_dom_contract_matches_index_html() -> None:
    source = _frontend_module_text("js/features/training-source/index.js")
    html = INDEX_HTML.read_text(encoding="utf-8")
    contract = _config_training_source_dom_contract()

    assert contract["required"] == {
        "continue-training-source",
        "continue-training-source-summary",
        "config-full-resume-panel",
        "config-full-resume-task-select",
        "config-full-resume-checkpoint-select",
        "config-full-resume-summary",
        "config-weight-hotstart-panel",
        "config-weight-hotstart-detail",
        "config-training-source-status",
    }
    missing = [
        dom_id
        for dom_id in sorted(contract["required"] | contract["optional"])
        if f'id="{dom_id}"' not in html
    ]
    assert not missing
    assert "export const CONFIG_TRAINING_SOURCE_DOM_CONTRACT = Object.freeze({" in source
    assert "globalThis.CONFIG_TRAINING_SOURCE_DOM_CONTRACT" not in source


def test_app_shell_theme_toggle_contract_matches_index_html() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    source = _frontend_module_text("js/features/app-shell/theme.js")

    toggle_match = re.search(r'<button id="theme-toggle"(?P<attrs>[^>]*)>', html)
    assert toggle_match is not None
    toggle_attrs = toggle_match.group("attrs")
    assert 'type="button"' in toggle_attrs
    assert 'aria-pressed="false"' in toggle_attrs
    assert 'id="theme-toggle-text"' in html

    assert "root.dataset.theme = safeTheme;" in source
    assert "document.getElementById('theme-toggle')" in source
    assert "document.getElementById('theme-toggle-text')" in source
    assert "toggle.setAttribute('aria-pressed', String(isLight));" in source
    assert "toggle.title = isLight ? '切换到深色主题' : '切换到浅色主题';" in source
    assert "label.textContent = safeTheme === 'light' ? '深色主题' : '浅色主题';" in source
    assert "toggle.addEventListener('click', () => {" in source
    assert "storage.getItem(storageKey)" in source
    assert "storage.setItem(storageKey, theme)" in source
    assert "getLossChart?.()?.setTheme?.(chartTheme());" in source


def test_app_shell_gpu_picker_contract_matches_index_html() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    source = _frontend_module_text("js/features/app-shell/gpu-picker.js")
    css = STYLE_CSS.read_text(encoding="utf-8")

    toggle_match = re.search(r'<button id="gpu-picker-toggle"(?P<attrs>[^>]*)>', html)
    assert toggle_match is not None
    toggle_attrs = toggle_match.group("attrs")
    assert 'type="button"' in toggle_attrs
    assert 'aria-expanded="false"' in toggle_attrs
    for dom_id in (
        "gpu-picker",
        "gpu-picker-panel",
        "gpu-all-checkbox",
        "gpu-option-list",
        "gpu-picker-note",
    ):
        assert f'id="{dom_id}"' in html
        assert f"document.getElementById('{dom_id}')" in source

    assert "api('/api/training/gpus')" in source
    assert "location.protocol === 'file:'" in source
    assert "selectedGpuPayload" in source
    assert "toggle.setAttribute('aria-expanded', String(nextOpen));" in source
    assert "toggle.setAttribute('aria-expanded', 'false');" in source
    assert ".gpu-picker-toggle[aria-expanded=\"true\"]" in css


def test_gpu_picker_all_selection_expands_available_gpu_indices() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for GPU picker payload checks")
    script = r"""
const { gpuPayloadForSelection } = await import('./web/static/js/features/app-shell/gpu-picker.js');
const payload = {
    all: gpuPayloadForSelection([], [{ index: 1 }, { index: 0 }]),
    explicit: gpuPayloadForSelection([1], [{ index: 0 }, { index: 1 }]),
    unavailable: gpuPayloadForSelection([], []),
};
console.log(JSON.stringify(payload));
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
        "all": [0, 1],
        "explicit": [1],
        "unavailable": [],
    }


def test_top_level_tab_buttons_have_matching_content_sections() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    source = _frontend_module_text("js/features/app-shell/tabs.js")
    button_tabs = re.findall(r'<button class="tab-btn[^"]*" data-tab="([^"]+)"', html)

    assert button_tabs == [
        "config",
        "datasets",
        "training",
        "weight-analysis",
        "settings",
        "environment",
        "image-test",
    ]
    assert "preview" not in button_tabs
    for tab_name in button_tabs:
        assert f'id="tab-{tab_name}"' in html

    assert 'document.querySelector(\'[data-tab="training"]\')' in source
    assert 'document.querySelector(\'[data-tab="config"]\')' in source
    assert "activeName !== 'preview'" in source
    assert "document.getElementById('tab-preview')?.classList.remove('active');" in source


def test_critical_workflow_dom_ids_are_present_in_index_html() -> None:
    """Critical save/queue/start/history/preview/settings anchors must stay in index.html."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    critical_ids = CRITICAL_WORKFLOW_DOM_IDS

    assert isinstance(critical_ids, frozenset)
    assert critical_ids == frozenset(
        {
            # config workflow
            "btn-load-config",
            "btn-save-toml",
            "btn-queue-from-config",
            "btn-start-from-config",
            "config-form",
            # training control
            "btn-stop-training",
            "status-indicator",
            "status-text",
            # global settings
            "btn-save-global-settings",
            "tab-settings",
            # history / preview
            "btn-refresh-history",
            "btn-preview-training-results",
            "task-history-list",
        }
    )

    missing = missing_dom_ids_in_html(html, critical_ids)
    assert not missing, f"critical workflow DOM ids missing from index.html: {sorted(missing)}"


def test_optional_node_syntax_smoke_for_shared_dom_helper() -> None:
    """Minimal node harness smoke: syntax-check a pure shared DOM helper module."""
    if not shutil.which("node"):
        pytest.skip("node is optional for DOM contract smoke")
    result = node_syntax_check("js/shared/dom.js")
    assert result.returncode == 0, result.stderr or result.stdout


def test_workflow_dom_contracts_match_index_html() -> None:
    """Queue/history/preview/settings required ids stay in index.html (no rename explosion)."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    expected_buckets = {"queue", "history", "preview", "settings"}
    assert set(WORKFLOW_DOM_CONTRACTS) == expected_buckets

    for name in sorted(expected_buckets):
        contract = workflow_dom_contract(name)
        assert "required" in contract and "optional" in contract
        assert contract["required"], f"{name} required set must not be empty"
        # optional may be empty, but keep disjoint when both present
        overlap = contract["required"] & contract["optional"]
        assert not overlap, f"{name} required/optional overlap: {sorted(overlap)}"
        missing_required = missing_dom_ids_in_html(html, contract["required"])
        assert not missing_required, f"{name} required missing: {sorted(missing_required)}"
        missing_optional = missing_dom_ids_in_html(html, contract["optional"])
        assert not missing_optional, f"{name} optional missing: {sorted(missing_optional)}"


