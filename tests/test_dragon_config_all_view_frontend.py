import json
import shutil
import subprocess
from pathlib import Path

import pytest


STATIC = Path(__file__).resolve().parents[1] / "web" / "static"


def _read(relative: str) -> str:
    return (STATIC / relative).read_text(encoding="utf-8")


def test_all_config_view_is_additive_and_has_a_stable_route() -> None:
    page = _read("js/dragon-ui/pages/config-page.js")
    view = _read("js/dragon-ui/pages/config-all-view.js")
    index = _read("js/dragon-ui/index.js")

    assert "ALL_CONFIG_VIEW_ID = 'all'" in view
    assert 'href="#config/training-config/all"' in view
    assert "#config/training-config/common" in view
    assert "resolveConfigView(entries, preferredConfigSubId(context.subId, category), category)" in page
    assert "subId: parts[2] || null" in index
    assert "configCategoryFromHash(acceptedHash) === configCategoryFromHash(nextHash)" in index


def test_all_config_view_deduplicates_fields_without_changing_source_order() -> None:
    view = _read("js/dragon-ui/pages/config-all-view.js")

    assert "const seen = new Set()" in view
    assert "if (seen.has(key)) return false" in view
    assert "seen.add(key)" in view
    assert "uniqueEntries.flatMap((entry) => entry.keys)" in view
    assert "grid-auto-flow: row;" in _read("css/dragon/04g-dragon-config-fields.css")


def test_all_config_view_builds_flat_block_metadata() -> None:
    page = _read("js/dragon-ui/pages/config-page.js")
    metadata = _read("js/dragon-ui/pages/config-block-metadata.js")
    catalog = _read("js/dragon-ui/pages/config-field-catalog.js")

    assert "buildConfigBlocks(" in page
    assert "availabilityContext," in page
    assert "cluster.blocks.map(renderBlock).join('')" in _read("js/dragon-ui/pages/config-all-view.js")
    assert "spanForField" in metadata
    assert "tagId: cluster.id" in metadata
    assert "chapterId: stage.id" in metadata
    assert "required: REQUIRED_KEYS.has(key)" in metadata
    assert "experimental," in metadata
    assert "defaultValue:" in metadata
    assert "SECTION_GROUPS" in metadata
    assert "CONFIG_FIELD_CATALOG" in catalog
    assert "sortConfigCatalogItems(rawBlocks)" in metadata


def test_all_config_view_uses_current_method_scope_and_preserves_drafts() -> None:
    page = _read("js/dragon-ui/pages/config-page.js")
    values = _read("js/dragon-ui/pages/config-values.js")

    assert "buildCategoryEntries(category, rawEntries, trainingContext, currentValues)" in page
    assert "visibleConfigKeys(entry.keys, trainingContext, values)" in page
    assert "function resetConfigFormState(state, originalValues, entries)" in page
    assert "function captureDraftValue(input, state)" in page
    assert "captureDraftValue(field, state)" in page
    assert "values: pageState.draftValues" in page
    assert "typeof originalValue === 'boolean'" in values
    assert "typeof originalValue === 'number' || input.type === 'number'" in values


def test_lokr_fused_backends_are_scoped_selectors_and_round_trip_network_args() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for LoKr config catalog checks")
    defaults = _read("js/config/catalog/defaults.js")
    layout = _read("js/config/catalog/form-layout.js")
    labels = _read("js/config/catalog/labels-options.js")
    field_map = _read("js/dragon-ui/pages/config-field-map.js")
    groups = _read("js/dragon-ui/pages/section-groups.js")
    advanced_help = _read("js/config/catalog/field-help-advanced.js")

    expected_defaults = {
        "lokr_grouped_delta_backend": "triton",
        "lokr_grouped_delta_backward_backend": "triton_grad_w1_w2_grad_x",
    }
    for key, expected_default in expected_defaults.items():
        assert f"'{key}'," in defaults
        assert f"key: '{key}', arg: '{key}', default: '{expected_default}', valueType: 'string'" in defaults
        assert key in layout
        assert key in field_map
        assert key in groups
        assert f"{key}: advancedHelp(" in advanced_help
    assert "lokr_grouped_delta_backend: ['eager', 'triton']" in labels
    for option in (
        "triton_grad_x",
        "triton_grad_w2_partial",
        "triton_grad_w2_grad_x",
        "triton_grad_w1_w2_grad_x",
    ):
        assert f"'{option}'," in labels

    values_uri = (STATIC / "js/dragon-ui/pages/config-values.js").resolve().as_uri()
    script = f"""
const mod = await import({json.dumps(values_uri + '?lokr-backend-round-trip')});
const original = {{
  lokr_grouped_delta_backend: 'triton',
  lokr_grouped_delta_backward_backend: 'triton_grad_x',
  network_args: [
    'lokr_factor_group_size=8',
    'lokr_grouped_delta_backend=eager',
    'unrelated_flag=keep',
  ],
}};
const displayed = {{
  forward: mod.displayConfigValue('lokr_grouped_delta_backend', original),
  backward: mod.displayConfigValue('lokr_grouped_delta_backward_backend', original),
}};
const patch = mod.prepareConfigPatch({{
  lokr_grouped_delta_backend: 'triton',
  lokr_grouped_delta_backward_backend: 'triton_grad_w1_w2_grad_x',
}}, original);
console.log(JSON.stringify({{ displayed, patch }}));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=STATIC.parents[1],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    payload = json.loads(result.stdout)
    assert payload["displayed"] == {"forward": "eager", "backward": "triton_grad_x"}
    assert payload["patch"] == {
        "network_args": [
            "lokr_factor_group_size=8",
            "lokr_grouped_delta_backend=triton",
            "unrelated_flag=keep",
            "lokr_grouped_delta_backward_backend=triton_grad_w1_w2_grad_x",
        ]
    }


def test_all_config_view_has_global_search_navigation_and_change_summary() -> None:
    page = _read("js/dragon-ui/pages/config-page.js")
    view = _read("js/dragon-ui/pages/config-all-view.js")

    for contract in (
        'aria-label="搜索全部适用参数"',
        'aria-label="配置阶段导航"',
        "data-config-tag-filter",
        "data-config-preset-toggle",
    ):
        assert contract in view
    assert "data-config-section-jump" not in view
    assert "data-config-sections" not in view
    assert "replaceConfigDirtyKeys(state, Object.keys(rawChanges))" in page
    assert "data-config-dirty-count" in page
    assert "data-config-changed-only" in page
    assert "matchesChanged" in _read("js/dragon-ui/pages/config-field-filter.js")


def test_all_config_restore_is_explicit_and_skips_unknown_defaults() -> None:
    page = _read("js/dragon-ui/pages/config-page.js")

    assert "Object.prototype.hasOwnProperty.call(FORM_UI_DEFAULTS, key)" in page
    assert "有界面默认值的 ${resetKeys.length} 个参数" in page
    assert "const confirmed = await confirmDragonDialog" in page
    assert "恢复后仍需点击“保存配置”才会生效。" in page
    assert "恢复全部默认" in page


def test_all_config_layout_bounds_fields_and_collapses_preset_library() -> None:
    css = _read("css/dragon/04c-dragon-config-all.css")
    fields = _read("css/dragon/04g-dragon-config-fields.css")
    routes = _read("js/dragon-ui/route-styles.js")
    assert "04c-dragon-config-all.css" in routes
    assert "04g-dragon-config-fields.css" in routes
    assert "grid-template-rows: auto minmax(0, 1fr) auto" in css
    assert "grid-template-columns: minmax(0, 1fr) 272px" in css
    assert '[data-preset-collapsed="true"]' in css
    assert "@media (max-width: 734px)" in css
    assert "repeat(2, minmax(0, 1fr))" in fields
    assert "grid-auto-flow: row;" in fields
    assert "row dense" not in fields
    assert "132px" not in fields


def test_all_config_persists_view_and_sidebar_preferences() -> None:
    page = _read("js/dragon-ui/pages/config-page.js")
    view = _read("js/dragon-ui/pages/config-all-view.js")
    preferences = _read("js/dragon-ui/pages/config-ui-preferences.js")

    assert "preferredConfigSubId(context.subId, category)" in page
    assert "persistPresetLibraryCollapsed(collapsed)" in page
    assert 'data-config-view-mode="grouped"' in view
    assert 'data-config-view-mode="all"' in view
    assert "anima_dragon_config_ui" in preferences
    assert "localStorage.setItem" in preferences
    assert "preferredConfigBilingual()" in page
    assert "persistConfigBilingual(state.bilingual)" in page
    assert "data-config-bilingual-toggle" in view
    assert "data-config-bilingual" in page
    assert "dragon-config-label-key" in page
    assert "dragon-config-label-key" in _read("js/dragon-ui/pages/config-training-data.js")
    assert "preferredConfigCapsuleMode" not in preferences
    assert "persistConfigCapsuleMode" not in preferences
    assert "data-config-capsule-mode" not in view
    assert "export function preferredConfigBilingual" in preferences
    assert "export function persistConfigBilingual" in preferences

    config_css = _read("css/dragon/04-dragon-config.css")
    all_css = _read("css/dragon/04c-dragon-config-all.css")
    assert '.dragon-config-label-key {\n    display: none;' in config_css
    assert '[data-config-bilingual="true"] .dragon-config-label-key' in config_css
    assert ".dragon-config-bilingual-toggle[data-active=\"true\"]" in all_css


def test_removed_capsule_mode_has_no_ui_or_preference_state() -> None:
    page = _read("js/dragon-ui/pages/config-page.js")
    view = _read("js/dragon-ui/pages/config-all-view.js")
    preferences = _read("js/dragon-ui/pages/config-ui-preferences.js")

    for source in (page, view, preferences):
        assert "ConfigCapsuleMode" not in source
        assert "config-capsule-mode" not in source
        assert "capsuleMode" not in source


def test_config_bilingual_preference_round_trips_without_overwriting_siblings() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for config preference checks")
    module_uri = (STATIC / "js/dragon-ui/pages/config-ui-preferences.js").resolve().as_uri()
    script = f"""
const storage = new Map([
  ['anima_dragon_config_ui', JSON.stringify({{ viewMode: 'all', presetCollapsed: true, capsuleMode: 'filter', bilingual: 'yes' }})],
]);
globalThis.localStorage = {{
  getItem(key) {{ return storage.has(key) ? storage.get(key) : null; }},
  setItem(key, value) {{ storage.set(key, String(value)); }},
}};
const mod = await import({json.dumps(module_uri + '?bilingual-test')});
const invalidFallback = mod.preferredConfigBilingual();
const customFallback = mod.preferredConfigBilingual(true);
mod.persistConfigBilingual(true);
const afterTrue = JSON.parse(storage.get('anima_dragon_config_ui'));
mod.persistConfigBilingual(false);
const afterFalse = JSON.parse(storage.get('anima_dragon_config_ui'));
console.log(JSON.stringify({{ invalidFallback, customFallback, afterTrue, afterFalse }}));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=STATIC.parents[1],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    payload = json.loads(result.stdout)

    assert payload["invalidFallback"] is False
    assert payload["customFallback"] is True
    assert payload["afterTrue"] == {
        "viewMode": "all",
        "presetCollapsed": True,
        "capsuleMode": "filter",
        "bilingual": True,
    }
    assert payload["afterFalse"]["bilingual"] is False
    assert payload["afterFalse"]["viewMode"] == "all"
    assert payload["afterFalse"]["presetCollapsed"] is True


def test_config_fields_show_dirty_state_and_support_single_field_undo() -> None:
    page = _read("js/dragon-ui/pages/config-page.js")
    css = _read("css/dragon/04-dragon-config.css")

    assert "data-config-field-key" in page
    assert "data-config-reset-field" in page
    dirty_state = _read("js/dragon-ui/pages/config-dirty-state.js")
    assert "const dirty = state.dirtyKeys.has(field.dataset.configFieldKey)" in dirty_state
    assert "if (reset && reset.hidden === dirty) reset.hidden = !dirty" in dirty_state
    assert "setConfigControlValue(input, displayConfigValue(key, state.baselineValues))" in page
    assert '.dragon-field[data-dirty="true"]::before' in css
    assert '.dragon-field[data-dirty="true"] .dragon-field-reset' in css


def test_all_config_search_shortcut_and_preflight_field_focus_are_wired() -> None:
    view = _read("js/dragon-ui/pages/config-all-view.js")
    controls = _read("js/dragon-ui/pages/training-controls.js")

    assert "event.ctrlKey || event.metaKey" in view
    assert "search?.focus()" in view
    assert "focusFirstPreflightError(root, preflight)" in controls
    assert "field?.scrollIntoView" in controls
    assert "control?.setAttribute('aria-invalid', 'true')" in controls


def test_filters_preserve_hidden_drafts_and_search_live_values() -> None:
    source = _read("js/dragon-ui/pages/config-field-filter.js")
    assert "field.hidden = !(participates && matchesQuery)" in source
    assert "state?.showAllCandidates || state?.showChangedOnly" in source
    assert "control?.value ?? control?.dataset.checked" in source
    assert "scrollConfigCanvasTo(fieldsRoot, target, dragonScrollBehavior())" in source


def test_search_hides_empty_stages_without_a_second_chapter_filter() -> None:
    source = _read("js/dragon-ui/pages/config-field-filter.js")
    assert "const matchesQuery = !query || currentText.includes(query)" in source
    assert "group.hidden = !hasVisibleField && Boolean(query || state?.showChangedOnly)" in source
    assert "matchesTag" not in source
    assert "filterMode" not in source


def test_sections_keep_related_fields_in_catalog_order() -> None:
    view = _read("js/dragon-ui/pages/config-all-view.js")
    metadata = _read("js/dragon-ui/pages/config-block-metadata.js")
    css = _read("css/dragon/04g-dragon-config-fields.css")
    assert "export function renderSectionDivider" in view
    assert 'id="section-${escapeHtml(chapter.id)}"' in view
    assert "data-config-section-divider" in view
    assert "CONFIG_STAGE_META" in metadata
    assert "chapters.flatMap((chapter) => chapter.blocks)" in metadata
    assert "chapter.clusters.map((cluster)" in view
    assert "data-config-cluster-group" in view
    assert "grid-auto-flow: row;" in css
    assert "grid-column: 1 / -1" in css


def test_stage_navigation_scrolls_and_tracks_desktop_and_mobile() -> None:
    source = _read("js/dragon-ui/pages/config-field-filter.js")
    scroll = _read("js/dragon-ui/pages/config-workspace-scroll.js")
    assert "button.setAttribute('aria-current', 'location')" in source
    assert "scrollConfigCanvasTo(fieldsRoot, target, dragonScrollBehavior())" in source
    assert "renderConfigDirtyState(state.dirtyBindings, state)" in source
    assert "export function scrollConfigCanvasTo" in scroll
    assert "window.requestAnimationFrame(syncRadar)" in scroll
    assert "if (active === activeRadar) return" in scroll
    for target in ("canvas", "window"):
        assert f"{target}.addEventListener('scroll', scheduleRadar" in scroll
        assert f"{target}.removeEventListener('scroll', scheduleRadar" in scroll


def test_workbench_has_desktop_inner_scroll_and_mobile_page_scroll() -> None:
    view = _read("js/dragon-ui/pages/config-all-view.js")
    css = _read("css/dragon/04c-dragon-config-all.css")
    assert 'class="dragon-config-all-footer"' in view
    assert "html:has(body[data-dragon-ui] .dragon-config-all-workspace)" in css
    assert ".dragon-config-all-detail > .dragon-config-block-grid" in css
    assert "overflow-y: auto" in css
    assert "scroll-padding-top: 16px" in css
    assert "scrollbar-gutter: stable" in css
    assert "position: sticky" in css
    assert "overflow: visible" in css


def test_variable_height_fields_do_not_clip_path_help() -> None:
    css = _read("css/dragon/04g-dragon-config-fields.css")
    assert "height: auto" in css
    assert "contain-intrinsic-size" not in css
    assert '.dragon-config-block[data-path-field="true"]:hover' in css
    assert ".dragon-config-path-tooltip" in css
    assert "overflow-wrap: anywhere" in css


def test_keyboard_focus_stays_visible_in_the_active_scroll_container() -> None:
    source = _read("js/dragon-ui/pages/config-workspace-scroll.js")
    assert "canvas.addEventListener('focusin', keepFocusVisible)" in source
    assert "canvas.removeEventListener('focusin', keepFocusVisible)" in source
    assert "event.target?.closest?.('.dragon-config-block')" in source
    assert "scrollConfigCanvasTo(canvas, block)" in source


def test_semantic_states_do_not_decorate_every_field() -> None:
    page = _read("js/dragon-ui/pages/config-page.js")
    css = _read("css/dragon/04g-dragon-config-fields.css")
    for marker in ("dragon-config-required-dot", "dragon-config-exp-badge", "dragon-config-path-tooltip"):
        assert marker in page
        assert marker in css
    assert '[data-dirty="true"]' in css
    assert '[data-preflight-error="true"]' in css
    assert '.dragon-config-block[data-config-tone="experimental"]' not in css


def test_section_metadata_retains_safe_accents_without_decorative_corners() -> None:
    view = _read("js/dragon-ui/pages/config-all-view.js")
    metadata = _read("js/dragon-ui/pages/config-block-metadata.js")
    css = _read("css/dragon/04g-dragon-config-fields.css")
    assert "accent: chapter.accent" in metadata
    assert "function sectionAccent(value)" in view
    assert "/^#[0-9a-f]{6}$/i.test(accent)" in view
    assert ".dragon-config-block::after" in css
    assert "border-top: 3px" not in css
    assert "border-left: 3px" not in css


def test_form_rows_use_one_surface_and_focus_from_empty_space() -> None:
    page = _read("js/dragon-ui/pages/config-page.js")
    css = _read("css/dragon/04g-dragon-config-fields.css")
    assert "wrapper.querySelectorAll('.dragon-config-block')" in page
    assert "control?.focus({ preventScroll: true })" in page
    assert ":focus-visible" in css
    assert "font-family: var(--dragon-font-mono)" in css
    assert "background: transparent" in css


def test_grouped_config_renders_complete_structured_field_help() -> None:
    page = _read("js/dragon-ui/pages/config-page.js")
    help_view = _read("js/dragon-ui/pages/config-field-help.js")
    css = _read("css/dragon/04-dragon-config.css")
    controls_css = _read("css/dragon/02a-dragon-controls.css")

    for heading in [
        '你现在的设置',
        '新手建议',
        '它是做什么的',
        '为什么通常这样设',
        '改了可能有什么好处',
        '同时会带来什么',
        '可能遇到什么问题',
        '补充说明',
    ]:
        assert heading in help_view
    assert "normalizeHelpItems(value)" in help_view
    assert "bindConfigFieldHelpDialog(wrapper, loadFieldHelpCatalog)" in page
    assert "FIELD_HELP_SUMMARY_ZH[key]" in page
    assert "import('../../config/catalog/field-help.js?v=" in page
    assert "import { FIELD_HELP_ZH }" not in page
    assert "const help = resolveConfigFieldHelp(key, label, helpCatalog)" in help_view
    assert "await resolveHelpCatalog(helpCatalogSource)" in help_view
    assert "这一项暂时没有单独的新手说明" in help_view
    assert "renderConfigHelpButton(key, label," in page
    assert "unavailableReason" in page
    assert "currentValue: controlValue" in page
    assert "FORM_UI_DEFAULTS[key]" in page
    assert "modelFamily: availabilityContext?.modelFamily" in page
    assert "data-help-current-value" in help_view
    assert "data-help-default-value" in help_view
    assert "data-help-model-family" in help_view
    assert "renderHelpContext({ ...context, unavailable: Boolean(unavailableReason) })" in help_view
    assert "resolveLiveHelpValue(button)" in help_view
    assert "control.dataset.checked" in help_view
    assert "helpButton.dataset.helpModelFamily = availabilityContext.modelFamily" in page
    assert 'aria-haspopup="dialog"' in help_view
    assert '<dialog class="dragon-config-help-dialog"' in help_view
    assert 'class="dragon-icon-button"' in help_view
    assert '.dragon-icon-button {' in controls_css
    assert 'border-radius: 50%' in controls_css
    assert "dialog.showModal()" in help_view
    assert "event.target === dialog" in help_view
    assert 'class="dragon-field-help"' not in help_view
    assert '.dragon-config-help-dialog::backdrop' in css
    assert '.dragon-config-help-dialog-body' in css
    assert 'grid-template-columns: repeat(2, minmax(0, 1fr))' in css
    assert 'data-help-open="true"' not in css


def test_inapplicable_fields_keep_disabled_reasons_and_disclosure_preserves_drafts() -> None:
    page = _read("js/dragon-ui/pages/config-page.js")
    help_view = _read("js/dragon-ui/pages/config-field-help.js")
    css = _read("css/dragon/04-dragon-config.css")

    assert "configFieldAvailability(key, availabilityContext)" in page
    assert 'data-config-availability="${unavailable ? \'unavailable\' : \'available\'}"' in page
    assert 'data-config-disabled="${unavailable}"' in page
    assert 'tabindex="${unavailable ? \'-1\' : \'0\'}"' in page
    assert 'aria-disabled="${unavailable}"' in page
    assert "${unavailable ? ' disabled' : ''}" in page
    assert "if (toggle.dataset.configDisabled === 'true') return" in page
    assert "control?.disabled || control?.dataset.configDisabled === 'true'" in page
    assert "delete helpButton.dataset.helpUnavailableReason" in page
    assert "configFieldDisclosure(key, availabilityContext)" in page
    assert "collectConfigDraftChanges(state)" in page
    assert "...(state?.baselineValues || {})" in page
    assert "...(state?.draftValues || {})" in page

    assert "data-help-unavailable-reason" in help_view
    assert "为什么现在不能改" in help_view
    assert "查看不可用原因" in help_view
    assert '[data-config-availability="unavailable"]' in css
    assert '.dragon-field-help-btn-unavailable' in css


def test_config_field_availability_tracks_adapter_and_compute_conditions() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for config availability checks")
    module_uri = (STATIC / "js/dragon-ui/pages/config-field-availability.js").resolve().as_uri()
    script = f"""
const mod = await import({json.dumps(module_uri + '?availability-state-test')});
const check = (key, context) => mod.configFieldAvailability(key, context);
console.log(JSON.stringify({{
  adapterFromDraft: mod.resolveConfigAdapterKind({{ lora_adapter_kind: 'lokr' }}),
  draftOverridesFlags: mod.resolveConfigAdapterKind({{ lora_adapter_kind: 'lora', use_lokr: true }}),
  adapterFromLegacyFlag: mod.resolveConfigAdapterKind({{ use_lokr: 'true' }}),
  falseStringIsFalse: mod.resolveConfigAdapterKind({{ use_lokr: 'false' }}),
  lokrDisabled: check('lokr_grouped_delta_backend', {{ method: 'lora', adapter: 'lora', baseCompute: 'bf16' }}),
  lokrEnabled: check('lokr_grouped_delta_backend', {{ method: 'lora', adapter: 'lokr', baseCompute: 'bf16' }}),
  convrotDisabled: check('convrot_group_size', {{ method: 'lora', adapter: 'lora', baseCompute: 'bf16' }}),
  convrotEnabled: check('convrot_group_size', {{ method: 'lora', adapter: 'lora', baseCompute: 'w8a16_convrot' }}),
}}));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=STATIC.parents[1],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    payload = json.loads(result.stdout)

    assert payload["adapterFromDraft"] == "lokr"
    assert payload["draftOverridesFlags"] == "lora"
    assert payload["adapterFromLegacyFlag"] == "lokr"
    assert payload["falseStringIsFalse"] == "lora"
    assert payload["lokrDisabled"]["enabled"] is False
    assert payload["lokrDisabled"]["code"] == "adapter-family"
    assert "LoKr" in payload["lokrDisabled"]["reason"]
    assert payload["lokrEnabled"] == {"enabled": True, "reason": "", "code": None}
    assert payload["convrotDisabled"]["enabled"] is False
    assert payload["convrotDisabled"]["code"] == "convrot-base-compute"
    assert payload["convrotEnabled"] == {"enabled": True, "reason": "", "code": None}


def test_config_help_summary_matches_full_catalog() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for config help catalog checks")
    script = """
import { FIELD_HELP_ZH } from './web/static/js/config/catalog/field-help.js';
import { FIELD_HELP_SUMMARY_ZH } from './web/static/js/config/catalog/field-help-summary.js';
const expected = Object.fromEntries(Object.entries(FIELD_HELP_ZH).map(([key, help]) => [key, String(help?.summary || help?.['作用'] || '').trim()]));
const keys = [...new Set([...Object.keys(expected), ...Object.keys(FIELD_HELP_SUMMARY_ZH)])];
const mismatches = keys.filter((key) => expected[key] !== FIELD_HELP_SUMMARY_ZH[key]);
console.log(JSON.stringify({ expected: Object.keys(expected).length, summaries: Object.keys(FIELD_HELP_SUMMARY_ZH).length, mismatches }));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=STATIC.parents[1],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    payload = json.loads(result.stdout)
    assert payload["expected"] == payload["summaries"]
    assert payload["mismatches"] == []


def test_advanced_config_fields_visible_in_flat_view_have_help_coverage() -> None:
    method_help = _read("js/config/catalog/field-help-method.js")
    training_help = _read("js/config/catalog/field-help-training.js")
    advanced_help = _read("js/config/catalog/field-help-advanced.js")
    field_help = _read("js/config/catalog/field-help.js")
    labels = _read("js/config/catalog/labels-options.js")

    method_keys = (
        "validation_baselines",
        "ip_pair_mode",
        "ip_pair_prob",
        "ip_pair_min_level",
        "ip_pair_caption_strip_p",
        "content_router_source",
        "content_router_init_std",
        "content_router_layer_norm",
    )
    training_keys = (
        "model_family",
        "compile_dynamic_seq",
        "compile_seq_bands",
        "activation_memory_budget",
        "train_adaln",
        "cpu_offload_checkpointing",
    )
    for key in method_keys:
        assert f"{key}: help(" in method_help
    for key in training_keys:
        assert f"{key}: help(" in training_help
    shipped_advanced_keys = (
        "path_pattern",
        "drop_lowres_images",
        "min_pixels",
        "use_cmmd",
        "use_chimera_hydra",
        "num_experts_content",
        "num_experts_freq",
        "balance_w_content",
        "balance_w_freq",
        "network_content_router_lr_scale",
        "network_freq_router_lr_scale",
        "freq_router_init_std",
        "freq_router_layer_norm",
        "lokr_grouped_delta_backend",
        "lokr_grouped_delta_backward_backend",
        "rank_dropout",
    )
    for key in shipped_advanced_keys:
        assert f"{key}: advancedHelp(" in advanced_help
    catalog_advanced_keys = (
        "apply_ffn_lora", "b_cond_init", "cache_fingerprint_mode", "channel_scaling_alpha",
        "cond_scale", "cond_token_count", "contrastive_every_n", "contrastive_jaccard_alpha",
        "contrastive_k", "contrastive_negative_mode", "contrastive_objective", "contrastive_tau",
        "contrastive_warmup_ratio", "contrastive_weight", "data_dir", "dit_path", "dual_bank",
        "encoder", "encoder_dim", "force_rebuild_preprocess_cache", "gate_lr", "init_std",
        "ip_diagnostics_epochs", "ip_scale", "iterations", "max_data_loader_n_workers", "n_layers",
        "n_t_buckets", "pe_lora_alpha", "pe_lora_enabled", "pe_lora_layer_from", "pe_lora_rank",
        "resampler_heads", "resampler_layers", "reuse_dataset_cache_copy",
        "reuse_text_encoder_cache", "reuse_vae_latents", "seed", "softrank_method",
        "softrank_softness", "splice_position", "timestep_mask_at_inference",
        "timestep_mask_mode", "weight_decay",
    )
    for key in catalog_advanced_keys:
        assert f"{key}: advancedHelp(" in advanced_help
    assert "...FIELD_HELP_ADVANCED_ZH" in field_help
    for key in ("model_family", "activation_memory_budget", "train_adaln", "cpu_offload_checkpointing"):
        assert f"{key}: '" in labels
    assert "lokr_grouped_delta_backend: 'LoKr 分组 Delta 后端'" in labels
    assert "lokr_grouped_delta_backward_backend: 'LoKr 分组 Delta 反向后端'" in labels
    assert "rank_dropout: '秩 Dropout'" in labels


def test_form_help_is_available_without_hover_and_toggles_are_not_cards() -> None:
    css = _read("css/dragon/04g-dragon-config-fields.css")
    assert ".dragon-config-block .dragon-field-help-btn" in css
    assert "pointer-events: auto" in css
    assert "opacity: 0.8" in css
    assert '.dragon-config-block:has(.dragon-toggle[data-checked="true"])' not in css


def test_global_chrome_preferences_remain_available_for_compatible_views() -> None:
    chrome = _read("js/dragon-ui/config-chrome.js")
    entry = _read("js/dragon-ui/index.js")
    assert "dragon_config_help_always_visible" in chrome
    assert "dragon_config_tags_always_visible" in chrome
    assert "applyDragonConfigChromeSettings(globalSettings || {})" in entry


def test_workbench_uses_neutral_surfaces_and_compact_headings() -> None:
    view = _read("js/dragon-ui/pages/config-all-view.js")
    css = _read("css/dragon/04c-dragon-config-all.css")
    fields = _read("css/dragon/04g-dragon-config-fields.css")
    assert "--config-surface: #fafafa" in css
    assert "--config-control: #fff" in css
    assert '[data-theme="dark"]' in css
    assert "font-size: 14px" in fields
    assert "font-size: 18px" in fields
    assert 'data-config-stage-count>(${visibleCount})</span>' in view


def test_path_tooltip_is_hover_only_and_does_not_intercept_editing() -> None:
    css = _read("css/dragon/04g-dragon-config-fields.css")
    assert '.dragon-config-block[data-path-field="true"]:hover .dragon-config-path-tooltip' in css.replace("\n  ", " ")
    assert '.dragon-config-block[data-path-field="true"]:focus-within .dragon-config-path-tooltip' not in css
    assert "pointer-events: none" in css
    assert "overflow-wrap: anywhere" in css
