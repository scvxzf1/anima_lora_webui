# Split from test_training_frontend_state.py (misc)

from __future__ import annotations

from tests.frontend_test_support import *  # noqa: F403


import tests.frontend_test_support as _frontend_support
for _k, _v in vars(_frontend_support).items():
    if not _k.startswith("__"):
        globals()[_k] = _v

def test_config_toolbar_is_first_visible_config_row() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    css = STYLE_CSS.read_text(encoding="utf-8")

    editor_html = _section(html, '<section class="config-preset-editor">', '<section id="continue-training-source"')
    toolbar_html = _section(html, '<div class="config-toolbar', '<div id="gpu-picker" class="gpu-picker">')
    form_workspace_css = _section(css, "#tab-config .config-form-workspace,", "#tab-config .config-toolbar {")
    toolbar_css = _section(css, "#tab-config .config-toolbar {", "#tab-config .config-toolbar label")
    assert 'class="config-toolbar ui-toolbar"' in html or 'config-toolbar ui-toolbar' in toolbar_html

    assert "config-preset-header" not in editor_html
    assert "CONFIG FORGE" not in editor_html
    assert "CONFIG PRESET" not in editor_html
    assert "配置工作台" not in editor_html
    assert "训练配置" not in editor_html
    assert "config-selection-state\" hidden" in toolbar_html
    hidden_selection_html = _section(toolbar_html, '<div class="config-selection-state" hidden>', '</div>')
    assert "method-select" in toolbar_html
    assert "variant-select" in toolbar_html
    assert "preset-select" in toolbar_html
    assert "preset-select" not in hidden_selection_html
    assert "config-run-preset" in toolbar_html
    assert "运行覆盖" in toolbar_html
    assert "来自 configs/presets.toml" in toolbar_html
    assert ".config-run-preset" in css
    assert 'id="choice-guide"' not in html
    assert "padding: 0.55rem 1.2rem 1.15rem;" in form_workspace_css
    assert "align-items: center;" in toolbar_css


def test_config_actions_are_de_noised_and_sticky_controls_are_wired() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    html = INDEX_HTML.read_text(encoding="utf-8")
    css = STYLE_CSS.read_text(encoding="utf-8")

    listener_section = _section(source, "function setupEventListeners", "function installBeginnerTooltips")

    assert "toml-more-actions" in html
    assert "toml-more-actions-popover" in html
    assert "toml-secondary-actions" not in html
    assert "btn-apply-toml" in html
    assert "btn-toggle-toml-editor" in html
    assert ".toml-more-actions-popover" in css
    toml_current_row_css = _section(css, "#tab-config .toml-file-row {", "#tab-config .toml-current-file {")
    toml_current_file_css = _section(css, "#tab-config .toml-current-file {", "#tab-config .toml-current-file span")
    assert "overflow: visible;" in toml_current_row_css
    assert "z-index: 65;" in toml_current_row_css
    assert "padding: 0 1rem 0.56rem calc(1rem + 4px);" in toml_current_row_css
    assert "position: relative;" in toml_current_file_css
    assert "overflow: visible;" in toml_current_file_css
    assert "box-sizing: border-box;" in toml_current_file_css
    assert "min-width: 0;" in toml_current_file_css
    assert "padding: 0.42rem 0.54rem 0.42rem 0.68rem;" in toml_current_file_css
    assert "box-shadow: inset 3px 0 0 var(--config-accent);" in toml_current_file_css
    assert "grid-template-columns: 1.42rem minmax(0, 1fr);" in css
    assert "width: 1.32rem;" in _section(css, "#tab-config .file-group-drag-handle", "#tab-config .toml-file-group summary")
    assert "min-height: 42px;" in _section(css, "#tab-config .toml-file-item", "#tab-config .toml-file-item:hover")
    toml_actions_css = _section(css, "#tab-config .toml-file-actions", "#tab-config .toml-primary-actions,")
    toml_more_css = _section(css, "#tab-config .toml-more-actions {", "#tab-config .toml-more-actions > summary")
    toml_more_open_css = _section(css, "#tab-config .toml-more-actions[open]", "#tab-config .toml-more-actions > summary")
    toml_popover_css = _section(css, "#tab-config .toml-more-actions-popover", "#tab-config #toml-status")
    assert "overflow: visible;" in toml_actions_css
    assert "z-index: 70;" in toml_actions_css
    assert "position: relative;" in toml_more_css
    assert "z-index: 220;" in toml_more_open_css
    assert "z-index: 220;" in toml_popover_css
    assert "opacity: 0;" in _section(css, ".toml-group-actions", ".toml-group-action-btn,")

    assert "config-sticky-actions" in html
    assert "配置目录" in html
    assert "data-sticky-config-category=\"required\"" in html
    assert "data-sticky-config-category=\"common\"" in html
    assert "data-sticky-config-category=\"preview\"" in html
    assert "data-sticky-config-category=\"optimization\"" in html
    assert "data-sticky-config-category=\"advanced\"" in html
    assert "btn-sticky-config-optimization" in html
    assert "优化</strong>" in html
    assert "btn-sticky-config-advanced" in html
    assert "高级</strong>" in html
    assert "btn-sticky-save-config" not in html
    assert "btn-sticky-start-from-config" not in html
    assert "btn-sticky-queue-from-config" not in html
    assert "data-sticky-config-category" in listener_section
    assert "selectConfigCategory(btn.dataset.stickyConfigCategory, { scrollToForm: true })" in listener_section
    assert "function updateConfigStickyDirectory" in source
    assert "STICKY_CONFIG_CATEGORY_IDS" in source
    assert "new Set(['required', 'common', 'preview', 'optimization', 'advanced'])" in source
    assert "ADVANCED_CATEGORY_DEFAULT_OPEN_GROUPS" in source
    assert "new Set(['缓存与预处理'])" in source
    assert "if (category?.advanced) {" in source
    assert "configFormState.showAdvanced = true;" in source
    assert "(visibleCategories.has(categoryId) || (category.advanced && hasFields))" in source
    assert "configFormState.activeCategory === 'advanced' && ADVANCED_CATEGORY_DEFAULT_OPEN_GROUPS.has(name)" in source
    assert "function scrollConfigFormContentToTop" in source
    assert "scroller.scrollTo({ top: 0, behavior });" in source
    assert "scrollIntoView" not in _section(source, "function selectConfigCategory", "function scrollConfigFormContentToTop")
    assert "function updateConfigStickyPlacement" in source
    assert "--config-sticky-left" in source
    assert "--config-sticky-width" in source
    assert "--config-sticky-safe-space" in source
    assert "--config-left-max-height" in source
    assert "stickyActions.hidden = nextMode !== 'project'" in source
    assert ".config-sticky-actions" in css
    assert ".config-sticky-tab" in css
    sticky_css = _section(css, "#tab-config .config-sticky-actions", "#tab-config .config-sticky-title")
    config_left_css = _section(css, "#tab-config .config-left", "#tab-config .config-direct-editor")
    assert "width: var(--config-sticky-width, min(1040px, calc(100vw - 2rem)));" in sticky_css
    assert "grid-template-columns: auto repeat(5, minmax(88px, 1fr));" in sticky_css
    assert "left: var(--config-sticky-left, 1rem);" in sticky_css
    assert "position: fixed;" in sticky_css
    assert "position: sticky;" not in sticky_css
    assert "height: var(--config-left-max-height);" in config_left_css
    assert "max-height: var(--config-left-max-height);" in config_left_css
    assert "min-height: 180px;" in config_left_css
    assert "box-sizing: border-box;" in config_left_css
    assert "overflow-y: auto;" in config_left_css
    assert "overscroll-behavior: contain;" in config_left_css
    assert "padding-bottom: var(--config-sticky-safe-space);" in config_left_css
    assert "min-height: calc(var(--control-height) + 0.1rem);" in _section(css, "#tab-config .config-sticky-tab", "#tab-config .config-sticky-tab:hover")


def test_config_catalog_exposes_automagic_and_constant_with_warmup_options() -> None:
    labels_options = _frontend_module_text("js/config/catalog/labels-options.js")
    field_help = _frontend_module_text("js/config/catalog/field-help-training.js")

    assert "lr_scheduler: ['constant', 'constant_with_warmup', 'cosine'" in labels_options
    assert "lr_warmup_steps: '预热步数'" in labels_options
    assert "optimizer_type: ['AdamW', 'CAME', 'Automagic'" in labels_options
    assert "Automagic 属于实验优化器" in field_help
    assert "constant_with_warmup 表示先线性热身再固定" in field_help
    assert "0.05 表示前 5% 的训练步数逐步升到目标学习率" in field_help


def test_config_workbench_manager_is_right_column() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    css = STYLE_CSS.read_text(encoding="utf-8")

    layout = _section(css, "#tab-config .config-forge-layout", "#tab-config .config-preset-manager,")
    manager = _section(css, "#tab-config .config-preset-manager {", "#tab-config .config-sidebar-project")
    editor = _section(css, "#tab-config .config-preset-editor {", "#tab-config .config-preset-header")
    compact = _section(css, "@media (max-width: 900px)", "@media (max-width: 520px)")
    phone = _section(css, "@media (max-width: 520px)", "@media (max-width: 640px)")

    assert "左侧训练配置工作台 + 右侧配置预设管理" in html
    assert "grid-template-columns: minmax(0, 1fr) clamp(240px, 20vw, 300px);" in layout
    assert "grid-column: 2;" in manager
    assert "isolation: isolate;" in manager
    assert "z-index: 40;" in manager
    assert "overflow: visible;" in manager
    assert "border-left: 1px solid var(--config-border);" in manager
    assert "grid-column: 1;" in editor
    assert "grid-template-columns: minmax(0, 1fr) minmax(220px, 260px);" in compact
    assert "grid-template-columns: 1fr;" in phone

