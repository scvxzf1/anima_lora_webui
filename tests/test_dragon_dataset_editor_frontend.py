from __future__ import annotations

from tests.frontend_test_support import INDEX_HTML, STATIC_DIR


def _read(relative: str) -> str:
    return (STATIC_DIR / relative).read_text(encoding="utf-8")


def test_dragon_dataset_editor_has_complete_workspace_contract() -> None:
    page = _read("js/dragon-ui/pages/dataset-editor.js")
    fields = _read("js/dragon-ui/pages/dataset-editor-fields.js")
    presets = _read("js/dragon-ui/pages/dataset-editor-presets.js")
    preview = _read("js/dragon-ui/pages/dataset-editor-preview.js")

    for marker in (
        "data-preset-search",
        "data-preset-action=\"new\"",
        "data-preset-action=\"import\"",
        'data-workspace-action="save-as"',
        'data-workspace-action="apply"',
        'data-workspace-action="stage"',
        "data-dataset-suggest",
        "data-dataset-preview",
        "data-dataset-apply-defaults",
        "beforeLeave: () => shouldLeaveEditor(state)",
    ):
        assert marker in page or marker in presets or marker in fields
    assert "dataset-presets/images" in preview
    assert "至少保留 1 组普通训练数据" in fields
    assert "至少需要保留 1 个数据集组" in page
    assert "hydrateDatasetFieldA11y" in fields
    assert "aria-invalid" in fields
    assert "最大桶尺寸不能小于训练分辨率" in fields
    assert "启用触发词复制后必须填写触发词" in fields
    assert "分阶段调度暂不支持正则化数据集" in fields
    assert "validateDatasetEditor(root, { stageScheduleEnabled: state.stageScheduleEnabled })" in page
    assert "DATASET_SETTING_KEYS.filter" in page
    assert "当前训练配置正在使用这个预设" in page
    assert "await applyDatasetPreset(api, result.file, state.context.configFile)" in page
    assert '<main class="dragon-dataset-editor-panel">' not in page
    assert "数据与标注" not in page
    assert "管理可复用的数据集预设" not in page


def test_dragon_dataset_editor_uses_compact_page_header() -> None:
    css = _read("css/dragon/06-dragon-pages.css")
    hero_rule = css[css.index(".dragon-dataset-hero {"):css.index(".dragon-dataset-hero > div:first-child")]
    title_rule = css[css.index(".dragon-dataset-hero h1 {"):css.index(".dragon-dataset-hero-actions,", css.index(".dragon-dataset-hero h1 {"))]

    assert "align-items: center;" in hero_rule
    assert "gap: var(--dragon-sp-4);" in hero_rule
    assert "padding: var(--dragon-sp-3) 0;" in hero_rule
    assert "margin: 0;" in title_rule


def test_dragon_dataset_preset_group_management_contract() -> None:
    page = _read("js/dragon-ui/pages/dataset-editor.js")
    presets = _read("js/dragon-ui/pages/dataset-editor-presets.js")

    for marker in (
        'data-preset-action="new-group"',
        'data-preset-group-action="rename"',
        'data-preset-group-action="delete"',
    ):
        assert marker in presets

    for label in (
        "新建预设分组",
        "重命名分组",
        "只删除分组，不删除其中的 TOML",
    ):
        assert label in presets or label in page

    assert "data-preset-move-select" not in presets
    assert "data-preset-move-file" not in presets
    assert 'data-preset-drop-group="${escapeAttribute(group.id || \'\')}"' in presets
    assert 'data-preset-dropzone="${escapeAttribute(group.id || \'\')}"' in presets
    assert 'draggable="${draggable ? \'true\' : \'false\'}"' in presets
    assert 'data-preset-drag-source' in presets
    assert presets.count('draggable="${draggable ? \'true\' : \'false\'}"') == 1
    assert "placeDatasetPreset" in page
    assert "bindPresetDragAndDrop" in page
    assert "addEventListener('dragstart'" in page
    assert "addEventListener('dragover'" in page
    assert "addEventListener('drop'" in page
    assert "presetOrderForDrop" in page
    assert "placePresetAt" in page
    shared_drag = _read("js/dragon-ui/ordered-drag-target.js")
    assert "requestAnimationFrame" in shared_drag
    assert "targetKey: 'presetDropTarget'" in page
    assert "clearOrderedDropTargets" in page
    assert "groupNode.contains(event.relatedTarget)" in page
    assert "api('/api/config/file-groups/place'," in presets
    assert "target: 'file'" in presets
    assert "order" in presets

    assert "api('/api/config/file-groups'," in presets
    assert "method: 'POST'" in presets
    assert "kind: 'dataset'" in presets
    assert "api(`/api/config/file-groups/${encodeURIComponent(groupId)}`" in presets
    assert "method: 'PATCH'" in presets
    assert "method: 'DELETE'" in presets

    css = _read("css/dragon/06-dragon-pages.css")
    assert "border: 1px solid var(--dragon-border);" in css
    assert "grid-template-columns: minmax(0, 1fr) auto;" in css
    assert "padding: 8px 9px;" in css
    assert ".dragon-dataset-preset-group.dragon-dataset-preset-drop-target .dragon-dataset-preset-dropzone::before" in css
    assert "box-shadow: inset 0 0 0 2px var(--dragon-accent)" not in css
    assert ".dragon-dataset-preset-dropzone[data-over=\"true\"]" in css
    assert ".dragon-dataset-preset-drop-before::before" in css
    assert ".dragon-dataset-dragging .dragon-dataset-preset-dropzone" in css
    assert ".dragon-dataset-preset-drag-handle" in css
    assert "-webkit-user-drag: element;" in css
    assert 'class="dragon-dataset-preset-drag-handle" draggable="${draggable ? \'true\' : \'false\'}"' in presets
    assert "handle?.addEventListener('dragstart'" in page
    assert "handle?.addEventListener('click', (event) => event.stopPropagation())" in page
    assert "schedulePresetListAutoScroll(state, list, event.clientY)" in page
    assert "const edge = Math.min(84, rect.height * 0.22)" in page
    assert "list.scrollTop += delta" in page
    assert "state.presetAutoScrollFrame = window.requestAnimationFrame(tick)" in page
    assert "stopPresetListAutoScroll(state)" in page
    assert 'role="region" aria-label="数据集配置列表" tabindex="0"' in presets
    assert "grid-template-rows: auto auto auto auto minmax(0, 1fr);" in css
    assert "height: min(820px, calc(100vh - var(--dragon-nav-height) - 48px));" in css
    assert ".dragon-dataset-preset-list {" in css
    assert "overflow-y: scroll;" in css
    assert "scrollbar-width: auto;" in css
    assert ".dragon-dataset-preset-list::-webkit-scrollbar-thumb" in css
    assert ".dragon-dataset-preset-list::-webkit-scrollbar { width: 11px; }" in css
    assert "min-height: 44px;" in css
    assert ".dragon-dataset-preset-list::-webkit-scrollbar-thumb:active" in css
    assert "overflow-x: hidden;" in css
    assert "overflow-y: scroll;" in css
    assert ".dragon-dataset-preset-row { position: relative; box-sizing: border-box; width: 100%; max-width: 100%;" in css
    assert "grid-template-columns: 28px minmax(0, 1fr) fit-content(104px);" in css
    assert "padding: 0 7px 14px 0;" in css
    assert ".dragon-dataset-preset-item-meta { display: flex; min-width: 0; max-width: 104px;" in css
    assert '<div class="dragon-dataset-preset-row dragon-dataset-preset-item" role="button" tabindex="0"' in presets
    assert '<button class="dragon-dataset-preset-row dragon-dataset-preset-item"' not in presets
    assert "presetSuppressClickUntil = performance.now() + 300" in page
    assert "performance.now() < state.presetSuppressClickUntil" in page
    assert "['Enter', ' '].includes(event.key)" in page
    assert "event.preventDefault();" in page
    dropzone_rule = css[css.index(".dragon-dataset-preset-dropzone {"):css.index(".dragon-dataset-preset-dropzone[data-empty=")]
    assert "transition: max-height" not in dropzone_rule


def test_dragon_dataset_editor_uses_desktop_internal_scroll_and_single_column_page_flow() -> None:
    page = _read("js/dragon-ui/pages/dataset-editor.js")
    css = _read("css/dragon/06-dragon-pages.css")

    assert 'data-dataset-form' in page
    render_page = page[page.index("function renderPage(state)"):page.index("function renderDatasetSavebar(state)")]
    assert render_page.index('class="dragon-dataset-editor-panel"') < render_page.index("${renderDatasetSavebar(state)}") < render_page.index("${renderDatasetPresetLibrary(state)}")
    assert 'data-workspace-action="save"' in page
    assert "保存数据集预设" in page
    assert 'panel.innerHTML = `${renderEditorPanel(state)}${renderDatasetSavebar(state)}`;' in page
    assert "state.ui = createDatasetEditorBindings(root);" in page
    form_rule = css[css.index('.dragon-dataset-form {'):css.index('.dragon-dataset-section {')]
    assert 'max-height: none;' in form_rule
    assert 'min-height: 0;' in form_rule
    assert 'overflow: visible;' in form_rule
    assert 'scrollbar-gutter: auto;' in form_rule
    assert 'scroll-padding-block: 0;' in form_rule
    assert 'padding: 0;' in form_rule
    assert 'margin: 0;' in form_rule
    assert 'overflow-y: auto;' not in form_rule
    desktop_rule = css[css.index('@media (min-width: 1069px) {'):css.index('.visually-hidden', css.index('@media (min-width: 1069px) {'))]
    assert 'body[data-dragon-ui]:has(.dragon-dataset-page-host)' in desktop_rule
    assert 'grid-template-rows: auto auto minmax(0, 1fr);' in desktop_rule
    assert '.dragon-dataset-editor-panel {' in desktop_rule
    assert 'flex-direction: column;' in desktop_rule
    assert '.dragon-dataset-form {' in desktop_rule
    assert 'overflow-y: auto;' in desktop_rule
    assert 'overscroll-behavior: contain;' in desktop_rule
    assert 'scrollbar-gutter: stable;' in desktop_rule
    assert '.dragon-dataset-savebar {' in desktop_rule
    assert 'position: relative;' in desktop_rule
    assert '.dragon-dataset-library {' in desktop_rule
    assert 'box-sizing: border-box;' in desktop_rule
    assert 'height: 100%;' in desktop_rule
    assert 'grid-template-columns: repeat(2, minmax(0, 1fr));' in css
    assert 'grid-template-columns: repeat(var(--dataset-field-columns, 2), minmax(0, 1fr));' in css
    assert 'grid-auto-rows: minmax(64px, auto);' in css
    assert 'grid-template-rows: minmax(20px, auto) 36px auto;' in css
    assert '.dragon-dataset-row-primary { grid-template-columns: repeat(4, minmax(0, 1fr));' in css


def test_dragon_dataset_fields_use_adaptive_columns_with_compact_tooltips() -> None:
    fields = _read("js/dragon-ui/pages/dataset-editor-fields.js")
    css = _read("css/dragon/06-dragon-pages.css")

    assert "function settingsGroup(title, icon, fields)" in fields
    assert "const columns = count <= 3 ? count : (count === 4 ? 2 : 3);" in fields
    assert "--dataset-field-columns:${columns}" in fields
    assert "data-field-count=\"${count}\"" in fields
    assert "options.help ?" in fields
    assert "data-tooltip=" in fields
    assert "dragon-field-hint" not in fields
    assert "renderIcon(icon, 'dragon-dataset-settings-icon')" in fields
    assert "<header><h3>${escapeHtml(title)}</h3><p>" not in fields
    assert fields.index("settingsGroup('验证集'") < fields.index("settingsGroup('分桶规则'")
    assert ".dragon-dataset-settings-group > header p" not in css
    assert ".dragon-dataset-page .dragon-field-hint" not in css
    assert "grid-template-rows: minmax(28px, auto) 1fr;" in css
    assert "body[data-dragon-ui] #dragon-root .dragon-dataset-settings-group > header" in css


def test_dragon_dataset_editor_exposes_visual_status_and_path_feedback() -> None:
    page = _read("js/dragon-ui/pages/dataset-editor.js")
    fields = _read("js/dragon-ui/pages/dataset-editor-fields.js")
    paths = _read("js/dragon-ui/pages/dataset-editor-paths.js")
    presets = _read("js/dragon-ui/pages/dataset-editor-presets.js")
    css = _read("css/dragon/06-dragon-pages.css")

    assert 'class="dragon-status-badge" data-dataset-link-state' in page
    assert 'data-dataset-dirty-text' in page
    assert "有未保存更改" in page
    assert "已同步至配置" in page
    assert "已同步至预设" in page
    assert 'data-workspace-action="apply" title="把已保存的数据集预设关联到当前训练配置"' in page
    assert "renderIcon('save', 'dragon-btn-icon')" in page
    assert 'data-dataset-browse' in fields
    assert 'data-dataset-copy' in fields
    assert 'data-dataset-path-status' in fields
    assert "根据原始目录补全缓存路径" in fields
    assert "source_image_dir: path, inspect: '1'" in paths
    assert "source_image_count" in paths
    assert "source_inspection_error" in paths
    assert "目录存在，未检测到图片" in paths
    assert "window.showDirectoryPicker" in paths
    assert "navigator.clipboard?.writeText" in paths
    assert "refreshDatasetPathStatus(api, row)" in page
    assert "circleHelp" in fields
    assert "先验损失权重" in fields and "打乱标注" in fields
    assert "dragon-dataset-settings-icon" in fields
    assert "Reps" in presets
    assert "当前正在编辑" in presets
    assert '.dragon-status-badge[data-state="dirty"]' in css
    assert '.dragon-dataset-path-input-actions' in css
    assert '.dragon-dataset-path-status[data-state="empty"]' in css
    assert '.dragon-field-help::after' in css
    assert '.dragon-dataset-preset-row[data-active="true"]' in css
    assert '.dragon-dataset-preset-group-actions { opacity: 0;' in css


def test_dragon_dataset_drag_state_recovers_after_bottom_or_noop_drop() -> None:
    page = _read("js/dragon-ui/pages/dataset-editor.js")

    assert "function bindPresetDragRecovery(root, state)" in page
    assert "window.addEventListener('dragend', finish, true)" in page
    assert "window.addEventListener('drop', finishAfterDrop, true)" in page
    assert "window.addEventListener('blur', finish)" in page
    assert "event.key === 'Escape'" in page
    assert "root.querySelectorAll('.dragon-dataset-preset-dragging')" in page
    assert "async function placePresetAt(root, state, file, groupId, anchorFile, position) {\n    finishPresetDrag(root, state);" in page
    noop_guard = "if (sourceGroup?.id === groupId && order.length === currentOrder.length && order.every((path, index) => path === currentOrder[index])) return;"
    assert page.index("finishPresetDrag(root, state);", page.index("async function placePresetAt")) < page.index(noop_guard)
    assert "function disposePresetDragRecovery(root, state)" in page
    assert "if (state.root) disposePresetDragRecovery(state.root, state);" in page

def test_dragon_dataset_editor_bindings_and_stage_bridge_are_safe() -> None:
    page = _read("js/dragon-ui/pages/dataset-editor.js")
    stage_model = _read("js/features/config-form/stage-resolution-model.js")
    stage_dialog = _read("js/features/config-form/stage-resolution-ui-dialog.js")

    assert "button.dataset.dragonDatasetBound === 'true'" in page
    assert "stage-resolution-ui.js?v=module-bootstrap-20260901-dialog-v1" in page
    for bridge in (
        "app-context-bridge.js?v=module-bootstrap-20260831-release-v1",
        "config-state-bridge.js?v=module-bootstrap-20260831-release-v1",
        "dataset-state-bridge.js?v=module-bootstrap-20260831-release-v1",
        "runtime-bridge.js?v=module-bootstrap-20260831-release-v1",
        "toml-action-state-bridge.js?v=module-bootstrap-20260831-release-v1",
    ):
        assert bridge in page
    assert "let legacyStageDatasetState = null" in page
    assert "anima-stage-schedule-change" in page
    assert "anima-stage-schedule-change" in stage_dialog
    assert "document.querySelector('[data-dataset-page]')" in stage_model
    assert "getAppContext" not in stage_model
    assert "state.fatalError = '';" in page
    assert "state.libraryError = '';" in page
    save_body = page[page.index("async function saveCurrentPreset"):page.index("async function saveAsCurrentPreset")]
    save_as_body = page[page.index("async function saveAsCurrentPreset"):page.index("async function applyCurrentPreset")]
    apply_body = page[page.index("async function applyCurrentPreset"):page.index("async function reloadCurrentPreset")]
    assert "state.datasetConfig =" not in save_body
    assert "state.datasetConfig =" not in save_as_body
    assert "state.datasetConfig =" in apply_body


def test_dragon_router_supports_dirty_page_lifecycle() -> None:
    router = _read("js/dragon-ui/router.js")
    entry = _read("js/dragon-ui/index.js")
    nav = _read("js/dragon-ui/nav.js")

    assert "export function canLeaveCurrentPage" in router
    assert "currentPage?.onUnmount?.()" in router
    assert "beforeLeave, onUnmount" in router
    assert "history.replaceState" in entry
    assert "dragon-route-restored" in entry
    assert "dragon-route-restored" in nav
    assert "route.type === 'external'" in entry
    assert "let navigationSequence = 0" in router
    assert "sequence !== navigationSequence" in router
    assert "let routeChangeSequence = 0" in entry
    assert "sequence !== routeChangeSequence" in entry


def test_dragon_dataset_layout_avoids_transformed_fixed_savebar() -> None:
    page = _read("js/dragon-ui/pages/dataset-editor.js")
    css = _read("css/dragon/06-dragon-pages.css")

    assert page.index('<div class="dragon-dataset-editor-panel">') < page.index("${renderDatasetPresetLibrary(state)}")
    assert "--dragon-dataset-page-gutter: clamp(12px, 1.75vw, 40px);" in css
    assert "--dragon-dataset-library-width: clamp(288px, 20vw, 420px);" in css
    assert "--dragon-dataset-workspace-gap: clamp(12px, 1.4vw, 24px);" in css
    assert "box-sizing: border-box;" in css
    assert "width: 100%;" in css
    assert "max-width: none;" in css
    assert "grid-template-columns: minmax(0, 1fr) var(--dragon-dataset-library-width);" in css
    assert "right: var(--dragon-dataset-page-gutter);" in css
    assert "left: var(--dragon-dataset-page-gutter);" in css
    assert "@media (min-width: 2800px)" in css
    assert ".dragon-dataset-advanced-body { grid-template-columns: repeat(3, minmax(0, 1fr));" in css
    assert "@media (min-width: 1920px)" in css
    assert "@media (max-width: 1280px)" in css
    assert "--dragon-dataset-content-width: 1500px;" not in css
    assert ".dragon-dataset-library { position: static; order: -1; height: auto; max-height: none; }" in css
    assert ".dragon-page-wrapper.dragon-dataset-page-host" in css
    assert "animation: none;" in css
    assert ".dragon-dataset-page.dragon-in-view" in css
    assert ".dragon-dataset-savebar" in css
    assert "position: fixed;" in css
    assert "right: calc(var(--dragon-dataset-page-gutter) + var(--dragon-dataset-library-width) + var(--dragon-dataset-workspace-gap));" in css
    assert ".dragon-dataset-savebar { right: var(--dragon-dataset-page-gutter); }" in css
    assert "#stage-resolution-dialog" in css
    assert ".dataset-preview-dialog" in css
    assert ".dragon-dataset-group-action { width: 40px; height: 40px; }" in css
    assert "grid-template-columns: repeat(3, minmax(0, 1fr));" in css


def test_dragon_dataset_release_token_is_consistent() -> None:
    bootstrap_token = "dragon-ui-20260902-training-nav-v3"
    entry_token = "dragon-ui-20260902-training-nav-v3"
    style_token = "dragon-ui-20260902-training-nav-v3"
    shell_token = "dragon-ui-20260902-training-nav-v3"
    config_page_token = "dragon-ui-20260901v154"
    page_token = "dragon-ui-20260825v118"
    fields_token = "dragon-ui-20260828v54"
    config_style_token = "dragon-ui-20260902-lokr-availability-v1"
    shared_style_token = "dragon-ui-20260902v78"
    dataset_style_token = "dragon-ui-20260902v91"
    index_html = INDEX_HTML.read_text(encoding="utf-8")
    bootstrap = _read("js/ui-bootstrap.js")
    entry = _read("js/dragon-ui/index.js")
    page_loaders = _read("js/dragon-ui/page-loaders.js")
    page = _read("js/dragon-ui/pages/dataset-editor.js")
    preview_controller = _read("js/dragon-ui/pages/dataset-preview-controller.js")
    preview = _read("js/dragon-ui/pages/dataset-editor-preview.js")
    stylesheet = _read("css/dragon-style.css")
    route_styles = _read("js/dragon-ui/route-styles.js")

    assert f"dragon-style.css?v={style_token}" in index_html
    assert f"ui-bootstrap.js?v={bootstrap_token}" in index_html
    assert f"dragon-ui/index.js?v={entry_token}" in bootstrap
    assert f"dragon-style.css?v={style_token}" in bootstrap
    assert f"router.js?v={shell_token}" in entry
    assert "nav.js?v=dragon-ui-20260902-training-nav-v3" in entry
    assert f"config-page.js?v={config_page_token}" in page_loaders
    assert "route-styles.js?v=dragon-ui-20260902-training-nav-v3" in page_loaders
    assert "dataset-editor.js?v=dragon-ui-20260902v138" in page_loaders
    assert "dataset-preview-controller.js?v=dragon-ui-20260902v8" in page
    assert "dataset-editor-preview.js?v=dragon-ui-20260902v53" in preview_controller
    assert "dataset-preview-window.js?v=dragon-ui-20260831v3" in preview
    assert "dataset-preview-detail.js?v=dragon-ui-20260902v4" in preview_controller or "dataset-preview-detail.js?v=dragon-ui-20260902v4" in _read("js/dragon-ui/pages/dataset-editor-preview.js")
    assert f"dataset-editor-fields.js?v={fields_token}" in page
    assert "dataset-editor-presets.js?v=dragon-ui-20260824v71" in page
    assert f"06-dragon-pages.css?v={dataset_style_token}" in route_styles
    assert f"04-dragon-config.css?v={config_style_token}" in route_styles
    assert "04a-dragon-training-presets.css?v=dragon-ui-20260817v84" in route_styles
    assert "06a-dragon-shared-dialogs.css?v=dragon-ui-20260902v78" in route_styles


def test_dragon_dataset_preset_library_exposes_current_and_group_exports():
    page = _read("js/dragon-ui/pages/dataset-editor.js")
    presets = _read("js/dragon-ui/pages/dataset-editor-presets.js")
    css = _read("css/dragon/06-dragon-pages.css")

    assert 'data-preset-action="export"' in presets
    assert "button.dataset.presetAction === 'export'" in page
    assert "exportCurrentPreset(root, state)" in page
    assert "/api/config/file-groups/${encodeURIComponent(group.id)}/export?kind=dataset" in presets
    assert 'title="导出分组 ZIP"' in presets
    assert "grid-template-columns: repeat(3, minmax(0, 1fr));" in css
