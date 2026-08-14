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
    assert "DATASET_SETTING_KEYS.filter" in page
    assert "当前训练配置正在使用这个预设" in page
    assert "await applyDatasetPreset(api, result.file, state.context.configFile)" in page
    assert '<main class="dragon-dataset-editor-panel">' not in page


def test_dragon_dataset_preset_group_management_contract() -> None:
    page = _read("js/dragon-ui/pages/dataset-editor.js")
    presets = _read("js/dragon-ui/pages/dataset-editor-presets.js")

    for marker in (
        'data-preset-action="new-group"',
        'data-preset-group-action="rename"',
        'data-preset-group-action="delete"',
        "data-preset-move-select",
        "data-preset-move-file",
    ):
        assert marker in presets

    for label in (
        "新建预设分组",
        "重命名分组",
        "只删除分组，不删除其中的 TOML",
        "移动到分组…",
        "请先选择目标分组",
    ):
        assert label in presets or label in page

    assert "api('/api/config/file-groups'," in presets
    assert "method: 'POST'" in presets
    assert "kind: 'dataset'" in presets
    assert "api(`/api/config/file-groups/${encodeURIComponent(groupId)}`" in presets
    assert "method: 'PATCH'" in presets
    assert "method: 'DELETE'" in presets
    assert "api('/api/config/file-groups/move-file'," in presets
    assert "JSON.stringify({ file, group: groupId })" in presets


def test_dragon_dataset_editor_bindings_and_stage_bridge_are_safe() -> None:
    page = _read("js/dragon-ui/pages/dataset-editor.js")
    stage_model = _read("js/features/config-form/stage-resolution-model.js")
    stage_dialog = _read("js/features/config-form/stage-resolution-ui-dialog.js")

    assert "button.dataset.dragonDatasetBound === 'true'" in page
    assert "stage-resolution-ui.js?v=module-bootstrap-20260809-nf4-v2" in page
    for bridge in (
        "app-context-bridge.js?v=module-bootstrap-20260809-nf4-v2",
        "config-state-bridge.js?v=module-bootstrap-20260809-nf4-v2",
        "dataset-state-bridge.js?v=module-bootstrap-20260809-nf4-v2",
        "runtime-bridge.js?v=module-bootstrap-20260809-nf4-v2",
        "toml-action-state-bridge.js?v=module-bootstrap-20260809-nf4-v2",
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
    css = _read("css/dragon/06-dragon-pages.css")

    assert ".dragon-page-wrapper.dragon-dataset-page-host" in css
    assert "animation: none;" in css
    assert ".dragon-dataset-page.dragon-in-view" in css
    assert ".dragon-dataset-savebar" in css
    assert "position: fixed;" in css
    assert "#stage-resolution-dialog" in css
    assert "#dataset-preview-dialog" in css
    assert ".dragon-dataset-group-action { width: 40px; height: 40px; }" in css
    assert "grid-template-columns: repeat(3, minmax(0, 1fr));" in css


def test_dragon_dataset_release_token_is_consistent() -> None:
    entry_token = "dragon-ui-20260814v45"
    page_token = "dragon-ui-20260814v43"
    index_html = INDEX_HTML.read_text(encoding="utf-8")
    bootstrap = _read("js/ui-bootstrap.js")
    entry = _read("js/dragon-ui/index.js")
    page = _read("js/dragon-ui/pages/dataset-editor.js")
    stylesheet = _read("css/dragon-style.css")

    assert f"dragon-style.css?v={entry_token}" in index_html
    assert f"ui-bootstrap.js?v={entry_token}" in index_html
    assert f"dragon-ui/index.js?v={entry_token}" in bootstrap
    assert f"router.js?v={entry_token}" in entry
    assert f"nav.js?v={entry_token}" in entry
    assert f"config-page.js?v={page_token}" in entry
    assert f"dataset-editor.js?v={page_token}" in entry
    assert page.count(page_token) >= 6
    assert stylesheet.count(page_token) == 7
    assert f"06a-dragon-shared-dialogs.css?v={entry_token}" in stylesheet
