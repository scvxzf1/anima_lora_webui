# Split from test_training_frontend_state.py (history)

from __future__ import annotations

from tests.frontend_test_support import *  # noqa: F403


import tests.frontend_test_support as _frontend_support
for _k, _v in vars(_frontend_support).items():
    if not _k.startswith("__"):
        globals()[_k] = _v

def test_history_artifact_helper_builds_urls() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for anima-app history artifact helper checks")
    script = r"""
import { makeHistoryArtifactUrl } from './web/static/js/features/anima-app/helpers/history-artifacts.js';

const result = {
    basic: makeHistoryArtifactUrl({ id: 'task 1' }, 'runtime-config'),
    download: makeHistoryArtifactUrl({ id: 'task/2' }, 'logs/latest.txt', { download: true }),
    missingTask: makeHistoryArtifactUrl({ id: '' }, 'logs'),
    missingKey: makeHistoryArtifactUrl({ id: 'task' }, ''),
};

console.log(JSON.stringify(result));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert json.loads(result.stdout) == {
        "basic": "/api/training/history/task%201/artifacts/runtime-config",
        "download": "/api/training/history/task%2F2/artifacts/logs%2Flatest.txt?download=1",
        "missingTask": "#",
        "missingKey": "#",
    }


def test_queue_and_history_detail_literal_dom_ids_match_index_html() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    ids_by_module = {
        "queue/render.js": _literal_get_element_by_id_targets(
            _frontend_feature_text(
                "js/features/queue/render.js",
                "js/features/queue/render-labels.js",
                "js/features/queue/render-items.js",
            )
        ),
        "queue/actions.js": _literal_get_element_by_id_targets(
            _frontend_module_text("js/features/queue/actions.js")
        ),
        "history-detail/dialog.js": _literal_get_element_by_id_targets(
            _frontend_module_text("js/features/history-detail/dialog.js")
        ),
    }

    assert "training-queue-manager-list" in ids_by_module["queue/render.js"]
    assert "training-queue-failure-policy" in ids_by_module["queue/actions.js"]
    assert "history-detail-dialog" in ids_by_module["history-detail/dialog.js"]

    missing = {
        name: sorted(
            dom_id
            for dom_id in ids
            if f'id="{dom_id}"' not in html
        )
        for name, ids in ids_by_module.items()
    }
    assert not any(missing.values()), missing


def test_image_test_ui_draft_persistence_and_history_reload_hooks_exist() -> None:
    image_test_index = _frontend_module_text("js/features/image-test/index.js")
    image_test_render = _frontend_module_text("js/features/image-test/render.js")
    image_test_gallery = _frontend_module_text("js/features/image-test/gallery.js")
    image_test_storage = _frontend_module_text("js/features/image-test/storage.js")
    image_test_api = _frontend_module_text("js/features/image-test/api.js")

    assert "const draftStore = createImageTestUiStorage();" in image_test_index
    assert "initialHistoryFilter: draftStore.storedHistoryRange()" in image_test_index
    assert "state.restoredDraftFieldIds = draftStore.restoreToDom();" in image_test_index
    assert "draftStore.bind((fieldId) => {" in image_test_index
    assert "state.restoredDraftFieldIds.add(fieldId);" in image_test_index
    assert "draftStore.persistFromDom({ history_range: nextRange });" in image_test_index
    assert "void loadImageTestImages({ force: true, historyRange: nextRange });" in image_test_index
    assert "draftStore.restoreDeferredField('image-test-gpu-index');" in image_test_index
    assert "draftStore.restoreDeferredField('image-test-weight-select');" in image_test_index
    assert "draftStore.persistFromDom({ history_range: renderer.currentHistoryFilter() });" in image_test_index
    assert "const historyRange = options.historyRange || renderer.currentHistoryFilter();" in image_test_index
    assert "loadImageTestGpus({ force: options.force })" in image_test_index
    assert "fetchImageTestGpus(ctx)" in image_test_index
    assert "deleteImageTestImagesRequest(ctx, { files })" in image_test_index
    assert "fetchImageTestImages(ctx, IMAGE_TEST_IMAGE_LIMIT, historyRange)" in image_test_index
    assert "if (state.restoredDraftFieldIds.has(id)) return;" in image_test_index
    assert "bindWeightDropTargetEvents" in image_test_index
    assert "resolveWeightPathFromCandidates();" in image_test_index
    assert "bindSingleWeightDropTarget" in image_test_index
    assert "clearWeightDropTargetState" in image_test_index
    assert "handleWeightDrop" in image_test_index
    assert "applyDroppedWeightPath" in image_test_index
    assert "resolveDroppedWeightPath" in image_test_index
    assert "if (payload?.ok === false) {" in image_test_index
    assert "resolveWeightPathFromCandidates" in image_test_index
    assert "resolvePreferredWeightOptionByName" in image_test_index
    assert "comparePreferredWeightCandidate" in image_test_index
    assert "weightCandidatePriority" in image_test_index
    assert "droppedSafetensorsPath" in image_test_index
    assert "firstDroppedSafetensorsFileInfo" in image_test_index
    assert "resolveDroppedPathCandidate" in image_test_index
    assert "joinDroppedPath" in image_test_index
    assert "isBareSafetensorsFileName" in image_test_index
    assert "stripSafetensorsExt" in image_test_index
    assert "normalizeDroppedWeightPath" in image_test_index
    assert "image-test-gpu-index" in image_test_index
    assert "image-test-weight-drop-target" in image_test_index
    assert "image-test-weight-path" in image_test_index
    assert "event.dataTransfer.dropEffect = 'copy';" in image_test_index
    assert "renderer.setImageTestStatus(`已读取拖入权重：" in image_test_index
    assert "renderer.setImageTestStatus(`拖入权重解析失败：" in image_test_index
    assert "currentHistoryFilter: () => gallery.currentFilter()" in image_test_render
    assert "initialFilterValue: initialHistoryFilter" in image_test_render
    assert "requestImageDelete" in image_test_render
    assert "requestHistoryReload" in image_test_render
    assert "initialFilterValue = DEFAULT_FILTER_VALUE" in image_test_gallery
    assert "filterValue: normalizeImageTestHistoryRange(initialFilterValue, DEFAULT_FILTER_VALUE)" in image_test_gallery
    assert "params.set('days', normalizedRange);" in image_test_api
    assert "IMAGE_TEST_PERSISTED_FIELD_IDS = Object.freeze([" in image_test_storage
    assert "image-test-gpu-index" in image_test_storage


def test_manual_history_refresh_announces_and_deduplicates_requests() -> None:
    history_source = _frontend_module_text("js/features/history-list/list.js")
    listener_source = _frontend_feature_text("js/features/app-shell/event-listeners.js", "js/features/app-shell/event-listeners-contract.js", "js/features/app-shell/event-listeners-setup.js", "js/features/app-shell/beginner-tooltips.js")

    assert "const HISTORY_REFRESH_BUTTON_LABELS = Object.freeze({" in history_source
    assert "let historyListLoadPromise = null;" in history_source
    assert "function setHistoryRefreshButtonState(state = 'idle') {" in history_source
    assert "if (historyListLoadPromise) {" in history_source
    assert "if (announce) setHistoryRefreshButtonState('pending');" in history_source
    assert "if (announce) setHistoryRefreshButtonState(failed ? 'error' : 'ok');" in history_source
    assert "historyListLoadPromise = null;" in history_source
    assert "loadTrainingHistoryList(options = {})" in history_source
    assert "on('btn-refresh-history', 'click', () => loadTrainingHistoryList({ announce: true }));" in listener_source
    assert "on('btn-history-manager-refresh', 'click', () => loadTrainingHistoryList({ announce: true }));" in listener_source


def test_live_status_merges_current_history_task_without_full_history_fetch() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for live history merge checks")
    script = r"""
import { configureHistoryStateBridge } from './web/static/js/features/anima-app/helpers/history-state-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { configureTrainingStateBridge } from './web/static/js/features/anima-app/helpers/training-state-bridge.js?v=module-bootstrap-20260714-stage-dataset5';

const historyState = {
    historyTasks: [{
        id: 'task-1',
        state: 'compiling',
        job: 'training',
        variant: 'old',
        log_count: 5,
        metric_count: 3,
    }],
};
configureHistoryStateBridge(historyState);
configureTrainingStateBridge({});
const { mergeLiveTrainingHistoryTask } = await import(
    './web/static/js/features/history-list/list.js?live-history-merge-test'
);

const merged = mergeLiveTrainingHistoryTask({
    task_id: 'task-1',
    status: 'running',
    variant: 'lora',
    run_dir: 'output/runs/task-1',
    last_output_at: 1234,
    last_log_id: 9,
    log_count: 9,
    metric_count: 2,
});
const missing = mergeLiveTrainingHistoryTask({ task_id: 'missing', status: 'running' });
console.log(JSON.stringify({ merged, missing, tasks: historyState.historyTasks }));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    assert payload["merged"] is True
    assert payload["missing"] is False
    assert payload["tasks"] == [{
        "id": "task-1",
        "state": "running",
        "job": "training",
        "variant": "lora",
        "log_count": 9,
        "metric_count": 3,
        "run_dir": "output/runs/task-1",
        "updated_at": 1234,
    }]


def test_history_list_imports_preview_helpers_for_refresh() -> None:
    """Regression: loadTrainingHistoryList must not throw on renderPreviewTaskSelect."""
    history_source = _frontend_module_text("js/features/history-list/list.js")
    load_section = _section(
        history_source,
        "export async function loadTrainingHistoryList",
        "export async function loadHistoryCollectionSettings",
    )
    assert "renderPreviewTaskSelect()" in load_section
    assert "setPreviewStatus(" in load_section
    _assert_imports_from(
        history_source,
        "../anima-app/helpers/preview-view-bridge.js",
        ("renderPreviewTaskSelect", "setPreviewStatus"),
    )


def test_history_list_marks_queue_tasks() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    history_task_source = _frontend_module_text(
        "js/features/anima-app/chunks/33-create-history-task-item.js"
    )

    queue_impl = _frontend_feature_text(
        "js/features/anima-app/chunks/32-history-task-collection-label.js",
        "js/features/history-list/task-collections.js",
    )
    queue_label = _section(queue_impl, "function historyQueueLabel", "function historyContinueLabel")
    task_item = _section(
        history_task_source,
        "function createHistoryTaskItem",
        "function compactPathLabel",
    )

    assert "来自队列" in queue_label
    assert "queue_attempt" in queue_label
    assert "historyQueueLabel(task)" in task_item


def test_history_task_dialog_busy_state_uses_toml_state() -> None:
    dialog_source = _frontend_feature_text(
        "js/features/anima-app/chunks/34-show-history-collection-select-dialog.js",
        "js/features/history-list/task-dialogs.js",
    )
    dialog_section = _section(
        dialog_source,
        "function showHistoryTaskDialog",
        "function normalizeHistoryDetailTab",
    )

    assert "import { getTomlState }" in dialog_source
    assert "const tomlState = getTomlState();" in dialog_source
    assert "tomlState.sharedDialogBusy" in dialog_section
    assert "sharedDialogBusy" not in dialog_section.replace("tomlState.sharedDialogBusy", "")


def test_history_manager_frontend_hooks_are_present() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    legacy_source = _anima_app_container_text()
    chart_source = CHART_JS.read_text(encoding="utf-8")
    html = INDEX_HTML.read_text(encoding="utf-8")
    css = STYLE_CSS.read_text(encoding="utf-8")
    preview_index = _frontend_module_text("js/features/preview/index.js")
    preview_workspace = _frontend_module_text("js/features/preview/workspace.js")
    preview_state = _frontend_module_text("js/features/preview/state.js")
    history_detail_index = _frontend_module_text("js/features/history-detail/index.js")
    history_detail_api = _frontend_module_text("js/features/history-detail/api.js")
    history_detail_dialog = _frontend_module_text("js/features/history-detail/dialog.js")
    history_detail_state = _frontend_module_text("js/features/history-detail/state.js")
    history_detail_workspace = _frontend_module_text("js/features/history-detail/workspace.js")
    history_detail_source = _frontend_feature_text(
        "js/features/history-detail/index.js",
        "js/features/history-detail/api.js",
        "js/features/history-detail/dialog.js",
        "js/features/history-detail/overview.js",
        "js/features/history-detail/resume/index.js",
        "js/features/history-detail/resume/state.js",
        "js/features/history-detail/resume/panel.js",
        "js/features/history-detail/resume/detail.js",
        "js/features/history-detail/resume/actions.js",
        "js/features/history-detail/analysis.js",
        "js/features/history-detail/curve/index.js",
        "js/features/history-detail/curve/data.js",
        "js/features/history-detail/curve/toolbar.js",
        "js/features/history-detail/curve/chart.js",
        "js/features/history-detail/curve/hover.js",
        "js/features/history-detail/system.js",
        "js/features/history-detail/logs.js",
        "js/features/history-detail/config-files.js",
        "js/features/history-detail/workspace.js",
        "js/features/history-detail/ui.js",
    )
    tabs_source = _frontend_module_text("js/features/app-shell/tabs.js")
    history_curve_chart = _frontend_module_text("js/features/history-detail/curve/chart.js")

    history_list_source = _frontend_module_text("js/features/history-list/list.js")
    history_section = _frontend_feature_text(
        "js/features/history-list/list.js",
        "js/features/anima-app/chunks/27-render-history-collections-workbench.js",
        "js/features/history-list/collections-workbench.js",
        "js/features/anima-app/chunks/28-history-collection-search-text.js",
        "js/features/history-list/workbench-cards.js",
        "js/features/history-list/workbench-collection-card.js",
        "js/features/history-list/workbench-config-group-card.js",
        "js/features/history-list/workbench-order.js",
        "js/features/anima-app/chunks/32-history-task-collection-label.js",
        "js/features/history-list/task-collections.js",
        "js/features/anima-app/chunks/33-create-history-task-item.js",
        "js/features/anima-app/chunks/34-show-history-collection-select-dialog.js",
        "js/features/history-list/task-dialogs.js",
        "js/features/anima-app/helpers/history-collections-bridge.js",
    )
    detail_section = history_detail_source
    listener_section = _section(legacy_source, "function setupEventListeners", "function installBeginnerTooltips")
    preview_open_section = _section(preview_index, "async function openTrainingPreview", "function openCurrentTrainingPreview")
    tab_setup_section = _section(tabs_source, "function setupTabs()", "return {")
    sidebar_history_section = _section(history_list_source, "function renderTrainingHistoryList()", "function recentTrainingSidebarTasks")
    recent_sidebar_section = _section(history_list_source, "function recentTrainingSidebarTasks()", "function renderHistoryManager")
    log_append_section = _section(legacy_source, "function appendLogRecord", "async function replayTrainingLogs")
    history_review_mode_section = _section(legacy_source, "function isHistoryReviewMode()", "function openTutorialDialog")
    sidebar_task_item_section = _section(legacy_source, "function createHistoryTaskItem", "function createHistoryActionButton")
    manager_row_section = _section(legacy_source, "function createHistoryManagerRow", "function selectedHistoryConfigGroups")
    collection_card_section = _section(legacy_source, "function createHistoryCollectionWorkbenchCard", "function createHistoryConfigGroupWorkbenchCard")
    config_card_section = _section(legacy_source, "function createHistoryConfigGroupWorkbenchCard", "function historyCollectionNamesForTasks")
    load_task_section = _section(history_detail_index, "async function loadHistoryTask", "function clearHistoryDetailState")
    chart_controls_section = _section(html, '<div class="live-chart-controls"', '<label class="live-chart-field">')
    monitor_view_section = _section(html, '<section id="training-monitor-view"', '<!-- 预览结果工作区')

    assert "training-history-manager" in html
    assert "history-manager-search" in html
    assert "history-collection-search" in html
    assert "history-config-group-search" in html
    assert "集合搜索" in html
    assert "配置组搜索" in html
    assert "history-group-mode" not in html
    assert "集合分组" not in html
    assert "集合管理" in html
    assert '<option value="config">配置分组</option>' not in html
    assert '<option value="flat">平铺列表</option>' not in html
    assert "btn-history-collections-workbench" in html
    assert "btn-preview-training-results" in html
    assert "当前预览" in html
    assert "btn-live-sampling-preview" in html
    assert "途中采样" in html
    assert chart_controls_section.index("btn-live-sampling-preview") < chart_controls_section.index("live-chart-toggle-lr")
    assert monitor_view_section.index('id="history-config-panel"') < monitor_view_section.index('id="history-resume-panel"')
    assert monitor_view_section.index('id="history-resume-panel"') < monitor_view_section.index('class="panel log-panel"')
    assert "预览结果" in html
    assert 'data-tab="preview"' not in html
    assert 'class="preview-workspace-host" hidden aria-hidden="true"' in html
    assert "preview-page-mount" in html
    assert "preview-workspace" in html
    assert "preview-panel-dialog" in html
    assert "preview-dialog-mount" in html
    assert "btn-close-preview-panel" in html
    assert "training-dashboard" in html
    assert "training-run-state" in html
    assert "training-run-summary" in html
    assert "metric-vram-peak" in html
    assert "metric-gpu-peak" in html
    assert "metric-temp" in html
    assert "metric-temp-peak" in html
    assert "metric-eta" in html
    assert "预计完成" in html
    assert "最近训练" in html
    assert "未归档 · 最新 6 个训练任务" in html
    assert "btn-open-history-manager" in html
    assert 'type="module" src="/static/app.js?v=' in html
    assert "import { MetricsChart } from './chart.js?v=module-bootstrap-" in source
    assert "style.css?v=" in html
    assert "app.js?v=" in html
    assert "history-bulk-bar" in html
    assert "history-bulk-primary-actions" in html
    assert "归档已选" in html
    assert "btn-history-bulk-delete" in html
    assert "设置集合" in html
    assert 'id="history-detail-panel"' not in html
    assert 'id="history-detail-dialog"' in html
    assert "history-detail-dialog-shell" in html
    assert "btn-close-history-detail" in html
    assert "history-detail-tabs" in html
    assert "HISTORY FORGE" in html
    assert "function renderHistoryDetailTabs" in history_detail_dialog
    assert "btn.dataset.historyDetailTab = item.key" in history_detail_dialog
    assert "function historyDetailTabsForPayload" in history_detail_dialog
    assert "const contentCache = {" in history_detail_dialog
    assert "function syncHistoryDetailContentCache(payload)" in history_detail_dialog
    assert "function selectHistoryDetailTab(tab)" in history_detail_dialog
    assert "renderHistoryDetailContent({ reuseCached: true })" in history_detail_dialog
    assert "contentCache.nodes.get(state.detailTab)" in history_detail_dialog
    assert "task?.job === 'preprocess'" in history_detail_dialog
    assert "['overview', 'logs', 'config_files'].includes(item.key)" in history_detail_dialog
    assert "normalizeVisibleHistoryDetailTab(payload, state.detailTab)" in history_detail_dialog
    assert "mainTaskReturn: null" in history_detail_state
    assert "linked_preprocess_task" in history_detail_dialog
    assert "查阅预处理" in history_detail_dialog
    assert "返回主项目" in history_detail_dialog
    assert "openLinkedPreprocessTask(task, preprocessTask)" in history_detail_dialog
    assert "loadHistoryTaskInDetail(preprocessTaskId, { detailTab: 'overview' })" in history_detail_dialog
    assert "loadHistoryTaskInDetail(target.taskId, { detailTab: target.detailTab || 'overview' })" in history_detail_dialog
    assert history_detail_dialog.index("查阅预处理") < history_detail_dialog.index("deps.createHistoryTaskPreviewButton(task)")
    assert "state.mainTaskReturn = null;" in load_task_section
    assert "{ key: 'overview', label: '概览' }" in history_detail_state
    assert "{ key: 'analysis', label: '训练分析' }" in history_detail_state
    assert "{ key: 'preview', label: '样张与权重' }" in history_detail_state
    assert "{ key: 'config_files', label: '配置与文件' }" in history_detail_state

    assert "renderHistoryManager()" in history_section
    assert "params.set('include_archived', '1');" in history_section
    assert "params.set('limit', '500')" not in history_section
    assert "recentTrainingSidebarTasks" in history_section
    assert "groupHistoryTasks(" not in sidebar_history_section
    assert "task.job === 'training' && !historyTaskIsArchived(task)" in recent_sidebar_section
    assert ".slice(0, 6)" in recent_sidebar_section
    assert "historyManagerFilteredTasks" in history_section
    assert "function historyManagerBaseFilteredTasks" in history_section
    assert "function historyManagerVisibleTasks" in history_section
    assert "normalizeHistoryGroupMode" not in source
    assert "const baseVisible = historyManagerBaseFilteredTasks();" in history_section
    assert "const visible = historyManagerVisibleTasks(baseVisible);" in history_section
    assert "mode === 'config'" not in history_section
    assert "mode === 'flat'" not in history_section
    assert "historyConfigGroupVisibleForSearch" not in source
    assert "uniqueHistoryTasks" in history_section
    assert "createHistoryManagerRow" in history_section
    assert "renderHistoryManagerGrouped" not in source
    assert "resetTrainingExpandedStateOnLeave" in history_section
    assert "collapseVisibleHistoryManagerGroups" not in source
    assert "collapsedHistoryCollections" not in source
    assert "collapsedHistoryConfigGroups" not in source
    assert "historyConfigGroupCollapseKey" not in source
    assert "expandHistoryCollectionConfigGroups" not in source
    assert "historyStatFilterIsActive" in history_section
    assert "archived: 'all'" in history_section
    assert "next.kind = state" in history_section
    assert "createHistoryManagerCollectionSection" not in source
    assert "createHistoryManagerConfigGroupSection" not in source
    assert "list.dataset.groupMode = 'collections';" in history_section
    workbench_fill = _frontend_module_text("js/features/history-list/workbench-chunk-fill.js")
    assert (
        "collection: selectedCollection" in history_section
        or "collection: selectedCollection" in workbench_fill
    )
    assert "history-current-group-content" in history_section
    assert "history-collection-nav" in history_section
    assert "HISTORY_UNGROUPED_COLLECTION_KEY" in source
    assert "selectedHistoryCollectionKey = HISTORY_UNGROUPED_COLLECTION_KEY" in source
    assert "未分类任务" in source
    assert "分组导航" in source
    assert "新建分组" in source
    assert "renderHistoryCollectionsWorkbench" in history_section
    assert "createHistoryCollectionWorkbenchCard" in history_section
    assert "createHistoryConfigGroupWorkbenchCard" in history_section
    assert "historyCollectionWorkbenchTarget" in source
    assert "historyCollectionSettings" in source
    assert "historyCollectionSearch" in source
    assert "historyConfigGroupSearch" in source
    assert "historyTaskMatchesCollectionSearch" in source
    assert "historyTaskMatchesConfigGroupSearch" not in source
    assert "historyCollectionMatchesSearch(collection, terms)" in source
    assert "selectedHistoryCollectionForWorkbench(collections, collectionSearchTerms)" in source
    assert "visibleHistoryCollectionsForSearch(allCollections, collectionSearchTerms)" in source
    assert "collectionSearchTerms.length ? candidates[0] : null" in source
    assert "createHistoryCollectionSearchEmptyCollection" in source
    assert "historySearchTextMatches(historyConfigGroupSearchText(group), configSearchTerms)" in source
    assert "on('history-collection-search', 'input'" in source
    assert "on('history-config-group-search', 'input'" in source
    assert "selectedHistoryCollectionKey" in source
    assert "/api/training/history/collections/settings" in source
    assert "loadHistoryCollectionSettings" in source
    assert "saveHistoryCollectionSettings" in source
    assert "collection_order" in source
    assert "config_group_order" in source
    assert "if (aIndex < 0) return -1;" in source
    assert "if (bIndex < 0) return 1;" in source
    assert "moveHistoryCollection" in source
    assert "moveHistoryConfigGroup" in source
    assert "reorderHistoryCollectionValue" in source
    assert "moveItemNearList" in source
    assert "moveHistoryCollection(collection, 'top', allCollections)" in collection_card_section
    assert "moveHistoryCollection(collection, 'bottom', allCollections)" in collection_card_section
    for direction in ("top", "up", "down", "bottom"):
        assert f"moveHistoryConfigGroup(group, '{direction}', options.groups, options.collection)" in config_card_section
    assert "applySelectedHistoryTasksToCollection" in source
    assert "groupHistoryTasks(scopedTasks)" in history_section
    assert "task?.group" in source
    assert "设置集合" in source
    assert "清除集合" in source
    assert "搜索或新建集合" in source
    assert "未分类" in source
    assert "selectedHistoryCollectionKey === collection.key ? '' : collection.key" not in source
    assert "selectHistoryCollectionInWorkbench(collection.key)" in source
    assert "historyState.selectedHistoryCollectionKey = key" in source
    assert "collection.is_ungrouped ? '未分类' : '移入'" in collection_card_section
    assert "if (selectedTaskCount > 0) actions.append(joinSelectedBtn);" in collection_card_section
    assert "加入目标" in source
    assert "查看集合" not in source
    assert "取消查看" not in source
    assert "选择分组" in source
    assert "拖拽分组调整顺序" in collection_card_section
    assert "history-collection-drag-handle" in collection_card_section
    assert "beginHistoryCollectionDrag(event, collection)" in collection_card_section
    assert "dropHistoryCollectionToSort(event, collection, allCollections)" in collection_card_section
    for label in ("置顶", "上移", "下移", "置底"):
        assert f"createHistoryManagerGroupButton('{label}'" in collection_card_section
    for label in ("置顶", "上移", "下移", "置底"):
        assert f"createHistoryManagerGroupButton('{label}'" in config_card_section
    assert "合并查看" in source
    assert "查阅分组详情" in source
    assert "createHistoryActionButton('配置'" in source
    assert "createHistoryTaskConfigButton(task)" in manager_row_section
    assert "createHistoryMoreActions([" in manager_row_section
    assert "compactHistoryPathLabel" in source
    assert "historyCompactGroupMetaParts" in source
    assert "history-compact-meta" in source
    assert "history-more-actions" in source
    assert "loadHistoryTask(task.id, { detailTab: 'config_files' })" in source
    assert "if (options.detailTab)" in source
    assert "normalizeHistoryDetailTab(options.detailTab)" in source
    assert "任务预览" in source
    assert "分组预览" in source
    assert "只查看这一次训练任务的样张和权重" in source
    assert "汇总查看这个配置分组下所有训练任务的样张和权重" in source
    assert "createHistoryConfigGroupMergeButton" in source
    assert "createHistoryConfigGroupPreviewButton" in source
    assert "loadConfigGroupTimeline(group, { skipSelectionDialog: true })" in source
    assert "查阅这个自动配置分组内的训练日志、Loss 曲线和任务明细" in source
    assert "openHistoryConfigGroupPreview(group)" in source
    assert "loadConfigGroupTimeline(group, { skipSelectionDialog: true, detailTab: 'preview' })" in source
    assert "loadHistoryTask(task.id, { detailTab: 'preview' })" in source
    assert "mountPreviewWorkspaceInHistoryDetail" in source
    assert "restorePreviewWorkspaceFromHistoryDetail" in source
    assert "history-detail-preview-mount" in history_detail_workspace
    assert "canPreviewHistoryConfigGroup" in source
    assert "normalizePreviewGroup" in preview_state
    assert "state.selectedGroup = normalizePreviewGroup(options.group)" in preview_index
    assert "openPreviewPanel" in source
    assert "closePreviewPanel" in source
    assert "mountPreviewWorkspaceInDialog" in preview_workspace
    assert "mountPreviewWorkspaceInPage" in preview_workspace
    assert "workspace.openPreviewPanel();" in preview_open_section
    assert "document.querySelector('[data-tab=\"preview\"]')?.click()" not in preview_open_section
    assert "if (nextTab === 'preview')" not in tab_setup_section
    assert "mountPreviewWorkspaceInPage();" not in tab_setup_section
    assert "btn-preview-training-results" in listener_section
    assert "btn-live-sampling-preview" in listener_section
    assert "openCurrentTrainingPreview" in source
    assert "openLiveSamplingPreview" in source
    assert "const historyTaskId = deps.getTrainingViewMode() === 'live'" in preview_index
    assert "state.selectedTaskId = historyTaskId;" in preview_index
    assert "getViewingHistoryTaskId: () => historyState.viewingHistoryTaskId" in source
    assert "event?.preventDefault?.()" in source
    assert "event?.stopPropagation?.()" in source
    assert "on('btn-preview-training-results', 'click', openCurrentTrainingPreview)" in listener_section
    assert "on('btn-live-sampling-preview', 'click', openLiveSamplingPreview)" in listener_section
    assert "chooseTimelineTasksForMerge" not in source
    assert "showTimelineTaskSelectionDialog" not in source
    assert "选择要合并查看的训练分组" not in source
    assert "选择合并查看" not in source
    assert "分布在 ${split.size} 个分组" in source
    assert "selectedHistoryTaskIds" in source
    assert "applyHistoryBatchAction" in source
    assert "deleteHistoryTasksThorough" in source
    assert "confirmed: true" in source
    assert "confirm_text: confirmText" not in source
    assert "const confirmText = '彻底删除';" not in source
    assert "title: '确认要删吗'" in source
    assert "confirmText: '确认要删吗'" in source
    assert "输入“彻底删除”确认" not in source
    assert "彻底删除" in source
    assert "runtime_cleanup_errors" in source
    assert "历史记录已删除，部分文件未清理" in source
    assert "detailLines: cleanupErrors" in source
    assert "/api/training/history/batch" in source
    assert "openHistoryDetailDialog" in source
    assert "closeHistoryDetailDialog" in source
    assert "createHistoryDetailFeature(ctx, {" in legacy_source
    for name in (
        "loadHistoryTask",
        "renderHistoryManagerDetail",
        "renderHistoryDetailDialog",
        "closeHistoryDetailDialog",
        "isHistoryDetailDialogOpen",
        "handleHistoryDetailWindowKeydown",
        "loadResumeOptionsForTask",
        "clearResumeOptions",
        "renderResumePanelState",
        "selectedResumeCheckpoint",
        "resumeTrainingFromCheckpoint",
        "selectedHistoryManagerResumeCheckpoint",
        "resumeTrainingFromHistoryDetail",
        "setResumeStatus",
        "getCurrentPayload",
        "getActiveTab",
        "setActiveTab",
        "clearHistoryDetailContentCache",
    ):
        assert name in history_detail_index
    assert "fetchHistoryTask(ctx, taskId)" in history_detail_index
    assert "/api/training/history/${encodeURIComponent(taskId)}" in history_detail_api
    assert "/api/training/history/${encodeURIComponent(taskId)}/resume-options" in history_detail_api
    assert "/api/preview/weights?task_id=" in history_detail_api
    assert "/api/training/continue-lora/inspect" in history_detail_api
    assert "/api/training/queue/resume" in history_detail_api
    assert "/api/training/resume" in history_detail_api
    assert "requestContinueLoraInspection" in legacy_source and "06-stronger-selective-checkpoint-value.js?v=module-bootstrap-" in legacy_source
    assert "inspectContinueLoraWeight: requestContinueLoraInspection" in legacy_source
    assert "globalThis.requestContinueLoraInspection" not in legacy_source
    assert "正在审查可热启动权重..." in history_detail_source
    assert "reviewHistoryResumeWeights(rawWeights)" in history_detail_source
    assert "inspectHistoryResumeWeight(weightPath)" in history_detail_source
    assert "return historyState.historyViewMode !== 'live';" in history_review_mode_section
    assert "Boolean(historyState.viewingHistoryTaskId)" not in history_review_mode_section
    assert "main.addEventListener('click', () => openSidebarHistoryTask(task.id))" in sidebar_task_item_section
    assert "createHistoryTaskPreviewButton(task)" in sidebar_task_item_section
    assert "createHistoryActionButton('查看', () => openSidebarHistoryTask(task.id))" in sidebar_task_item_section
    assert "renameHistoryTask(task)" not in sidebar_task_item_section
    assert "archiveHistoryTask(task)" not in sidebar_task_item_section
    assert "deleteHistoryTask(task)" not in sidebar_task_item_section
    assert "main.addEventListener('click', () => loadHistoryTask(task.id))" in manager_row_section
    assert "createHistoryActionButton('查看', () => loadHistoryTask(task.id))" in manager_row_section
    assert "function openSidebarHistoryTask" in source
    assert "renderHistoryTask(payload, { stickLogsToBottom: true });" in source
    assert "historyState.historyViewMode = 'task';" in source
    sidebar_history_state = _frontend_module_text("js/features/anima-app/state/history-state.js")
    assert "sidebarHistoryPayloadCache: new Map()" in sidebar_history_state
    assert "sidebarHistoryRequestId: 0" in sidebar_history_state
    assert "syncRecentHistorySidebarSelection()" in source
    assert "rememberSidebarHistoryPayload" in source
    assert "SIDEBAR_HISTORY_LOG_RENDER_LIMIT" in source
    assert "if (requestId !== historyState.sidebarHistoryRequestId) return;" in source
    assert "await openSidebarHistoryTask(historyState.viewingHistoryTaskId);" in source
    assert "showTrainingView('history')" not in load_task_section
    assert "renderHistoryTask(payload)" not in load_task_section
    assert "deps.setViewingHistoryTaskContext({" in load_task_section
    assert "task: payload.task || null" in load_task_section
    assert "dialog.setResumeLoadingForTask(taskId);" in load_task_section
    assert "dialog.renderHistoryManagerDetail(payload, { open: true })" in load_task_section
    assert "await dialog.loadResumeOptionsForTask(taskId);" in load_task_section
    assert "deps.clearViewingHistoryTaskContext?.(state.currentPayload);" in detail_section
    assert "function clearViewingHistoryTaskContext" in source
    assert "historyState.currentHistoryTaskForResume = null;" in _section(source, "function clearViewingHistoryTaskContext", "function handleHistoryDetailWindowKeydown")

    assert "renderHistoryDetailDialog" in detail_section
    assert "renderHistoryDetailOverview" in detail_section
    assert "renderHistoryDetailAnalysis" in detail_section
    assert "renderHistoryDetailResume" in detail_section
    assert "renderHistoryDetailChart" in detail_section
    assert "renderHistoryDetailLogs" in detail_section
    assert "renderHistoryDetailSystem" in detail_section
    assert "renderHistoryDetailConfig" in detail_section
    assert "renderHistoryDetailPaths" in detail_section
    assert "renderHistoryDetailConfigFiles" in detail_section
    assert "renderHistoryDetailPathSummary" in detail_section
    assert "historyCurveState" in detail_section
    assert "renderHistoryCurveStats" in detail_section
    assert "renderHistoryCurveToolbar" in detail_section
    assert "renderHistoryCurveMainChart" in detail_section
    assert "createHistoryCurveSvg" in detail_section
    assert "renderHistoryCurveInspector" in detail_section
    assert "renderHistoryCurveSegments" in detail_section
    assert "function historyCurveMetric(" not in detail_section
    assert "曲线指标" not in detail_section
    assert "HISTORY_CURVE_METRICS" in detail_section
    assert "historyCurvePointHasAnyMetric" in detail_section
    assert "historyCurveRawPointHasAnyMetric" in detail_section
    assert ".filter(historyCurveRawPointHasAnyMetric)" in detail_section
    assert ".map(historyCurveNormalizeRawMetricPoint)" in detail_section
    assert "historyCurveMetricStats" in detail_section
    assert "historyCurveMetricRange" in detail_section
    assert "appendHistoryCurveLineSegments" in detail_section
    assert "renderHistoryCurveLegend" not in detail_section
    assert "drawHistoryCurveMetricPoints" in detail_section
    assert "historyCurveStatsWithHover" in detail_section
    assert "updateHistoryCurveHoverLayer" in detail_section
    assert "renderHistoryCurveInspectorRows" in detail_section
    assert "requestAnimationFrame" in detail_section
    assert "scheduleHoverStep" in detail_section
    assert "renderHistoryDetailContent();" not in _section(history_curve_chart, "function createHistoryCurveSvg", "function renderHistoryCurveSegments")
    assert "dual-metric" in detail_section
    assert "history-curve-hover-layer" in detail_section
    assert "loss-axis" in detail_section
    assert "lr-axis" in detail_section
    assert "学习率点" in detail_section
    assert "最后有效学习率" in detail_section
    assert "峰值学习率" in detail_section
    assert "没有可绘制的 Loss 或学习率数据。" in detail_section
    assert "当前范围没有可绘制的 Loss 或学习率点。请调整范围筛选。" in detail_section
    assert "smoothLoss" in detail_section
    assert "smoothLr" in detail_section
    assert "formatSignedLr" in detail_section
    assert "historyCurveSmoothPoints" in detail_section
    assert "historyCurveFilteredPoints" in detail_section
    assert "historyCurveDisplayPoints" in detail_section
    assert "HISTORY_CURVE_RENDER_POINT_LIMIT" in detail_section
    assert "绘图已降采样" in detail_section
    assert "stageBreakBefore" in detail_section
    assert "display_step" in detail_section
    assert "平滑窗口" in detail_section
    assert "最近100点" in detail_section
    assert "最近25%" in detail_section
    assert "自定义 Step" in detail_section
    assert "box.appendChild(createHistorySparkline(lossPoints));" not in source
    assert "historySystemSummary" in detail_section
    assert "historySystemRecords" in detail_section
    assert "HISTORY_SYSTEM_TABLE_RENDER_LIMIT" in detail_section
    assert "historyDetailLimitNotice" in detail_section
    assert "仅显示最近" in detail_section
    assert "syncHistoryLogConsoleState" in detail_section
    assert "HISTORY_LOG_RENDER_BATCH_SIZE" in detail_section
    assert "consoleRenderToken" in detail_section
    assert "scheduleHistoryLogRenderBatch(() => appendBatch(end))" in detail_section
    assert "pre.dataset.rendering = 'true'" in detail_section
    assert "if (pre.dataset.rendering === 'true') return;" in detail_section
    assert "renderHistoryLogCommandCard" in detail_section
    assert "复制完整命令" in detail_section
    assert "搜索 Error、Epoch..." in detail_section
    assert "historyLogMatchesLevel" in detail_section
    assert "appendAnsiLogText" in detail_section
    assert "stripAnsiCodes" in detail_section
    assert "/logs/download" in detail_section
    assert "下载完整日志" in detail_section
    assert "最后 VRAM" in detail_section
    assert "峰值 GPU" in detail_section
    assert "无系统采样记录" in detail_section
    assert "system.jsonl" in detail_section
    assert "history-detail-metrics-body" in detail_section
    assert "task.job === 'preprocess'" in detail_section
    assert "renderPreprocessHistoryOverview(payload, box)" in detail_section
    assert "预处理摘要" in detail_section
    assert "预处理文件" in detail_section
    assert "history-preprocess-summary-body" in detail_section
    assert "history-preprocess-stat-grid" in detail_section
    assert "compactHistoryPathName(task.dataset_cache_dir || task.run_dir)" in detail_section
    assert "运行时数据集配置" in detail_section
    assert "日志目录" in detail_section
    assert "task.job === 'training'" in detail_section
    assert "historyDetailSection('任务信息'" not in detail_section
    assert "history-detail-section info" not in detail_section
    assert "loadHistoryResumeWeights" in detail_section
    assert "/api/preview/weights?task_id=" in history_detail_api
    assert "diagnostic" in detail_section
    assert "权重热启动" in detail_section
    assert "optimizer、scheduler 和已完成步数" in detail_section

    assert "btn-history-manager-refresh" in listener_section
    assert "btn-history-collections-workbench" in listener_section
    assert "btn-history-bulk-archive" in listener_section
    assert "btn-live-training" in listener_section
    assert "returnToLiveTraining" in listener_section
    assert "history-filter-kind" in listener_section
    assert "history-group-mode" not in listener_section
    assert "groupMode" not in listener_section
    assert "ensureHistoryDetailFeature().bindHistoryDetailEvents();" in listener_section
    assert "history-detail-tab" in history_detail_dialog
    assert "selectHistoryDetailTab(btn.dataset.historyDetailTab)" in history_detail_dialog
    assert "dialog.clearHistoryDetailContentCache();" in history_detail_index
    assert "btn-close-history-detail" in history_detail_dialog
    assert "logBuffer" in source
    assert "logOutputLines" in source
    assert "logRenderToken" in source
    assert "LOG_RENDER_BATCH_SIZE" in source
    assert "scheduleLogRenderBatch(() => appendBatch(end))" in source
    assert "appendLogOutputLines(pending, { stickToBottom })" in log_append_section
    assert "isLogNearBottom(el)" in log_append_section
    assert "scheduleLogFlush" in log_append_section
    assert "requestAnimationFrame" in log_append_section
    assert "MAX_LOG_LINES" in log_append_section
    assert "appendLogOutputLines" in source
    assert "lastLrText" not in source
    assert "recordLearningRateChange" not in source
    assert "announceLr" not in source
    assert "updatePointMetadata" in source
    assert "['Loss', formatLossValue(lossPoint.loss)]" in source
    assert "['平滑 Loss', formatLossValue(lossPoint.smoothLoss)]" in source
    assert "['学习率', formatLr(lrPoint.lr)]" in source
    assert "['平滑学习率', formatLr(lrPoint.smoothLr)]" in source
    assert "lr: item.lr" in source
    assert "peakVramUsedGb" in source
    assert "peakGpuUtil" in source
    assert "peakGpuTemp" in source
    assert "renderLiveTrainingDashboard" in source
    assert "function trainingEtaMetricInfo" in source
    assert "calculateTrainingEtaMetricInfo({" in source
    assert "parseProgressRateSeconds(msg.rate)" in source
    assert "setEtaMetricText(trainingEtaMetricInfo());" in source
    assert "progressSecondsPerStep" in source
    assert "resetLiveSystemPeaks" in source

    assert ".training-workspace.main-wide" in css
    assert ".training-workspace.main-wide .training-sidebar" in css
    assert ".training-workspace.history-mode .training-main" in css
    assert "#tab-training .training-history-manager" in css
    training_history_css = _section(
        css,
        "#tab-training .training-history-manager {",
        "#tab-training .history-forge-eyebrow {",
    )
    assert '"head content"\n        "stats content"\n        "tools content"\n        "bulk content"\n        ". content"' in training_history_css
    assert "grid-template-rows: auto auto auto auto minmax(0, 1fr);" in training_history_css
    assert "#tab-training .history-manager-head {\n    grid-area: head;\n    align-self: start;" in css
    assert "#tab-training .history-manager-stats {\n    grid-area: stats;\n    align-self: start;" in css
    assert "#tab-training .history-manager-tools {\n    grid-area: tools;\n    align-self: start;" in css
    assert "#tab-training .history-bulk-bar {\n    grid-area: bulk;" in css
    assert "#tab-training .history-forge-eyebrow" in css
    assert ".training-workspace.history-wide" not in css
    assert ".training-dashboard-head" in css
    assert ".training-run-state" in css
    assert ".training-run-summary" in css
    assert ".metric-item-eta .metric-value" in css
    assert ".metric-icon-clock::before" in css
    assert ".history-curve-legend" not in css
    assert ".history-curve-legend-swatch" not in css
    assert ".history-curve-axis-label.loss-axis" in css
    assert ".history-curve-axis-label.lr-axis" in css
    assert ".history-curve-line.loss.smooth" in css
    assert ".history-curve-line.lr.smooth" in css
    assert ".history-curve-hover-layer" in css
    assert "pointer-events: none;" in css
    assert ".history-curve-svg.metric-lr .history-curve-line.smooth" not in css
    assert ".training-panels.training-dashboard" in css
    assert ".metrics-panel,\n.chart-panel" in css
    assert "grid-column: 1 / -1;" in css
    assert ".history-manager-grid" in css
    assert ".history-manager-row" in css
    assert ".history-manager-collection" not in css
    assert ".history-manager-config-group" not in css
    assert "--history-manager-bg" in css
    assert "--history-collection-bg" in css
    assert "--history-config-bg" in css
    assert "--history-row-bg" in css
    assert "--history-row-hover-bg" in css
    assert "--history-level-border" in css
    assert "--history-collection-accent" in css
    assert "--history-config-accent" in css
    assert "--history-ungrouped-accent" in css
    assert ".history-manager-collection.ungrouped" not in css
    assert "border-left: 5px solid var(--history-collection-accent)" not in css
    assert "border-left: 3px solid var(--history-config-accent)" not in css
    assert "background: var(--history-row-bg)" in css
    assert "background: var(--history-row-hover-bg)" in css
    assert ".history-collections-workbench.compact" in css
    assert ".history-compact-meta" in css
    assert ".history-more-actions" in css
    assert ".history-more-actions-menu" in css
    assert ".history-row-state.done" in css
    assert ".history-row-state.running" in css
    assert ".history-row-state.queued" in css
    assert ".history-row-state.interrupted" in css
    assert ".history-collections-workbench" in css
    assert "workbench.className = 'history-collections-workbench compact'" in history_section
    assert "`当前: ${selectedCollection.label}`" in history_section
    assert ".history-collections-body" in css
    assert ".history-collection-nav .history-collection-card" in css
    assert "max-height: min(520px, calc(100vh - 18rem));" in css
    assert ".history-manager-list[data-group-mode=\"collections\"]" in css
    assert "height: min(680px, calc(100vh - 14rem));" in css
    assert "height: calc(100vh - 176px);" in css
    assert "max-height: calc(100vh - 176px);" in css
    assert "--history-training-panel-head-height: 42px;" in css
    assert "#tab-training .history-collection-nav-head {" in css
    nav_head_css = _section(css, "#tab-training .history-collection-nav-head {", "#tab-training .history-collection-nav-head .history-collections-panel-title")
    assert "height: var(--history-training-panel-head-height);" in nav_head_css
    assert "min-height: var(--history-training-panel-head-height);" in nav_head_css
    assert "max-height: var(--history-training-panel-head-height);" in nav_head_css
    assert "padding: 0 0.72rem;" in nav_head_css
    nav_title_css = _section(css, "#tab-training .history-collection-nav-head .history-collections-panel-title", "#tab-training .history-collection-create-btn")
    assert "height: auto;" in nav_title_css
    assert "padding: 0;" in nav_title_css
    assert "background: transparent;" in nav_title_css
    assert "grid-template-rows: auto auto minmax(0, 1fr);" in css
    assert "align-items: stretch;" in css
    assert "overflow: hidden;" in css
    assert "overflow-y: scroll;" in css
    assert "scrollbar-gutter: stable;" in css
    assert "overscroll-behavior: contain;" in css
    assert ".history-collection-card" in css
    assert ".history-collection-card.active" in css
    assert ".history-collection-select-dialog" in css
    assert ".history-collection-select-list" in css
    assert ".history-config-group-card" in css
    assert ".history-config-group-task-list" in css
    assert ".history-manager-group-head" not in css
    assert ".history-manager-group-actions" not in css
    assert ".history-manager-stat.active" in css
    assert ".history-detail-dialog" in css
    assert ".history-detail-overview-dashboard" in css
    assert ".history-detail-overview-dashboard.preprocess-task" in css
    assert ".history-preprocess-stat-grid" in css
    assert ".history-preprocess-summary-body" in css
    assert "align-content: start;" in css
    assert ".history-detail-preview" in css
    assert ".history-detail-preview-mount" in css
    assert ".history-detail-progress" in css
    assert ".history-detail-metrics-body" in css
    assert ".history-detail-analysis" in css
    assert ".history-detail-config-files" in css
    assert ".history-curve-workbench" in css
    assert ".history-curve-toolbar" in css
    assert ".history-curve-svg" in css
    assert ".history-curve-inspector" in css
    assert ".history-curve-segment-line" in css
    assert ".history-detail-limit-note" in css
    assert ".history-command-card" in css
    assert ".history-log-console" in css
    assert ".history-log-toolbar" in css
    assert ".history-log-output.history-detail-pre" in css
    assert ".history-log-line.error" in css
    assert ".history-log-line.warning" in css
    assert ".history-log-line.progress" in css
    assert ".history-log-match.current" in css
    assert ".history-log-ansi-red" in css
    assert ".history-system-trends" in css
    assert ".history-system-table" in css
    assert ".history-detail-section.info" not in css
    assert ".preview-panel-dialog" in css
    assert ".preview-panel-body" in css
    assert ".preview-panel-dialog-sampling .preview-layout" in css
    assert ".preview-panel-dialog-sampling .preview-sidebar" in css
    assert "#tab-training .live-chart-sample-btn" in css

    assert "this.lrColor" in chart_source
    assert "_drawLrLine" in chart_source
    assert "updatePointMetadata" in chart_source
    assert "LR:" in chart_source
    assert "_formatLr" in chart_source
    assert "if (value === undefined || value === null || value === '') return '-';" in chart_source
    assert "if (value === undefined || value === null || value === '') return null;" in source


def test_history_collection_drag_drop_frontend_hooks_are_present() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    app_constants = _frontend_module_text("js/features/anima-app/helpers/app-constants.js")
    history_state_source = _frontend_module_text("js/features/anima-app/state/history-state.js")
    css = STYLE_CSS.read_text(encoding="utf-8")

    workbench_impl = _frontend_feature_text(
        "js/features/anima-app/chunks/27-render-history-collections-workbench.js",
        "js/features/history-list/collections-workbench.js",
    )
    workbench = _section(workbench_impl, "function renderHistoryCollectionsWorkbench", "function renderHistoryManagerStats")
    drag_helpers = _section(source, "function historyDragTaskIdsForGroup", "function createHistoryCollectionWorkbenchCard")
    card_impl = _frontend_feature_text(
        "js/features/history-list/workbench-cards.js",
        "js/features/history-list/workbench-collection-card.js",
        "js/features/history-list/workbench-config-group-card.js",
        "js/features/history-list/workbench-order.js",
    )
    collection_card = _section(card_impl, "function createHistoryCollectionWorkbenchCard", "function createHistoryConfigGroupWorkbenchCard")
    config_card = _section(card_impl, "function createHistoryConfigGroupWorkbenchCard", "function historyCollectionNamesForTasks")
    config_card_module = _frontend_module_text("js/features/history-list/workbench-config-group-card.js")
    order_module = _frontend_module_text("js/features/history-list/workbench-order.js")

    # Regression: config-group cards must import the storage-key helper they call at render time.
    # Missing this import crashes history workbench rendering and surfaces as
    # "historyCollectionStorageKey is not defined" when opening task preview/detail.
    assert "export function historyCollectionStorageKey" in order_module
    assert "historyCollectionStorageKey," in config_card_module
    assert "from './workbench-order.js" in config_card_module
    assert "card.dataset.collectionKey = historyCollectionStorageKey(options.collection || '__all__');" in config_card

    assert "export const HISTORY_TASK_DRAG_MIME = 'application/x-anima-history-task-ids';" in app_constants
    assert "export const HISTORY_COLLECTION_DRAG_MIME = 'application/x-anima-history-collection';" in app_constants
    assert "export const HISTORY_CONFIG_GROUP_DRAG_MIME = 'application/x-anima-history-config-group';" in app_constants
    assert "historyDragState: {" in history_state_source
    for key in ("active: false", "taskIds: []", "sourceGroupKey: ''", "activeDropTarget: ''", "pending: false", "popover: {"):
        assert key in history_state_source
    assert "historyCollectionDragState: {" in history_state_source
    for key in ("sourceValue: ''", "dropPosition: 'after'", "pending: false"):
        assert key in history_state_source
    assert "historyConfigGroupSortState: {" in history_state_source
    for key in ("sourceKey: ''", "collectionKey: ''", "activeDropTarget: ''", "dropPosition: 'after'"):
        assert key in history_state_source
    assert "historyConfigGroupPointerDrag: null" in history_state_source
    assert "historyConfigGroupDropPreviewElement: null" in history_state_source
    assert "historyCollectionPointerDrag: null" in history_state_source
    assert "application/x-anima-history-task-ids" in source
    assert "application/x-anima-history-collection" in source
    assert "application/x-anima-history-config-group" in source
    assert "event.dataTransfer.setData(HISTORY_TASK_DRAG_MIME" in source
    assert "event.dataTransfer.setData(HISTORY_COLLECTION_DRAG_MIME" in source
    assert "event.dataTransfer.setData(HISTORY_CONFIG_GROUP_DRAG_MIME" in source
    assert "event.dataTransfer.setData('text/plain'" in source

    assert "collectionList.appendChild(createHistoryCollectionDropzone());" not in workbench
    assert "renderHistoryDropPopover(workbench);" in workbench
    assert "createHistoryCollectionDropzone" not in source
    assert "history-collection-dropzone" not in source
    assert "createHistoryCollectionConfigChip" not in source
    assert "history-collection-config-chip" not in source
    assert "history-collection-create-btn" in workbench
    assert "openHistoryNewCollectionPopover(event, [])" in workbench
    assert "新建分组" in workbench
    assert "historyCompactGroupMetaParts" in source
    assert "history-compact-meta" in source
    assert "createHistoryMoreActions([" in source
    assert "renameHistoryCollection(collection)" in collection_card
    assert "clearHistoryCollection(collection)" in collection_card
    assert "setHistoryCollectionForTasks(collection.tasks, collection.value, collection.label)" not in collection_card
    assert "clearHistoryCollectionForTasks(collection.tasks, collection.label)" not in collection_card
    assert "renameHistoryCollectionOrderValue" in source
    assert "renameHistoryConfigGroupOrderKey" in source
    assert "removeHistoryCollectionSettingValue" in source
    assert "ids.length ? '清空集合' : '删除空集合'" in source

    assert "history-drag-handle" in config_card
    assert "history-config-group-card-head" in config_card
    assert "head.append(select, handle, main, actions)" in config_card
    assert "拖拽配置分组调整顺序或移到右侧分组" in config_card
    assert "history-config-group-drag-handle" in config_card
    assert "handle.draggable = true;" in config_card
    assert "handle.addEventListener('pointerdown'" in config_card
    assert "startHistoryConfigGroupPointerDrag(event, group, options, handle)" in config_card
    assert "startHistoryConfigGroupMouseDrag(event, group, options, handle)" in config_card
    assert "startHistoryConfigGroupTouchDrag(event, group, options, handle)" in config_card
    assert "beginHistoryConfigGroupDrag(event, group, options)" in config_card
    assert "finishHistoryDrag()" in config_card
    assert "historyConfigGroupOrderDragEnter(event, group, card, options)" in config_card
    assert "historyConfigGroupOrderDragLeave(event, group, card)" in config_card
    assert "dropHistoryConfigGroupToSort(event, group, options)" in config_card
    assert "reorderHistoryConfigGroupValue" in source
    assert "function ensureHistoryConfigGroupDropPreview" in drag_helpers
    assert "function placeHistoryConfigGroupDropPreview" in drag_helpers
    assert "function removeHistoryConfigGroupDropPreview" in drag_helpers
    assert "释放后插入到这里" in drag_helpers
    assert "placeHistoryConfigGroupDropPreview(element, historyState.historyConfigGroupSortState.dropPosition)" in source
    assert "preview.style.top" in drag_helpers
    assert "parent.appendChild(preview)" in drag_helpers
    assert "event.relatedTarget.closest('.history-config-group-card-list')" in drag_helpers
    for label in ("置顶", "上移", "下移", "置底"):
        assert f"createHistoryManagerGroupButton('{label}'" in config_card
    for hook in (
        "function startHistoryConfigGroupPointerDrag",
        "function startHistoryConfigGroupMouseDrag",
        "function startHistoryConfigGroupTouchDrag",
        "function finishHistoryConfigGroupPointerDrag",
        "function historyConfigGroupPointerTargetFromPoint",
        "function historyCollectionDropTargetFromPoint",
        "document.addEventListener('pointermove', drag.onMove, { passive: false })",
        "document.addEventListener('pointerup', drag.onUp, { passive: false })",
        "document.addEventListener('pointercancel', drag.onCancel, { passive: false })",
        "document.addEventListener('mousemove', drag.onMouseMove, { passive: false })",
        "document.addEventListener('mouseup', drag.onMouseUp, { passive: false })",
        "document.addEventListener('touchmove', drag.onTouchMove, { passive: false })",
        "document.addEventListener('touchend', drag.onTouchEnd, { passive: false })",
        "document.addEventListener('touchcancel', drag.onTouchCancel, { passive: false })",
    ):
        assert hook in drag_helpers

    assert "history-collection-drag-handle" in collection_card
    assert "拖拽分组调整顺序" in collection_card
    assert "dragHandle.draggable = true;" in collection_card
    assert "dragHandle.addEventListener('pointerdown'" in collection_card
    assert "beginHistoryCollectionDrag(event, collection)" in collection_card
    assert "startHistoryCollectionPointerDrag(event, collection, allCollections, dragHandle)" in collection_card
    assert "finishHistoryCollectionDrag()" in collection_card
    assert "historyCollectionOrderDragEnter(event, collection, card)" in collection_card
    assert "dropHistoryCollectionToSort(event, collection, allCollections)" in collection_card
    assert "reorderHistoryCollectionValue(source, target, position, allCollections)" in source
    for label in ("置顶", "上移", "下移", "置底"):
        assert f"createHistoryManagerGroupButton('{label}'" in collection_card

    assert "dragenter" in collection_card
    assert "dragover" in collection_card
    assert "dragleave" in collection_card
    assert "dropHistoryTasksToCollection(event, collection.value || '', collection.label)" in collection_card
    assert "collection.value || '__ungrouped__'" in collection_card
    assert "applyHistoryTaskIdsToCollection(taskIds, clean, { clearSelection: true })" in drag_helpers
    assert "historyDraggedTasksAlreadyInCollection(taskIds, clean)" in drag_helpers
    assert "selectedHistoryCollectionKey = clean ? `collection:${clean}` : HISTORY_UNGROUPED_COLLECTION_KEY" in drag_helpers
    for hook in (
        "function startHistoryCollectionPointerDrag",
        "function startHistoryCollectionMouseDrag",
        "function startHistoryCollectionTouchDrag",
        "function finishHistoryCollectionPointerDrag",
        "function historyCollectionEventPoint",
        "function historyCollectionPointerTargetFromPoint",
        "function autoScrollHistoryCollectionPointerDrag",
        "document.addEventListener('pointermove', drag.onMove, { passive: false })",
        "document.addEventListener('pointerup', drag.onUp, { passive: false })",
        "document.addEventListener('pointercancel', drag.onCancel, { passive: false })",
        "document.addEventListener('mousemove', drag.onMouseMove, { passive: false })",
        "document.addEventListener('mouseup', drag.onMouseUp, { passive: false })",
        "document.addEventListener('touchmove', drag.onTouchMove, { passive: false })",
        "document.addEventListener('touchend', drag.onTouchEnd, { passive: false })",
        "document.addEventListener('touchcancel', drag.onTouchCancel, { passive: false })",
        "document.addEventListener('keydown', drag.onKeydown)",
    ):
        assert hook in drag_helpers
    assert "dragHandle.addEventListener('mousedown'" in collection_card
    assert "dragHandle.addEventListener('touchstart'" in collection_card
    assert "startHistoryCollectionMouseDrag(event, collection, allCollections, dragHandle)" in collection_card
    assert "startHistoryCollectionTouchDrag(event, collection, allCollections, dragHandle)" in collection_card

    assert "history-drop-popover" in source
    assert "event.key === 'Escape'" in source
    assert "event.key === 'Enter'" in source
    assert "input.maxLength = 48;" in source
    assert "state.taskIds.length ? `${state.taskIds.length} 条任务归入新分组` : '新建分组'" in source
    assert "defaultHistoryCollectionName" in source
    assert "uniqueHistoryCollectionName" in source

    assert "/api/training/history/batch" in source
    assert "/api/collections/create-and-assign" not in source
    assert "/api/tasks/assign-collection" not in source

    for selector in (
        ".history-config-group-card.draggable",
        ".history-config-group-card.config-sort-active",
        ".history-config-group-card.config-sort-source",
        ".history-config-group-card-head",
        ".history-config-group-drop-preview",
        ".history-config-group-pointer-drag-active",
        ".history-drag-handle",
        ".history-current-group-content",
        ".history-collection-nav",
        ".history-collection-drag-handle",
        ".history-collection-card.nav-card",
        ".history-collection-card.drop-active",
        ".history-collection-card.sort-active",
        ".history-collections-workbench.collection-reordering",
        ".history-collection-pointer-drag-active",
        ".history-collection-drag-image-pointer",
        ".history-collection-create-btn",
        ".history-collections-workbench.dragging",
        ".history-collections-workbench.compact",
        ".history-compact-meta",
        ".history-more-actions",
        ".history-drop-popover",
        "prefers-reduced-motion",
    ):
        assert selector in css
    assert ".history-config-group-card.selected > .history-config-group-card-actions" not in css
    assert "--history-config-group-select-width: 18px;" in css
    assert "--history-config-group-handle-width: 28px;" in css
    assert "grid-template-columns: var(--history-config-group-select-width) var(--history-config-group-handle-width) minmax(0, 1fr);" in css
    config_card_head_css = _section(
        css,
        ".history-config-group-card-head {",
        ".history-config-group-card.draggable {",
    )
    assert "min-height: 32px;" in config_card_head_css
    assert "width: 100%;" in config_card_head_css
    config_card_actions_css = _section(
        css,
        ".history-config-group-card-head > .history-config-group-card-actions {",
        ".history-config-group-card.single-task .history-config-group-card-head > .history-config-group-card-actions {",
    )
    assert "position: absolute;" in config_card_actions_css
    single_task_actions_css = _section(
        css,
        ".history-config-group-card.single-task .history-config-group-card-head > .history-config-group-card-actions {",
        ".history-config-group-card-main strong",
    )
    assert "position: absolute;" not in single_task_actions_css
    assert "transform: translateY(-50%);" in single_task_actions_css
    assert ".history-config-group-card:focus-within > .history-config-group-card-actions" not in css
    assert ".history-manager-row:focus-within .history-row-actions" not in css
    assert ".history-config-group-card-head > .history-config-group-card-actions:focus-within" in css
    assert ".history-row-actions:focus-within" in css
    config_group_select_css = _section(
        css,
        ".history-config-group-select {",
        ".history-config-group-task-list {",
    )
    assert "font-size: 0;" in config_group_select_css
    assert "gap: 0;" in config_group_select_css
    assert "justify-content: center;" in config_group_select_css
    assert ".history-config-group-select input" in css
    config_group_handle_css = _section(
        css,
        ".history-config-group-card-head > .history-drag-handle {",
        ".history-config-group-card.single-task .history-config-group-card-head > .history-drag-handle {",
    )
    assert "min-width: 0;" in config_group_handle_css
    assert "width: 100%;" in config_group_handle_css
    single_task_handle_css = _section(
        css,
        ".history-config-group-card.single-task .history-config-group-card-head > .history-drag-handle {",
        "\n.history-drag-handle {",
    )
    assert "min-width: 0;" in single_task_handle_css


def test_history_curve_data_helpers_normalize_filter_and_downsample() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for history curve data checks")
    script = r"""
import {
    HISTORY_CURVE_RENDER_POINT_LIMIT,
    createHistoryCurveMetrics,
    historyCurveDisplayPoints,
    historyCurveFilteredPoints,
    historyCurveNormalizePoint,
    historyCurveNormalizeRawMetricPoint,
    historyCurveSmoothPoints,
} from './web/static/js/features/history-detail/curve/data.js';

const metrics = createHistoryCurveMetrics((value) => `lr:${value}`);
const normalized = [
    historyCurveNormalizePoint({ step: 3, loss: '0.9', learningRate: '0.0003' }, 0, false),
    historyCurveNormalizePoint({ step: 1, learning_rate: '0.0001' }, 1, false),
    historyCurveNormalizePoint({ step: 2, loss: 'bad', lr: 'bad' }, 2, false),
].filter(Boolean);
const raw = [
    historyCurveNormalizeRawMetricPoint({ step: 1, learningRate: '0.0001' }),
    historyCurveNormalizeRawMetricPoint({ step: 2, learning_rate: '0.0002' }),
    historyCurveNormalizeRawMetricPoint({ step: 3, lr: '0.0003' }),
];
const filtered = historyCurveFilteredPoints(normalized, {
    rangeMode: 'custom',
    customStart: '2',
    customEnd: '3',
});
const smooth = historyCurveSmoothPoints(filtered, 2, metrics);
const many = Array.from(
    { length: HISTORY_CURVE_RENDER_POINT_LIMIT + 25 },
    (_, index) => ({ step: index, index }),
);
const display = historyCurveDisplayPoints(many);

console.log(JSON.stringify({
    normalized: normalized.map((point) => ({
        step: point.step,
        loss: point.loss,
        lr: point.lr,
        index: point.index,
    })),
    raw: raw.map((point) => point.lr),
    filteredSteps: filtered.map((point) => point.step),
    smooth: smooth.map((point) => ({
        step: point.step,
        smoothLoss: point.smoothLoss ?? null,
        smoothLr: point.smoothLr ?? null,
    })),
    display: {
        limit: HISTORY_CURVE_RENDER_POINT_LIMIT,
        count: display.length,
        first: display[0].step,
        last: display[display.length - 1].step,
        unique: new Set(display.map((point) => point.index)).size,
    },
}));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    assert payload == {
        "normalized": [
            {"step": 3, "loss": 0.9, "lr": 0.0003, "index": 0},
            {"step": 1, "loss": None, "lr": 0.0001, "index": 1},
        ],
        "raw": [0.0001, 0.0002, 0.0003],
        "filteredSteps": [3],
        "smooth": [
            {"step": 3, "smoothLoss": 0.9, "smoothLr": 0.0003},
        ],
        "display": {
            "limit": 1600,
            "count": 1600,
            "first": 0,
            "last": 1624,
            "unique": 1600,
        },
    }


def test_history_detail_state_aliases_and_resume_labels() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for history detail state checks")
    script = r"""
import {
    createHistoryDetailState,
    normalizeHistoryDetailTab,
    resetHistoryDetailViewState,
    setHistoryDetailTab,
} from './web/static/js/features/history-detail/state.js';
import {
    resumeCheckpointRemainingText,
    resumeCheckpointProgressText,
} from './web/static/js/features/history-detail/resume/state.js';

const state = createHistoryDetailState();
state.currentPayload = { task: { id: 'task-1' } };
state.returnState = { mode: 'history' };
state.mainTaskReturn = { group: 'main' };
state.curve.hoverStep = 42;

const aliases = [
    normalizeHistoryDetailTab('resume'),
    normalizeHistoryDetailTab('chart'),
    normalizeHistoryDetailTab('samples'),
    normalizeHistoryDetailTab('paths'),
    normalizeHistoryDetailTab('missing'),
];
const selected = setHistoryDetailTab(state, 'config');
resetHistoryDetailViewState(state);

console.log(JSON.stringify({
    aliases,
    selected,
    reset: {
        detailTab: state.detailTab,
        currentPayload: state.currentPayload,
        returnState: state.returnState,
        mainTaskReturn: state.mainTaskReturn,
        hoverStep: state.curve.hoverStep,
    },
    labels: [
        resumeCheckpointProgressText({ epoch: 2, step: 30 }),
        resumeCheckpointProgressText({}),
        resumeCheckpointRemainingText({ step: 30, target_total_steps: 100 }),
        resumeCheckpointRemainingText({
            epoch: 1,
            step: 30,
            estimate_error: 'missing runtime',
        }),
    ],
}));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    assert payload == {
        "aliases": ["overview", "analysis", "preview", "config_files", "overview"],
        "selected": "config_files",
        "reset": {
            "detailTab": "overview",
            "currentPayload": None,
            "returnState": None,
            "mainTaskReturn": None,
            "hoverStep": None,
        },
        "labels": [
            "Epoch 2 / Step 30",
            "步数未知",
            "已训练到 Step 30 / 目标 Step 100 / 剩余 70",
            "Epoch 1 / Step 30 / 无法确认剩余步数",
        ],
    }


def test_history_detail_overview_uses_full_copyable_paths_and_resume_weights() -> None:
    overview_source = _frontend_module_text("js/features/history-detail/overview.js")
    resume_source = _frontend_module_text("js/features/history-detail/resume/detail.js")
    ui_source = _frontend_module_text("js/features/history-detail/ui.js")
    css = STYLE_CSS.read_text(encoding="utf-8")

    overview = _section(overview_source, "function renderHistoryDetailOverview", "function renderHistoryDetailProgress")
    progress = _section(overview_source, "function renderHistoryDetailProgress", "function renderHistoryDetailPathSummary")
    path_summary = _section(overview_source, "function renderHistoryDetailPathSummary", "return { renderHistoryDetailOverview")
    curve_index_source = _frontend_module_text("js/features/history-detail/curve/index.js")
    resume = _section(resume_source, "function renderHistoryDetailResume", "function renderResumeDiagnosticBlock")
    weights = _section(resume_source, "function renderHistoryResumeWeightOptions", "function formatDiagnosticBool")
    row_helpers = _section(ui_source, "export function historyDetailRow", "export function historyDetailEmptyText")

    assert "icon.className = `metric-icon metric-icon-${iconName}`" in overview
    assert "['平均速度', formatHistoryAverageSpeed(task), 'gauge']" in overview
    assert "['训练总时间', formatHistoryTaskDuration(task), 'time']" in overview
    assert "['训练精度', formatHistoryTrainingPrecision(payload.config_toml), 'chip']" in overview
    assert "['训练变体', formatHistoryTrainingVariant(task, payload.config_toml), 'chip']" in overview
    assert "['预处理精度', formatHistoryPreprocessPrecision(payload.config_toml), 'chip']" in overview
    assert "['块交换精度', formatHistoryBlockSwapPrecision(payload.config_toml), 'chip']" in overview
    assert "function formatHistoryAverageSpeed(record)" in overview_source
    assert "function formatHistoryTaskDuration(record)" in overview_source
    assert "function formatHistoryTrainingPrecision(configText)" in overview_source
    assert "function formatHistoryTrainingVariant(task, configText)" in overview_source
    assert "function formatHistoryPreprocessPrecision(configText)" in overview_source
    assert "function formatHistoryBlockSwapPrecision(configText)" in overview_source
    assert "function readConfigString(configText, key)" in overview_source
    assert "precision_preference" in overview_source
    assert "mixed_precision" in overview_source
    assert "preprocess_precision_preference" in overview_source
    assert "block_swap_transfer_dtype" in overview_source
    assert "use_loha" in overview_source
    assert "use_lokr" in overview_source
    assert "use_chimera_hydra" in overview_source
    assert "use_timestep_mask" in overview_source
    assert "hasSnapshot" in overview_source
    assert "无法生成配置快照" in overview_source
    assert "replace(/-8gb$/" in overview_source
    assert "ctx.format.formatDuration" in overview_source
    assert "muted: taskFinished && ['队列', '续训'].includes(label) && value === '-'" in overview
    assert "'训练精度'" in overview_source
    assert "'训练变体'" in overview_source
    assert "'预处理精度'" in overview_source
    assert "'块交换精度'" in overview_source
    assert "section.classList.toggle('is-complete', finished);" in progress
    assert "historyCurveStatGroup('速度组'" in curve_index_source
    assert "['平均速度', formatHistoryAverageSpeed(task)]" in curve_index_source
    assert "['采样范围', formatAverageSpeedStepRange(task)]" in curve_index_source

    assert "historyDetailRunRoot(task)" in path_summary
    assert "return normalizedHistoryDetailPath(task.run_dir_abs || task.run_dir || '');" in ui_source
    assert "'运行根目录'" in path_summary
    assert "relativeHistoryDetailPath(value, rootPath)" not in path_summary
    assert "copyValue: value" in path_summary
    assert "export function relativeHistoryDetailPath" in ui_source
    assert "export function selectAllTextOnDoubleClick" in ui_source
    assert "range.selectNodeContents(el)" in ui_source
    assert "selectAllTextOnDoubleClick(val)" in row_helpers
    assert "row.appendChild(helpers.copyButton(options.copyValue" in row_helpers

    assert "controls.className = 'history-resume-control-row';" in resume
    assert "fullResume.append(controls, summary);" in resume
    assert "resumeCheckpointRemainingText(selected)" in resume
    assert "selected.resume_available !== false" in resume
    assert "resumeSummaryLine('不可用原因', selected.unavailable_reason)" in resume
    assert "resumeSummaryLine('步数估算', `无法确认剩余步数: ${selected.estimate_error}`)" in resume
    assert "checkpointWeightPaths.has(String(weightPath || '').trim())" in weights
    assert "item.inspect_status === 'ok'" in weights
    assert "item.inspect_compatible !== false" in weights
    assert "审查未通过" in weights
    assert "useBtn.disabled = !canUseWeightDirectly;" in weights
    assert "缺少对应的 checkpoint-state/train_state.json" in weights
    assert "name.textContent = fileNameFromPath(item.name || weightPath)" in weights
    assert "info.append(name, meta);" in weights
    assert "info.append(name, path, meta);" not in weights
    assert "history-resume-weight-actions" in weights
    assert "historyDetailCopyButton(weightPath" in weights

    assert ".history-detail-progress.is-complete .history-detail-progress-bar span" in css
    assert ".history-detail-stat .metric-icon-time::before" in css
    assert ".history-curve-stat-group.speed" in css
    assert ".history-detail-path-summary .history-detail-path-root" in css
    assert ".history-detail-select-all" in css
    assert ".history-detail-copy-btn" in css
    assert ".history-resume-control-row" in css
    assert ".history-resume-hint.warning" in css
    assert ".history-resume-weight-actions" in css


def test_history_detail_config_files_are_tool_ready() -> None:
    legacy_source = _anima_app_container_text()
    config_files_source = _frontend_module_text("js/features/history-detail/config-files.js")
    html = INDEX_HTML.read_text(encoding="utf-8")
    css = STYLE_CSS.read_text(encoding="utf-8")

    config_files = _section(config_files_source, "function renderHistoryDetailConfig", "function renderHistoryDetailConfigFiles")
    path_items = _section(legacy_source, "function runtimePathItems", "function historyStateLabel")

    assert "history-config-viewer" in config_files
    assert "history-config-toolbar" in config_files
    assert "history-config-search" in config_files
    assert "renderHistoryConfigCode(pre, content, searchText, currentMatch)" in config_files
    assert "history-config-token-key" in config_files
    assert "history-config-token-path" in config_files
    assert "history-config-search-hit current" not in config_files
    assert "historyConfigMatchCount(content, searchText)" in config_files
    assert "downloadBlob(new Blob([content], { type: 'text/plain;charset=utf-8' }), filename)" in config_files
    assert "history-detail-file-browser" in config_files
    assert "history-file-root" in config_files
    assert "historyDetailFileRow(task, label, value, artifactKey)" in config_files
    assert "relativeHistoryDetailPath(rawValue, rootPath)" not in config_files
    assert "val.textContent = rawValue" in config_files
    assert "selectAllTextOnDoubleClick(val)" in config_files
    assert "deps.historyArtifactUrl(task, artifactKey)" in config_files
    assert "deps.historyArtifactUrl(task, artifactKey, { download: true })" in config_files
    assert "function makeHistoryArtifactUrl" not in legacy_source
    assert "historyArtifactUrl: makeHistoryArtifactUrl" in legacy_source
    assert "makeHistoryArtifactUrl(task, artifactKey" in _frontend_module_text(
        "js/features/anima-app/helpers/history-artifacts.js"
    )
    assert "choiceHelp, help" in _frontend_module_text("js/config/catalog.js")
    assert "Object.assign(globalThis, ctx.catalog);" not in legacy_source
    for catalog_user in (
        "js/features/app-shell/startup.js",
        "js/features/config-form/index.js",
        "js/features/config-form/group-entry.js",
        "js/features/config-form/resource-values.js",
        "js/features/config-form/field-rows.js",
        "js/features/dataset-editor/dataset-editor-panel.js",
        "js/features/dataset-editor/inline-help.js",
        "js/features/dataset-editor/row.js",
        "js/features/dataset-editor/row-fields.js",
        "js/features/config-form/choice-guide-ui.js",
        "js/features/dataset-editor/mutations.js",
        "js/features/config-form/form-fields-ui.js",
        "js/features/dataset-editor/row-settings-basic.js",
        "js/features/sample-prompts/row-ui.js",
        "js/features/config-form/config-value-collector.js",
        "js/features/config-form/adapter-field-state.js",
        "js/features/anima-app/chunks/19-current-sample-prompt-text.js",
        "js/features/anima-app/chunks/21-update-toml-selection-ui.js",
        "js/features/anima-app/chunks/22-update-toml-action-state.js",
        "js/features/preflight-dialog/index.js",
        "js/features/live-log/index.js",
        "js/features/global-settings/settings.js",
        "js/features/app-shell/event-listeners-setup.js",
    ):
        source = _frontend_module_text(catalog_user)
        assert (
            "../../../config/catalog.js" in source
            or "../../config/catalog.js" in source
        ), catalog_user


    for artifact in (
        "'runtime-config'",
        "'original-config'",
        "'dataset-config'",
        "'logs'",
        "'metrics'",
        "'system'",
        "'config-snapshot'",
    ):
        assert artifact in path_items
    assert "const runDir = absolutePath(task.run_dir_abs || task.run_dir)" in path_items
    assert "function historyAbsolutePath(value, task = {}, basePath = '')" in path_items
    assert "function historyProjectRoot(task = {})" in path_items
    assert "project_root_abs" in path_items

    assert "module-bootstrap-" in html
    for selector in (
        ".history-config-viewer",
        ".history-config-toolbar",
        ".history-config-code.history-detail-pre",
        ".history-config-token-key",
        ".history-config-token-path",
        ".history-detail-file-browser",
        ".history-file-root",
        ".history-file-row",
        ".history-file-actions",
        ".history-detail-icon-btn",
        ".history-detail-select-all",
    ):
        assert selector in css
    assert "text-overflow: ellipsis;" not in _section(
        css,
        ".history-detail-path-row code,",
        ".history-detail-select-all {",
    )
    assert ".history-detail-config-files > .history-detail-section" in css
    assert ".history-detail-kv > div" in css
    assert ".history-detail-kv div" not in css

def test_history_collection_switch_uses_partial_refresh() -> None:
    """Switching collection nav should refresh left config panel without full workbench rebuild."""
    workbench = _frontend_module_text("js/features/history-list/collections-workbench.js")
    collection_card = _frontend_module_text("js/features/history-list/workbench-collection-card.js")
    config_card = _frontend_module_text("js/features/history-list/workbench-config-group-card.js")
    chunked = _frontend_module_text("js/features/history-list/chunked-render.js")
    fill = _frontend_module_text("js/features/history-list/workbench-chunk-fill.js")
    history_state = _frontend_module_text("js/features/anima-app/state/history-state.js")
    open_section = _section(collection_card, "export function createHistoryCollectionWorkbenchCard", "const head = document.createElement('div');")

    assert "expandedHistoryConfigGroupKeys: new Set()" in history_state
    assert "collapsedHistoryConfigGroupKeys: new Set()" in history_state
    assert "historyWorkbenchCollectionsCache: null" in history_state
    assert "export function selectHistoryCollectionInWorkbench" in workbench
    assert "export function refreshHistoryWorkbenchConfigPanel" in workbench
    assert "selectHistoryCollectionInWorkbench(collection.key);" in open_section
    assert "renderHistoryManager();" not in open_section
    assert "configOnly = false" in fill
    assert "const node = createNode(source[index], index);" in chunked
    assert "toggleHistoryConfigGroupExpanded" in config_card
    assert "createHistoryManagerGroupButton(" in config_card
    assert "history-config-group-collapse-summary" in config_card
    assert "isHistoryConfigGroupExpanded(group, options.collection)" in config_card
    assert "ensureLiveHistoryConfigGroupsExpanded" in workbench
    assert "historyConfigGroupHasLiveTasks" in workbench
    assert "监控中 · ${liveLabel}" in config_card
    assert "card.classList.toggle('is-live', hasLiveTasks)" in config_card


def test_history_workbench_cache_key_includes_live_task_fields() -> None:
    """Running task counters must bust the workbench collection cache so the history tab re-renders."""
    workbench = _frontend_module_text("js/features/history-list/collections-workbench.js")
    cache_section = _section(workbench, "function workbenchCollectionsCacheKey", "function getHistoryWorkbenchCollections")
    for token in (
        "task?.state || ''",
        "Number(task?.log_count || 0)",
        "Number(task?.metric_count || 0)",
        "Number(task?.updated_at || 0)",
        "ensureLiveHistoryConfigGroupsExpanded",
        "collapsedHistoryConfigGroupKeys",
    ):
        assert token in workbench
    assert "Number(task?.log_count || 0)" in cache_section
    assert "task?.state || ''" in cache_section


def test_sidebar_history_switch_avoids_full_list_rerender() -> None:
    """Sidebar task switching should reuse selection sync/cache instead of rebuilding the list every time."""
    task_dialogs = _frontend_module_text("js/features/history-list/task-dialogs.js")
    history_list = _frontend_module_text("js/features/history-list/list.js")
    history_state = _frontend_module_text("js/features/anima-app/state/history-state.js")
    task_item = _frontend_feature_text(
        "js/features/anima-app/chunks/33-create-history-task-item.js",
    )
    open_section = _section(task_dialogs, "async function openSidebarHistoryTask", "async function refreshHistoryView")

    assert "sidebarHistoryPayloadCache: new Map()" in history_state
    assert "sidebarHistoryRequestId: 0" in history_state
    assert "function syncRecentHistorySidebarSelection" in history_list
    assert "card.dataset.taskId" in task_item
    assert "syncRecentHistorySidebarSelection();" in open_section
    assert "renderTrainingHistoryList();" not in open_section
    assert "rememberSidebarHistoryPayload(id, payload);" in open_section
    assert "if (requestId !== historyState.sidebarHistoryRequestId) return;" in open_section
    assert "sidebarHistoryLogLines" in task_dialogs
    assert "SIDEBAR_HISTORY_LOG_RENDER_LIMIT" in task_dialogs
    assert "export function renderHistoryTask(payload, options = {})" in task_dialogs
    assert "renderLogOutputLines(logLines, { stickToBottom: options.stickLogsToBottom !== false });" in task_dialogs


def test_history_config_group_card_imports_storage_key() -> None:
    """Config-group cards must import historyCollectionStorageKey from workbench-order."""
    config_card = _frontend_module_text("js/features/history-list/workbench-config-group-card.js")
    order = _frontend_module_text("js/features/history-list/workbench-order.js")
    assert "export function historyCollectionStorageKey" in order
    assert "historyCollectionStorageKey," in config_card
    assert "from './workbench-order.js" in config_card
    assert "card.dataset.collectionKey = historyCollectionStorageKey(options.collection || '__all__');" in config_card


def test_history_workbench_renders_items_in_chunks() -> None:
    """Large history workbench lists should append cards in rAF chunks, not one blocking loop only."""
    helper = _frontend_module_text("js/features/history-list/chunked-render.js")
    workbench = _frontend_feature_text(
        "js/features/anima-app/chunks/27-render-history-collections-workbench.js",
        "js/features/history-list/collections-workbench.js",
    )

    assert "export function appendNodesInChunks" in helper
    assert "export function renderItemsInChunks" in helper
    assert "HISTORY_RENDER_CHUNK_SIZE" in helper
    assert "requestAnimationFrame" in helper
    assert "const node = createNode(source[index], index);" in helper

    assert "fillHistoryWorkbenchCardLists" in workbench
    fill_helper = _frontend_module_text("js/features/history-list/workbench-chunk-fill.js")
    assert "renderItemsInChunks" in fill_helper
    assert "historyWorkbenchRenderSignal" in workbench
    assert "createHistoryConfigGroupWorkbenchCard" in workbench
    assert "createHistoryCollectionWorkbenchCard" in workbench

    if not shutil.which("node"):
        return

    script = r"""
import { appendNodesInChunks, HISTORY_RENDER_CHUNK_SIZE } from './web/static/js/features/history-list/chunked-render.js';

class FakeNode {
  constructor(name='div') { this.name = name; this.children = []; }
  appendChild(node) { this.children.push(node); return node; }
}
class FakeFragment extends FakeNode {
  constructor() { super('fragment'); }
}

// Minimal DOM polyfill for pure chunk scheduling test
const parent = new FakeNode('parent');
globalThis.document = {
  createDocumentFragment: () => new FakeFragment(),
};
let frames = 0;
globalThis.requestAnimationFrame = (fn) => {
  frames += 1;
  setTimeout(fn, 0);
  return frames;
};

const nodes = Array.from({ length: 50 }, (_, i) => new FakeNode(`n${i}`));
const { done } = appendNodesInChunks(parent, nodes, { chunkSize: 10 });
await done;
const childCount = parent.children.reduce((sum, node) => sum + (node.children?.length || 0), parent.children.length);
// fragments count as children; total leaf nodes should be 50 across fragments
let leaves = 0;
for (const child of parent.children) {
  if (child.children?.length) leaves += child.children.length;
  else leaves += 1;
}
console.log(JSON.stringify({ leaves, frames, chunk: HISTORY_RENDER_CHUNK_SIZE }));
"""
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    payload = json.loads(proc.stdout)
    assert payload["leaves"] == 50
    assert payload["frames"] >= 1
